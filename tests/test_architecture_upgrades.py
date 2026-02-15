#!/usr/bin/env python3
"""
Tests for CineCast architecture upgrades.

Covers:
- JSON robust repair (repair_json_array, salvage_json_entries)
- Atomic file writes (atomic_json_write)
- Narrator merging (merge_consecutive_narrators)
- Context sliding window
- Group-by-voice rendering indices
- Dynamic pause logic in CinematicPackager
- Audacity multi-track export
"""

import json
import os
import sys
import tempfile

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import (
    atomic_json_write,
    merge_consecutive_narrators,
    repair_json_array,
    salvage_json_entries,
    LLMScriptDirector,
)

# mlx_tts_engine imports mlx which is macOS-only; import only the pure-Python
# helper we need for testing.
try:
    from modules.mlx_tts_engine import group_indices_by_voice_type
except ImportError:
    # Fallback: import the function source directly to avoid mlx dependency
    import importlib.util
    import types

    _spec = importlib.util.spec_from_file_location(
        "_mlx_helpers",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "modules", "mlx_tts_engine.py"),
    )
    _source = open(_spec.origin, "r").read()
    # Extract just the function we need without executing the full module
    _ns: dict = {"defaultdict": __import__("collections").defaultdict}
    exec(compile(
        "from collections import defaultdict\nfrom typing import List, Dict\n" +
        _source[_source.index("def group_indices_by_voice_type"):
                _source.index("\nclass ")],
        "<mlx_helpers>",
        "exec",
    ), _ns)
    group_indices_by_voice_type = _ns["group_indices_by_voice_type"]

from modules.cinematic_packager import (
    CinematicPackager,
    CROSS_SPEAKER_PAUSE_MS,
    SAME_SPEAKER_PAUSE_MS,
)


# ---------------------------------------------------------------------------
# P0-1: JSON Robust Repair
# ---------------------------------------------------------------------------

class TestRepairJsonArray:
    def test_valid_json_passthrough(self):
        raw = json.dumps([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "测试内容。"}
        ])
        result = repair_json_array(raw)
        assert result is not None
        assert len(result) == 1
        assert result[0]["content"] == "测试内容。"

    def test_truncated_json_repair(self):
        """Simulate a truncated JSON array (missing closing bracket)."""
        raw = '[{"type": "narration", "speaker": "narrator", "gender": "male", "emotion": "平静", "content": "第一句。"}, {"type": "dialogue", "speaker": "老渔夫", "gender": "male", "emotion": "沧桑", "content": "你好"'
        result = repair_json_array(raw)
        # Should recover at least the first complete entry
        assert result is not None
        assert len(result) >= 1

    def test_missing_comma_repair(self):
        """JSON with missing comma between objects."""
        raw = '''[{"type": "narration", "speaker": "narrator", "gender": "male", "emotion": "平静", "content": "你好。"}
        {"type": "dialogue", "speaker": "老渔夫", "gender": "male", "emotion": "沧桑", "content": "再见。"}]'''
        result = repair_json_array(raw)
        assert result is not None
        assert len(result) >= 1

    def test_completely_broken_returns_none(self):
        raw = "This is not JSON at all, just random text."
        result = repair_json_array(raw)
        assert result is None

    def test_empty_string(self):
        result = repair_json_array("")
        assert result is None


class TestSalvageJsonEntries:
    def test_extracts_entries_from_broken_json(self):
        raw = '''{"type": "narration", "speaker": "narrator", "gender": "male", "emotion": "平静", "content": "一句话。"} GARBAGE {"type": "dialogue", "speaker": "张三", "gender": "male", "emotion": "激动", "content": "你好"}'''
        result = salvage_json_entries(raw)
        assert result is not None
        assert len(result) == 2
        assert result[0]["speaker"] == "narrator"
        assert result[1]["speaker"] == "张三"

    def test_loose_pattern_fallback(self):
        """When the strict pattern fails, try the loose one."""
        raw = '"speaker": "narrator", "content": "fallback text"'
        result = salvage_json_entries(raw)
        assert result is not None
        assert len(result) >= 1
        assert result[0]["content"] == "fallback text"


# ---------------------------------------------------------------------------
# P0-2: Atomic File Writes
# ---------------------------------------------------------------------------

class TestAtomicJsonWrite:
    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = [{"key": "value", "中文": "测试"}]
            atomic_json_write(path, data)

            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded == data

    def test_no_tmp_file_left_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            atomic_json_write(path, {"a": 1})
            # Only the final file should exist
            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert files[0] == "test.json"

    def test_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            atomic_json_write(path, {"version": 1})
            atomic_json_write(path, {"version": 2})

            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["version"] == 2


# ---------------------------------------------------------------------------
# P1-2: Narrator Merging
# ---------------------------------------------------------------------------

class TestMergeConsecutiveNarrators:
    def test_merges_same_emotion_narrators(self):
        script = [
            {"speaker": "narrator", "type": "narration", "emotion": "平静",
             "content": "第一句。", "pause_ms": 600},
            {"speaker": "narrator", "type": "narration", "emotion": "平静",
             "content": "第二句。", "pause_ms": 1000},
        ]
        result = merge_consecutive_narrators(script)
        assert len(result) == 1
        assert result[0]["content"] == "第一句。第二句。"
        assert result[0]["pause_ms"] == 1000  # max of both

    def test_does_not_merge_different_emotions(self):
        script = [
            {"speaker": "narrator", "type": "narration", "emotion": "平静",
             "content": "平静句。", "pause_ms": 600},
            {"speaker": "narrator", "type": "narration", "emotion": "激动",
             "content": "激动句。", "pause_ms": 600},
        ]
        result = merge_consecutive_narrators(script)
        assert len(result) == 2

    def test_does_not_merge_dialogue(self):
        script = [
            {"speaker": "narrator", "type": "narration", "emotion": "平静",
             "content": "旁白。", "pause_ms": 600},
            {"speaker": "老渔夫", "type": "dialogue", "emotion": "沧桑",
             "content": "对白。", "pause_ms": 600},
            {"speaker": "narrator", "type": "narration", "emotion": "平静",
             "content": "又一个旁白。", "pause_ms": 600},
        ]
        result = merge_consecutive_narrators(script)
        assert len(result) == 3

    def test_respects_max_chars(self):
        long_a = "a" * 500
        long_b = "b" * 500
        script = [
            {"speaker": "narrator", "type": "narration", "emotion": "平静",
             "content": long_a, "pause_ms": 600},
            {"speaker": "narrator", "type": "narration", "emotion": "平静",
             "content": long_b, "pause_ms": 600},
        ]
        result = merge_consecutive_narrators(script, max_chars=800)
        # 500 + 500 = 1000 > 800, should NOT merge
        assert len(result) == 2

    def test_empty_script(self):
        assert merge_consecutive_narrators([]) == []

    def test_does_not_merge_different_types(self):
        script = [
            {"speaker": "narrator", "type": "narration", "emotion": "平静",
             "content": "旁白。", "pause_ms": 600},
            {"speaker": "narrator", "type": "title", "emotion": "平静",
             "content": "标题", "pause_ms": 600},
        ]
        result = merge_consecutive_narrators(script)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# P1-3: Group-by-Voice Rendering
# ---------------------------------------------------------------------------

class TestGroupByVoiceType:
    def test_groups_narration_and_dialogue(self):
        script = [
            {"type": "narration", "speaker": "narrator"},
            {"type": "dialogue", "speaker": "老渔夫"},
            {"type": "narration", "speaker": "narrator"},
            {"type": "dialogue", "speaker": "年轻人"},
            {"type": "dialogue", "speaker": "老渔夫"},
        ]
        groups = group_indices_by_voice_type(script)
        assert groups["narration"] == [0, 2]
        assert groups["dialogue:老渔夫"] == [1, 4]
        assert groups["dialogue:年轻人"] == [3]

    def test_title_and_subtitle(self):
        script = [
            {"type": "title", "speaker": "narrator"},
            {"type": "subtitle", "speaker": "narrator"},
            {"type": "narration", "speaker": "narrator"},
        ]
        groups = group_indices_by_voice_type(script)
        assert "title" in groups
        assert "subtitle" in groups
        assert "narration" in groups

    def test_empty_script(self):
        groups = group_indices_by_voice_type([])
        assert groups == {}


# ---------------------------------------------------------------------------
# P2-1: Dynamic Pause Logic
# ---------------------------------------------------------------------------

class TestDynamicPauses:
    def test_pause_constants_defined(self):
        assert CROSS_SPEAKER_PAUSE_MS == 500
        assert SAME_SPEAKER_PAUSE_MS == 250

    def test_packager_initializes_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            packager = CinematicPackager(tmpdir)
            assert packager._speaker_tracks == {}
            assert packager._labels == []
            assert packager._timeline_ms == 0


# ---------------------------------------------------------------------------
# P2-2: Audacity Export
# ---------------------------------------------------------------------------

class TestAudacityExport:
    def test_export_returns_none_when_no_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            packager = CinematicPackager(tmpdir)
            result = packager.export_audacity()
            assert result is None

    def test_export_creates_zip_with_tracks(self):
        """Test that export_audacity creates a valid ZIP with stems and labels."""
        import zipfile
        from pydub import AudioSegment

        with tempfile.TemporaryDirectory() as tmpdir:
            packager = CinematicPackager(tmpdir)
            
            # Simulate data that process_from_cache would populate
            silence = AudioSegment.silent(duration=1000)
            packager._speaker_tracks["narrator"] = silence
            packager._speaker_tracks["老渔夫"] = silence
            packager._labels = [
                {"start_ms": 0, "end_ms": 1000, "speaker": "narrator", "text": "测试旁白"},
                {"start_ms": 1500, "end_ms": 2500, "speaker": "老渔夫", "text": "测试对白"},
            ]

            zip_path = packager.export_audacity()
            assert zip_path is not None
            assert os.path.exists(zip_path)

            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                assert "narrator.wav" in names
                assert "老渔夫.wav" in names
                assert "labels.txt" in names

                labels_content = zf.read("labels.txt").decode("utf-8")
                assert "narrator" in labels_content
                assert "老渔夫" in labels_content


# ---------------------------------------------------------------------------
# P1-1: Context Sliding Window
# ---------------------------------------------------------------------------

class TestContextSlidingWindow:
    def test_initial_state_empty(self):
        """Director starts with empty context."""
        director = LLMScriptDirector()
        assert director._prev_characters == []
        assert director._prev_tail_entries == []


# ---------------------------------------------------------------------------
# LLM Director - Validate Script Elements
# ---------------------------------------------------------------------------

class TestValidateScriptElements:
    def test_fills_missing_fields(self):
        director = LLMScriptDirector()
        incomplete = [
            {"content": "测试内容"},  # Missing type, speaker, gender, emotion
        ]
        result = director._validate_script_elements(incomplete)
        assert len(result) == 1
        assert result[0]["type"] == "narration"
        assert result[0]["speaker"] == "narrator"
        assert result[0]["gender"] == "unknown"
        assert result[0]["emotion"] == "平静"

    def test_skips_non_dict(self):
        director = LLMScriptDirector()
        mixed = [
            "not a dict",
            {"type": "narration", "speaker": "narrator", "content": "OK"},
        ]
        result = director._validate_script_elements(mixed)
        assert len(result) == 1

    def test_preserves_valid_elements(self):
        director = LLMScriptDirector()
        valid = [
            {"type": "dialogue", "speaker": "张三", "content": "你好",
             "gender": "male", "emotion": "激动"},
        ]
        result = director._validate_script_elements(valid)
        assert len(result) == 1
        assert result[0]["emotion"] == "激动"


# ---------------------------------------------------------------------------
# Fallback Regex Parse
# ---------------------------------------------------------------------------

class TestFallbackRegexParse:
    def test_detects_title(self):
        director = LLMScriptDirector()
        result = director._fallback_regex_parse("第一章 风雪来袭")
        assert len(result) >= 1
        assert result[0]["type"] == "title"

    def test_detects_dialogue(self):
        director = LLMScriptDirector()
        result = director._fallback_regex_parse('"你好吗？"他说。')
        assert len(result) >= 1
        # Should detect dialogue due to quotes
        has_dialogue = any(e["type"] == "dialogue" for e in result)
        assert has_dialogue

    def test_defaults_to_narration(self):
        director = LLMScriptDirector()
        result = director._fallback_regex_parse("远处的灯塔开始旋转。")
        assert len(result) >= 1
        assert result[0]["type"] == "narration"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
