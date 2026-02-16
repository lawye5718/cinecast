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
from modules.asset_manager import AssetManager

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
    with open(_spec.origin, "r") as _fh:
        _source = _fh.read()
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

    def test_fixes_none_speaker(self):
        """When speaker is explicitly None, it should be fixed to 'narrator'."""
        director = LLMScriptDirector()
        elements = [
            {"type": "narration", "speaker": None, "content": "测试",
             "gender": "male", "emotion": "平静"},
        ]
        result = director._validate_script_elements(elements)
        assert len(result) == 1
        assert result[0]["speaker"] == "narrator"

    def test_fixes_none_gender(self):
        """When gender is explicitly None, it should be fixed to 'unknown'."""
        director = LLMScriptDirector()
        elements = [
            {"type": "narration", "speaker": "narrator", "content": "测试",
             "gender": None, "emotion": "平静"},
        ]
        result = director._validate_script_elements(elements)
        assert len(result) == 1
        assert result[0]["gender"] == "unknown"

    def test_fixes_both_none_speaker_and_gender(self):
        """When both speaker and gender are None, both should be fixed."""
        director = LLMScriptDirector()
        elements = [
            {"type": "narration", "speaker": None, "content": "测试",
             "gender": None, "emotion": "平静"},
        ]
        result = director._validate_script_elements(elements)
        assert len(result) == 1
        assert result[0]["speaker"] == "narrator"
        assert result[0]["gender"] == "unknown"


# ---------------------------------------------------------------------------
# LLM Dict-Format Tolerance
# ---------------------------------------------------------------------------

class TestLLMDictFormatTolerance:
    """Tests for _request_ollama handling of dict responses from LLM."""

    def _make_director_with_mock_response(self, json_content):
        """Create a director where _request_ollama returns a mocked HTTP response."""
        import unittest.mock as mock

        director = LLMScriptDirector()
        fake_resp = mock.MagicMock()
        fake_resp.status_code = 200
        fake_resp.raise_for_status = mock.MagicMock()
        fake_resp.json.return_value = {
            "message": {"content": json.dumps(json_content, ensure_ascii=False)}
        }

        with mock.patch("modules.llm_director.requests.post", return_value=fake_resp):
            return director._request_ollama("任意文本")

    def test_name_content_dict_converted_to_narration(self):
        """LLM returns {"name": "第一章 风雪", "content": "原文..."} — should become a single narration."""
        result = self._make_director_with_mock_response(
            {"name": "第一章 风雪", "content": "夜幕降临，港口的灯火开始闪烁。"}
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "narration"
        assert result[0]["speaker"] == "narrator"
        assert result[0]["content"] == "夜幕降临，港口的灯火开始闪烁。"

    def test_dict_containing_list_extracted(self):
        """LLM returns {"script": [...]} — should extract the inner list."""
        inner = [
            {"type": "narration", "speaker": "narrator", "content": "测试内容。",
             "gender": "male", "emotion": "平静"},
        ]
        result = self._make_director_with_mock_response({"script": inner})
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["content"] == "测试内容。"

    def test_valid_array_passthrough(self):
        """LLM returns a proper JSON array — should pass through normally."""
        arr = [
            {"type": "narration", "speaker": "narrator", "content": "正常。",
             "gender": "male", "emotion": "平静"},
        ]
        result = self._make_director_with_mock_response(arr)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_single_object_with_type_and_content_wrapped(self):
        """LLM returns {"type": "narration", "speaker": "narrator", "content": "..."} — should wrap in list."""
        result = self._make_director_with_mock_response(
            {"type": "narration", "speaker": "narrator", "content": "第二章 1976年的故事。",
             "gender": "male", "emotion": "平静"}
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "narration"
        assert result[0]["speaker"] == "narrator"
        assert result[0]["content"] == "第二章 1976年的故事。"

    def test_single_object_with_only_type_wrapped(self):
        """LLM returns a dict with 'type' but no 'name' — should wrap in list."""
        result = self._make_director_with_mock_response(
            {"type": "dialogue", "speaker": "老渔夫", "content": "你相信命运吗？",
             "gender": "male", "emotion": "沧桑"}
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "dialogue"
        assert result[0]["content"] == "你相信命运吗？"

    def test_dict_without_list_or_name_raises(self):
        """LLM returns a dict without any recognisable structure — should raise."""
        with pytest.raises(RuntimeError, match="非预期的 JSON 结构"):
            self._make_director_with_mock_response({"random_key": "random_value"})


# ---------------------------------------------------------------------------
# Error Raising on LLM Failure (no regex fallback)
# ---------------------------------------------------------------------------

class TestNoRegexFallback:
    def test_fallback_regex_parse_removed(self):
        """Ensure _fallback_regex_parse method no longer exists."""
        director = LLMScriptDirector()
        assert not hasattr(director, '_fallback_regex_parse')

    def test_request_ollama_raises_on_connection_error(self):
        """When Ollama is unreachable, _request_ollama should raise RuntimeError."""
        director = LLMScriptDirector(ollama_url="http://127.0.0.1:19999")
        with pytest.raises(RuntimeError, match="Ollama 解析失败"):
            director._request_ollama("测试文本")

    def test_parse_text_to_script_raises_on_empty(self):
        """parse_text_to_script should raise RuntimeError when result is empty."""
        director = LLMScriptDirector()
        # Monkey-patch _request_ollama to return an empty list
        director._request_ollama = lambda text_chunk, context=None: []
        with pytest.raises(RuntimeError, match="剧本解析结果为空"):
            director.parse_text_to_script("测试文本")


# ---------------------------------------------------------------------------
# Micro-chunking Fallback Logic
# ---------------------------------------------------------------------------

class TestMicroChunkFallback:
    def test_hard_cut_fallback_for_no_punctuation(self):
        """Content without any punctuation should still produce chunks."""
        director = LLMScriptDirector()
        # Monkey-patch parse_text_to_script to return a single element with no
        # Chinese punctuation (pure English / no separable marks).
        long_content = "A" * 180  # 180 chars, no Chinese punctuation
        director.parse_text_to_script = lambda text: [
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": long_content}
        ]
        result = director.parse_and_micro_chunk("test", chapter_prefix="ch01")
        # Should produce at least one chunk (not lose content)
        assert len(result) > 0
        # Total content should be preserved
        total_content = "".join(item["content"] for item in result)
        assert total_content == long_content

    def test_empty_content_units_skipped(self):
        """Script units with empty content should be skipped without errors."""
        director = LLMScriptDirector()
        director.parse_text_to_script = lambda text: [
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": ""},
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "有内容的句子。"},
        ]
        result = director.parse_and_micro_chunk("test", chapter_prefix="ch01")
        # Only the non-empty unit should produce chunks
        assert len(result) >= 1
        assert all(item["content"].strip() for item in result)

    def test_special_symbols_only_hard_cut(self):
        """Content with only special symbols that regex can't split should be hard-cut."""
        director = LLMScriptDirector()
        # Content of special symbols without Chinese punctuation
        special_content = "★☆◆◇■□▲△○●" * 10  # 100 special chars
        director.parse_text_to_script = lambda text: [
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": special_content}
        ]
        result = director.parse_and_micro_chunk("test", chapter_prefix="ch01")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# AssetManager Config Loading
# ---------------------------------------------------------------------------

class TestAssetManagerConfigLoading:
    def test_loads_voice_reference_from_config(self):
        """AssetManager should load voice_reference from audio_assets_config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "voice_reference": {
                    "narrator": {
                        "acoustic_description": "test narrator voice"
                    },
                    "male_default": {
                        "acoustic_description": "test male voice"
                    },
                    "female_default": {
                        "acoustic_description": "test female voice"
                    }
                },
                "audio_processing": {
                    "target_sample_rate": 44100
                }
            }
            config_path = os.path.join(tmpdir, "audio_assets_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Create AssetManager with asset_dir as a subdirectory so
            # the config is found at ../audio_assets_config.json
            asset_dir = os.path.join(tmpdir, "assets")
            os.makedirs(asset_dir, exist_ok=True)
            manager = AssetManager(asset_dir)

            assert manager.voices["narrator"]["text"] == "test narrator voice"
            assert manager.voices["male_pool"][0]["text"] == "test male voice"
            assert manager.voices["female_pool"][0]["text"] == "test female voice"
            assert manager.target_sr == 44100

    def test_works_without_config_file(self):
        """AssetManager should work with defaults when config file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AssetManager(tmpdir)
            # Should have default voices
            assert "narrator" in manager.voices
            assert len(manager.voices["male_pool"]) > 0


# ---------------------------------------------------------------------------
# Phase 1 Chapter-Skip Logic
# ---------------------------------------------------------------------------

class TestPhase1ChapterSkip:
    def test_continues_on_chapter_failure(self):
        """phase_1_generate_scripts should skip failed chapters, not abort."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only), skipping")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input directory with two chapter files
            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir, exist_ok=True)
            with open(os.path.join(input_dir, "chapter_01.txt"), 'w') as f:
                f.write("好的内容。正常文本。")
            with open(os.path.join(input_dir, "chapter_02.txt"), 'w') as f:
                f.write("更多好的内容。正常文本。")

            producer = CineCastProducer(config={
                "assets_dir": os.path.join(tmpdir, "assets"),
                "output_dir": os.path.join(tmpdir, "output"),
                "model_path": "dummy",
                "ambient_theme": "default",
                "target_duration_min": 30,
                "min_tail_min": 10,
                "use_local_llm": False,
            })

            # Monkey-patch check_ollama_alive to return True
            producer.check_ollama_alive = lambda: True
            # Monkey-patch _eject_ollama_memory to be a no-op
            producer._eject_ollama_memory = lambda: None

            # Create a director that fails on first chapter, succeeds on second
            call_count = [0]

            def mock_parse_and_micro_chunk(self_dir, content, chapter_prefix="chunk"):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("Simulated LLM failure")
                return [
                    {"chunk_id": f"{chapter_prefix}_00001",
                     "type": "narration", "speaker": "narrator",
                     "gender": "male", "content": "成功内容。", "pause_ms": 600}
                ]

            # Patch LLMScriptDirector
            original_parse = LLMScriptDirector.parse_and_micro_chunk
            LLMScriptDirector.parse_and_micro_chunk = mock_parse_and_micro_chunk
            try:
                result = producer.phase_1_generate_scripts(input_dir)
                # Should return True (continued processing) rather than False
                assert result is True
                # Second chapter should have been written
                script_files = os.listdir(producer.script_dir)
                assert len(script_files) >= 1
            finally:
                LLMScriptDirector.parse_and_micro_chunk = original_parse


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
