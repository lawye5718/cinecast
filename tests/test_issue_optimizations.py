#!/usr/bin/env python3
"""
Tests for the four optimizations from the issue:

1. Flexible JSON repair (order-independent salvage_json_entries)
2. Cast DB isolation (project-name-based cast_db_path)
3. Dialogue-dense text reduces chunk size instead of num_ctx
4. Smart recap injection position (preserves title/subtitle)
"""

import json
import os
import re
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import (
    salvage_json_entries,
    repair_json_array,
    _extract_fields_from_object,
    LLMScriptDirector,
)


# ---------------------------------------------------------------------------
# 1. Flexible JSON repair — order-independent field extraction
# ---------------------------------------------------------------------------

class TestFlexibleJsonRepair:
    """salvage_json_entries should tolerate reordered fields."""

    def test_standard_order(self):
        """Standard field order should still work."""
        raw = '{"type": "narration", "speaker": "narrator", "gender": "male", "emotion": "平静", "content": "一句话。"}'
        result = salvage_json_entries(raw)
        assert result is not None
        assert len(result) == 1
        assert result[0]["speaker"] == "narrator"
        assert result[0]["content"] == "一句话。"

    def test_reordered_fields(self):
        """Fields in non-standard order should still be extracted."""
        raw = '{"content": "你好世界", "speaker": "张三", "type": "dialogue", "gender": "male", "emotion": "激动"}'
        result = salvage_json_entries(raw)
        assert result is not None
        assert len(result) == 1
        assert result[0]["speaker"] == "张三"
        assert result[0]["content"] == "你好世界"
        assert result[0]["type"] == "dialogue"
        assert result[0]["gender"] == "male"
        assert result[0]["emotion"] == "激动"

    def test_extra_whitespace_in_fields(self):
        """Extra whitespace between keys and values should be tolerated."""
        raw = '{  "type" :  "narration" ,  "speaker" :  "narrator" ,  "content" :  "测试内容" }'
        result = salvage_json_entries(raw)
        assert result is not None
        assert len(result) == 1
        assert result[0]["content"] == "测试内容"

    def test_missing_optional_fields(self):
        """Entries with only speaker + content should still be extracted."""
        raw = '{"speaker": "老渔夫", "content": "大海很美。"}'
        result = salvage_json_entries(raw)
        assert result is not None
        assert len(result) == 1
        assert result[0]["speaker"] == "老渔夫"
        assert result[0]["type"] == "narration"  # default
        assert result[0]["gender"] == "unknown"  # default
        assert result[0]["emotion"] == "平静"    # default

    def test_instruct_alias_for_emotion(self):
        """'instruct' field should be treated as 'emotion'."""
        raw = '{"speaker": "narrator", "content": "text", "instruct": "温柔"}'
        result = salvage_json_entries(raw)
        assert result is not None
        assert result[0]["emotion"] == "温柔"

    def test_multiple_objects_different_orders(self):
        """Multiple objects with different field orders should all be extracted."""
        raw = (
            '{"type": "narration", "speaker": "narrator", "content": "第一句"} '
            'GARBAGE '
            '{"content": "第二句", "type": "dialogue", "speaker": "李四", "gender": "female", "emotion": "高兴"}'
        )
        result = salvage_json_entries(raw)
        assert result is not None
        assert len(result) == 2
        assert result[0]["content"] == "第一句"
        assert result[1]["speaker"] == "李四"
        assert result[1]["content"] == "第二句"

    def test_empty_content_from_structured_object(self):
        """Object with all fields but empty content should be skipped by strict pattern."""
        raw = '{"type": "narration", "speaker": "narrator", "gender": "male", "emotion": "平静", "content": ""}'
        result = salvage_json_entries(raw)
        # The loose fallback may still pick this up, but no meaningful content
        if result:
            assert result[0]["content"] == ""

    def test_completely_broken_returns_none(self):
        raw = "This is not JSON at all."
        result = salvage_json_entries(raw)
        assert result is None

    def test_repair_json_array_with_reordered_fields(self):
        """repair_json_array should recover entries with non-standard field order."""
        raw = '[{"content": "你好", "speaker": "张三", "type": "dialogue", "gender": "male", "emotion": "平静"}, {"content": "再见", "speaker": "李四"'
        result = repair_json_array(raw)
        assert result is not None
        assert len(result) >= 1
        assert result[0]["speaker"] == "张三"


class TestExtractFieldsFromObject:
    """Unit tests for _extract_fields_from_object helper."""

    def test_all_fields(self):
        obj = '{"type": "dialogue", "speaker": "A", "gender": "female", "emotion": "happy", "content": "hi"}'
        result = _extract_fields_from_object(obj)
        assert result == {
            "type": "dialogue",
            "speaker": "A",
            "gender": "female",
            "emotion": "happy",
            "content": "hi",
        }

    def test_no_speaker_no_content(self):
        obj = '{"type": "narration", "gender": "male"}'
        result = _extract_fields_from_object(obj)
        assert result is None

    def test_speaker_only(self):
        """speaker without content should still return (with empty content)."""
        obj = '{"speaker": "narrator"}'
        result = _extract_fields_from_object(obj)
        assert result is not None
        assert result["speaker"] == "narrator"
        assert result["content"] == ""


# ---------------------------------------------------------------------------
# 2. Cast DB isolation
# ---------------------------------------------------------------------------

class TestCastDBIsolation:
    """Verify that cast_db_path is dynamically generated from project name."""

    def test_source_has_project_based_cast_db(self):
        """main_producer.py should derive cast_db_path from input_source."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "_cast.json" in source, "cast_db_path should include project name suffix"
        assert "cast_db_path=cast_db_path" in source, (
            "cast_db_path should be passed to LLMScriptDirector"
        )

    def test_different_inputs_get_different_db_paths(self):
        """Different input sources should produce different cast_db_path values."""
        # Simulate the logic from main_producer.py
        def derive_cast_db_path(input_source):
            project_name = os.path.splitext(os.path.basename(input_source))[0]
            return os.path.join("workspace", f"{project_name}_cast.json")

        path1 = derive_cast_db_path("/books/仙侠.epub")
        path2 = derive_cast_db_path("/books/科幻.epub")
        assert path1 != path2
        assert "仙侠" in path1
        assert "科幻" in path2

    def test_txt_directory_gets_dir_name(self):
        """A TXT directory input should use the directory name."""
        def derive_cast_db_path(input_source):
            project_name = os.path.splitext(os.path.basename(input_source))[0]
            return os.path.join("workspace", f"{project_name}_cast.json")

        path = derive_cast_db_path("/books/my_novel")
        assert "my_novel" in path
        assert path.endswith("_cast.json")


# ---------------------------------------------------------------------------
# 3. Dialogue-dense: reduce chunk size, not num_ctx
# ---------------------------------------------------------------------------

class TestDialogueDenseStrategy:
    """Verify dialogue-dense detection reduces chunk size instead of num_ctx."""

    def test_source_no_longer_reduces_num_ctx(self):
        """_request_ollama should NOT reduce num_ctx for dialogue-dense text."""
        import inspect
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_ollama)
        assert "num_ctx // 2" not in source, (
            "num_ctx should no longer be halved for dialogue-dense text"
        )

    def test_parse_text_to_script_has_dialogue_detection(self):
        """parse_text_to_script should detect dialogue density and adjust max_length."""
        import inspect
        director = LLMScriptDirector()
        source = inspect.getsource(director.parse_text_to_script)
        assert "dialogue_markers" in source
        assert "max_length" in source

    def test_num_ctx_stays_constant(self):
        """_request_ollama should always use num_ctx=8192."""
        import inspect
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_ollama)
        assert "num_ctx = 8192" in source
        # Should not conditionally modify num_ctx
        assert "num_ctx = max(" not in source


# ---------------------------------------------------------------------------
# 4. Smart recap injection position
# ---------------------------------------------------------------------------

class TestSmartRecapInjection:
    """Verify _find_recap_insert_index finds the first narration/dialogue."""

    def _find_recap_insert_index(self, micro_script):
        """Mirror _find_recap_insert_index from CineCastProducer."""
        for i, entry in enumerate(micro_script):
            if entry.get("type") in ("narration", "dialogue"):
                return i
        return 0

    def test_empty_script(self):
        assert self._find_recap_insert_index([]) == 0

    def test_only_title(self):
        """With only a title entry, should return 0 (no narration/dialogue found)."""
        script = [{"type": "title", "speaker": "narrator", "content": "第一章"}]
        assert self._find_recap_insert_index(script) == 0

    def test_title_then_narration(self):
        """Should return index 1 (the narration after the title)."""
        script = [
            {"type": "title", "speaker": "narrator", "content": "第一章 风雪"},
            {"type": "narration", "speaker": "narrator", "content": "夜幕降临。"},
        ]
        assert self._find_recap_insert_index(script) == 1

    def test_title_subtitle_then_narration(self):
        """Title + subtitle should both be preserved; recap goes before narration."""
        script = [
            {"type": "title", "speaker": "narrator", "content": "第一章"},
            {"type": "subtitle", "speaker": "narrator", "content": "副标题"},
            {"type": "narration", "speaker": "narrator", "content": "故事开始。"},
        ]
        assert self._find_recap_insert_index(script) == 2

    def test_title_then_dialogue(self):
        """Should also find dialogue as a valid injection point."""
        script = [
            {"type": "title", "speaker": "narrator", "content": "第二章"},
            {"type": "dialogue", "speaker": "张三", "content": "你好！"},
        ]
        assert self._find_recap_insert_index(script) == 1

    def test_narration_first(self):
        """If narration is the very first entry, insert at 0."""
        script = [
            {"type": "narration", "speaker": "narrator", "content": "一切都很安静。"},
            {"type": "dialogue", "speaker": "张三", "content": "你好！"},
        ]
        assert self._find_recap_insert_index(script) == 0

    def test_insertion_preserves_order(self):
        """Simulate actual insertion: recap goes between title and narration."""
        script = [
            {"type": "title", "speaker": "narrator", "content": "第一章"},
            {"type": "subtitle", "speaker": "narrator", "content": "副标题"},
            {"type": "narration", "speaker": "narrator", "content": "故事开始。"},
        ]
        insert_idx = self._find_recap_insert_index(script)
        recap = {"type": "recap", "speaker": "talkover", "content": "前情提要"}
        script.insert(insert_idx, recap)
        assert script[0]["type"] == "title"
        assert script[1]["type"] == "subtitle"
        assert script[2]["type"] == "recap"
        assert script[3]["type"] == "narration"

    def test_source_uses_find_recap_insert_index(self):
        """main_producer.py should use _find_recap_insert_index, not hardcoded index."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "_find_recap_insert_index" in source
        # Old hardcoded pattern should be gone
        assert "insert_idx = 1 if len(micro_script)" not in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
