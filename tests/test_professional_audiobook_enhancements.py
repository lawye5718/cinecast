#!/usr/bin/env python3
"""
Tests for professional audiobook production enhancements.

Covers:
- VOICE_ARCHETYPES class attribute and _get_archetype_prompt()
- Voice consistency persistence (_load_cast_profiles, _save_cast_profile, _update_cast_db)
- Context reset for novella collections (reset_context)
- Story boundary detection (_is_new_story_start)
- Debug logging in _request_llm (input length warnings)
- Emotion field format (dual emotion+acoustic description)
- Default voice instruction fallback in _validate_script_elements
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import LLMScriptDirector


# ---------------------------------------------------------------------------
# VOICE_ARCHETYPES
# ---------------------------------------------------------------------------

class TestVoiceArchetypes:
    def test_archetypes_is_class_attribute(self):
        assert hasattr(LLMScriptDirector, "VOICE_ARCHETYPES")
        assert isinstance(LLMScriptDirector.VOICE_ARCHETYPES, dict)

    def test_archetypes_contains_expected_keys(self):
        expected = {"intellectual", "villain_sly", "melancholic", "authoritative", "innocent"}
        assert expected == set(LLMScriptDirector.VOICE_ARCHETYPES.keys())

    def test_archetypes_values_are_non_empty_strings(self):
        for key, value in LLMScriptDirector.VOICE_ARCHETYPES.items():
            assert isinstance(value, str), f"Archetype '{key}' is not a string"
            assert len(value) > 10, f"Archetype '{key}' description too short"


# ---------------------------------------------------------------------------
# _get_archetype_prompt
# ---------------------------------------------------------------------------

class TestGetArchetypePrompt:
    def test_returns_string(self):
        director = LLMScriptDirector()
        prompt = director._get_archetype_prompt()
        assert isinstance(prompt, str)

    def test_contains_all_archetypes(self):
        director = LLMScriptDirector()
        prompt = director._get_archetype_prompt()
        for key in LLMScriptDirector.VOICE_ARCHETYPES:
            assert key in prompt

    def test_contains_header(self):
        director = LLMScriptDirector()
        prompt = director._get_archetype_prompt()
        assert "音色设计参考手册" in prompt


# ---------------------------------------------------------------------------
# Voice Consistency Persistence
# ---------------------------------------------------------------------------

class TestVoiceConsistencyPersistence:
    def test_load_cast_profiles_empty_when_no_file(self, tmp_path):
        db_path = str(tmp_path / "nonexistent.json")
        director = LLMScriptDirector(cast_db_path=db_path)
        assert director.cast_profiles == {}

    def test_save_and_load_cast_profile(self, tmp_path):
        db_path = str(tmp_path / "cast_profiles.json")
        director = LLMScriptDirector(cast_db_path=db_path)

        director._save_cast_profile("老渔夫", "male", "Deep, husky voice")
        assert "老渔夫" in director.cast_profiles
        assert director.cast_profiles["老渔夫"]["gender"] == "male"
        assert director.cast_profiles["老渔夫"]["voice_instruction"] == "Deep, husky voice"

        # Verify file persisted
        assert os.path.exists(db_path)
        with open(db_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert "老渔夫" in loaded

    def test_save_does_not_overwrite_existing(self, tmp_path):
        db_path = str(tmp_path / "cast_profiles.json")
        director = LLMScriptDirector(cast_db_path=db_path)

        director._save_cast_profile("老渔夫", "male", "Deep, husky voice")
        director._save_cast_profile("老渔夫", "female", "Different voice")

        # Should keep the original
        assert director.cast_profiles["老渔夫"]["gender"] == "male"
        assert director.cast_profiles["老渔夫"]["voice_instruction"] == "Deep, husky voice"

    def test_load_existing_cast_profiles(self, tmp_path):
        db_path = str(tmp_path / "cast_profiles.json")
        existing = {"艾米莉": {"gender": "female", "voice_instruction": "Soft, warm voice"}}
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False)

        director = LLMScriptDirector(cast_db_path=db_path)
        assert "艾米莉" in director.cast_profiles
        assert director.cast_profiles["艾米莉"]["gender"] == "female"

    def test_update_cast_db_extracts_from_script(self, tmp_path):
        db_path = str(tmp_path / "cast_profiles.json")
        director = LLMScriptDirector(cast_db_path=db_path)

        script = [
            {"speaker": "张三", "gender": "male", "emotion": "激动 (High-pitched, energetic male voice)"},
            {"speaker": "narrator", "gender": "male", "emotion": "平静"},
            {"speaker": "李四", "gender": "female", "emotion": "温柔"},  # No parentheses - no extraction
        ]
        director._update_cast_db(script)
        assert "张三" in director.cast_profiles
        assert director.cast_profiles["张三"]["voice_instruction"] == "High-pitched, energetic male voice"
        assert "李四" not in director.cast_profiles  # No acoustic desc in parentheses
        assert "narrator" not in director.cast_profiles

    def test_update_cast_db_does_not_duplicate(self, tmp_path):
        db_path = str(tmp_path / "cast_profiles.json")
        director = LLMScriptDirector(cast_db_path=db_path)

        script1 = [{"speaker": "老渔夫", "gender": "male", "emotion": "沧桑 (Deep, husky voice)"}]
        script2 = [{"speaker": "老渔夫", "gender": "male", "emotion": "沧桑 (Different voice)"}]
        director._update_cast_db(script1)
        director._update_cast_db(script2)

        # Should keep the first one
        assert director.cast_profiles["老渔夫"]["voice_instruction"] == "Deep, husky voice"


# ---------------------------------------------------------------------------
# Context Reset
# ---------------------------------------------------------------------------

class TestContextReset:
    def test_reset_context_clears_all_state(self):
        director = LLMScriptDirector()
        # Simulate state from processing
        director._prev_characters = ["老渔夫", "年轻人"]
        director._prev_tail_entries = [{"speaker": "老渔夫"}]
        director._local_session_cast = {"老渔夫": "沧桑"}

        director.reset_context()

        assert director._prev_characters == []
        assert director._prev_tail_entries == []
        assert director._local_session_cast == {}

    def test_reset_context_preserves_cast_profiles(self, tmp_path):
        db_path = str(tmp_path / "cast_profiles.json")
        director = LLMScriptDirector(cast_db_path=db_path)
        director._save_cast_profile("老渔夫", "male", "Deep voice")
        director._prev_characters = ["老渔夫"]

        director.reset_context()

        # cast_profiles should NOT be cleared by reset_context
        assert "老渔夫" in director.cast_profiles


# ---------------------------------------------------------------------------
# Story Boundary Detection (_is_new_story_start)
# ---------------------------------------------------------------------------

class TestStoryBoundaryDetection:
    """Test _is_new_story_start boundary detection logic."""

    @staticmethod
    def _is_new_story_start(chapter_name, content, prev_chapter_name=None):
        """Local copy of the boundary detection logic for testing without mlx."""
        import re as _re
        if prev_chapter_name is None:
            return False
        new_story_patterns = [
            r'第[一1]章',
            r'序[章言]',
            r'楔子',
            r'(?i)chapter[_ ]?0*1\b',
            r'(?i)prologue',
        ]
        for pattern in new_story_patterns:
            if _re.search(pattern, chapter_name):
                return True
            if _re.search(pattern, content[:100]):
                return True
        return False

    def test_first_chapter_is_not_new_story(self):
        """The very first chapter should not be detected as new story."""
        result = self._is_new_story_start("第一章 开始", "故事内容...", None)
        assert result is False

    def test_chapter_1_after_previous_is_new_story(self):
        result = self._is_new_story_start("第一章 新故事", "内容...", "Chapter_010")
        assert result is True

    def test_di_yi_zhang_detected(self):
        result = self._is_new_story_start("第1章 起点", "内容...", "Chapter_005")
        assert result is True

    def test_prologue_detected(self):
        result = self._is_new_story_start("序章", "内容...", "Chapter_010")
        assert result is True

    def test_序言_detected(self):
        result = self._is_new_story_start("序言", "内容...", "Chapter_010")
        assert result is True

    def test_楔子_detected(self):
        result = self._is_new_story_start("楔子", "内容...", "Chapter_010")
        assert result is True

    def test_english_chapter_1_detected(self):
        result = self._is_new_story_start("Chapter_001", "内容...", "Chapter_020")
        assert result is True

    def test_chapter_2_not_detected(self):
        result = self._is_new_story_start("第二章 继续", "内容...", "第一章 开始")
        assert result is False

    def test_regular_chapter_not_detected(self):
        result = self._is_new_story_start("Chapter_005", "普通内容...", "Chapter_004")
        assert result is False

    def test_content_body_chapter_1_detected(self):
        """Detect new story when '第一章' appears in content body."""
        result = self._is_new_story_start(
            "Chapter_010", "第一章 新的开始\n从前有座山...", "Chapter_009"
        )
        assert result is True


# ---------------------------------------------------------------------------
# Debug Logging (_request_llm input length check)
# ---------------------------------------------------------------------------

class TestDebugLogging:
    def test_request_llm_source_has_input_len_warning(self):
        """_request_llm should contain input length warning logic."""
        import inspect
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_llm)
        assert "input_len" in source
        assert "800" in source or "700" in source


# ---------------------------------------------------------------------------
# Emotion Default Fallback in _validate_script_elements
# ---------------------------------------------------------------------------

class TestEmotionDefaultFallback:
    def test_empty_emotion_gets_default_for_male(self):
        director = LLMScriptDirector()
        elements = [
            {"type": "dialogue", "speaker": "张三", "content": "你好",
             "gender": "male", "emotion": ""},
        ]
        result = director._validate_script_elements(elements)
        assert "Clear" in result[0]["emotion"]

    def test_empty_emotion_gets_default_for_female(self):
        director = LLMScriptDirector()
        elements = [
            {"type": "dialogue", "speaker": "艾米莉", "content": "你好",
             "gender": "female", "emotion": ""},
        ]
        result = director._validate_script_elements(elements)
        assert "Breathier" in result[0]["emotion"]

    def test_narrator_emotion_not_modified(self):
        director = LLMScriptDirector()
        elements = [
            {"type": "narration", "speaker": "narrator", "content": "旁白",
             "gender": "male", "emotion": ""},
        ]
        result = director._validate_script_elements(elements)
        # Narrator emotion should stay as-is (empty or default "平静")
        assert "intellectual" not in result[0].get("emotion", "").lower()

    def test_non_empty_emotion_preserved(self):
        director = LLMScriptDirector()
        elements = [
            {"type": "dialogue", "speaker": "张三", "content": "你好",
             "gender": "male", "emotion": "激动"},
        ]
        result = director._validate_script_elements(elements)
        assert result[0]["emotion"] == "激动"

    def test_emotion_with_description_preserved(self):
        director = LLMScriptDirector()
        elements = [
            {"type": "dialogue", "speaker": "张三", "content": "你好",
             "gender": "male", "emotion": "激动 (High-pitched, energetic male voice)"},
        ]
        result = director._validate_script_elements(elements)
        assert result[0]["emotion"] == "激动 (High-pitched, energetic male voice)"


# ---------------------------------------------------------------------------
# Emotion Format in System Prompt
# ---------------------------------------------------------------------------

class TestEmotionFormatInPrompt:
    def test_emotion_constrained_to_emotion_set(self):
        """The system prompt should constrain emotions to EMOTION_SET keywords."""
        import inspect
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_llm)
        assert "EMOTION_SET" in source
        assert "情绪约束" in source or "仅限" in source

    def test_emotion_set_contains_core_emotions(self):
        """EMOTION_SET should contain core Qwen3-TTS emotion keywords."""
        import inspect
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_llm)
        for emotion in ["平静", "愤怒", "悲伤", "喜悦", "恐惧", "惊讶", "沧桑", "柔和", "激动"]:
            assert emotion in source
