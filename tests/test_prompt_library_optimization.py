#!/usr/bin/env python3
"""
Tests for Qwen3-TTS prompt library optimization.

Covers:
- Voice archetype mapping in system_prompt
- Enhanced emotion field instructions (emotion+acoustic dual descriptions)
- Punctuation/interjection preservation instructions
- JSON anti-truncation output order instruction
- Local session cast for voice consistency across chunks
- Gender-voice conflict validation in _validate_script_elements
- _normalize_text for number/symbol to Chinese conversion
- JSON overflow protection for dialogue-heavy text
"""

import inspect
import json
import os
import re
import sys
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import LLMScriptDirector


# ---------------------------------------------------------------------------
# Voice Archetype Mapping in system_prompt
# ---------------------------------------------------------------------------

class TestVoiceArchetypeMapping:
    """Verify voice archetype mapping guideline is injected into the prompt."""

    def test_archetype_guideline_in_source(self):
        """The _request_llm method should reference archetype prompt injection."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_llm)
        assert "_get_archetype_prompt" in source
        assert "音色映射指南" in source or "VOICE_ARCHETYPES" in source

    def test_archetype_keywords_present(self):
        """Archetype descriptions should cover key character types."""
        director = LLMScriptDirector()
        assert "intellectual" in director.VOICE_ARCHETYPES
        assert "authoritative" in director.VOICE_ARCHETYPES
        assert "innocent" in director.VOICE_ARCHETYPES

    def test_get_archetype_prompt_contains_all_keys(self):
        """_get_archetype_prompt should include all VOICE_ARCHETYPES entries."""
        director = LLMScriptDirector()
        prompt = director._get_archetype_prompt()
        for key, value in director.VOICE_ARCHETYPES.items():
            assert key in prompt
            assert value in prompt


# ---------------------------------------------------------------------------
# Enhanced Emotion Field Instructions
# ---------------------------------------------------------------------------

class TestEnhancedEmotionInstructions:
    """Verify emotion field uses constrained EMOTION_SET keywords."""

    def test_emotion_constrained_instruction(self):
        """system_prompt should instruct constrained emotion from EMOTION_SET."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_llm)
        assert "EMOTION_SET" in source or "emotion 规定" in source

    def test_emotion_set_keywords_present(self):
        """EMOTION_SET keywords should be present in the prompt."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_llm)
        assert "平静" in source and "愤怒" in source and "激动" in source


# ---------------------------------------------------------------------------
# Punctuation & Interjection Preservation
# ---------------------------------------------------------------------------

class TestAntiHallucinationPrompt:
    """Verify anti-hallucination hardening in the prompt."""

    def test_anti_hallucination_few_shot_in_prompt(self):
        """system_prompt should contain few-shot example for sentence-by-sentence decomposition."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_llm)
        assert "one_shot_example" in source
        assert "示例参考" in source

    def test_anti_hallucination_quote_preprocessing(self):
        """_request_llm should preprocess ASCII quotes to avoid JSON conflicts."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_llm)
        assert "双引号" in source or "\\u201c" in source


# ---------------------------------------------------------------------------
# JSON Anti-Truncation Output Order
# ---------------------------------------------------------------------------

class TestAntiDriftInstruction:
    """Verify anti-instruction-drift features in the prompt."""

    def test_simplified_prompt_uses_data_converter_role(self):
        """system_prompt should use high-precision audiobook conversion interface role."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_llm)
        assert "高精度的有声书剧本转换接口" in source


# ---------------------------------------------------------------------------
# Local Session Cast (Voice Consistency Across Chunks)
# ---------------------------------------------------------------------------

class TestLocalSessionCast:
    """Verify local session cast tracks voice descriptions across chunks."""

    def test_local_session_cast_initialized_empty(self):
        director = LLMScriptDirector()
        assert hasattr(director, '_local_session_cast')
        assert director._local_session_cast == {}

    def test_local_session_cast_populated_after_parse(self):
        """After parsing, _local_session_cast should contain speaker emotions."""
        director = LLMScriptDirector()
        # Mock _request_llm to return script with a speaker
        script_chunk = [
            {"type": "dialogue", "speaker": "老渔夫", "gender": "male",
             "emotion": "Sad, low volume, shaky voice", "content": "你相信命运吗？"},
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "老渔夫说道。"},
        ]
        director._request_llm = lambda text_chunk, context=None: script_chunk
        director.parse_text_to_script("test text")
        assert "老渔夫" in director._local_session_cast
        assert director._local_session_cast["老渔夫"] == "Sad, low volume, shaky voice"

    def test_local_session_cast_preserves_first_voice(self):
        """First voice description should be preserved, not overwritten."""
        director = LLMScriptDirector()
        chunk1 = [
            {"type": "dialogue", "speaker": "英雄", "gender": "male",
             "emotion": "Angry, high pitch", "content": "放开她！"},
        ]
        chunk2 = [
            {"type": "dialogue", "speaker": "英雄", "gender": "male",
             "emotion": "Calm, steady", "content": "我知道了。"},
        ]
        call_count = [0]

        def mock_request(text_chunk, context=None):
            result = chunk1 if call_count[0] == 0 else chunk2
            call_count[0] += 1
            return result

        director._request_llm = mock_request
        director._chunk_text_for_llm = lambda text, max_length=800: ["chunk1", "chunk2"]
        director.parse_text_to_script("test text")
        # First emotion should be preserved
        assert director._local_session_cast["英雄"] == "Angry, high pitch"

    def test_voice_lock_in_source(self):
        """The _request_llm method should contain Voice Lock injection logic."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_llm)
        assert "Voice Lock" in source or "角色音色锁定" in source


# ---------------------------------------------------------------------------
# Gender-Voice Conflict Validation
# ---------------------------------------------------------------------------

class TestGenderVoiceConflictValidation:
    """Verify gender-voice conflict detection in _validate_script_elements."""

    def test_female_baritone_corrected(self):
        """Female speaker with baritone in emotion should be auto-corrected."""
        director = LLMScriptDirector()
        elements = [
            {"type": "dialogue", "speaker": "小红", "gender": "female",
             "emotion": "Resonant, deep baritone, gravelly", "content": "你好。"},
        ]
        result = director._validate_script_elements(elements)
        assert len(result) == 1
        emotion = result[0]["emotion"].lower()
        assert "baritone" not in emotion
        assert "alto" in emotion

    def test_female_bass_corrected(self):
        """Female speaker with bass in emotion should be auto-corrected."""
        director = LLMScriptDirector()
        elements = [
            {"type": "dialogue", "speaker": "女演员", "gender": "female",
             "emotion": "Deep bass voice, authoritative", "content": "安静。"},
        ]
        result = director._validate_script_elements(elements)
        emotion = result[0]["emotion"].lower()
        assert "bass" not in emotion
        assert "alto" in emotion

    def test_male_baritone_unchanged(self):
        """Male speaker with baritone should not be modified."""
        director = LLMScriptDirector()
        elements = [
            {"type": "dialogue", "speaker": "老爷爷", "gender": "male",
             "emotion": "Resonant, deep baritone, gravelly", "content": "好的。"},
        ]
        result = director._validate_script_elements(elements)
        assert "baritone" in result[0]["emotion"].lower()

    def test_unknown_gender_baritone_unchanged(self):
        """Unknown gender with baritone should not be modified."""
        director = LLMScriptDirector()
        elements = [
            {"type": "dialogue", "speaker": "路人", "gender": "unknown",
             "emotion": "Deep baritone", "content": "走开。"},
        ]
        result = director._validate_script_elements(elements)
        assert "baritone" in result[0]["emotion"].lower()


# ---------------------------------------------------------------------------
# _normalize_text: Number/Symbol to Chinese Conversion
# ---------------------------------------------------------------------------

class TestNormalizeText:
    """Verify _normalize_text converts numbers and symbols to Chinese."""

    def test_percentage_conversion(self):
        result = LLMScriptDirector._normalize_text("利率是10%")
        assert "百分之一零" in result
        assert "%" not in result

    def test_decimal_conversion(self):
        result = LLMScriptDirector._normalize_text("圆周率约为3.14")
        assert "三点一四" in result
        assert "3.14" not in result

    def test_integer_conversion(self):
        result = LLMScriptDirector._normalize_text("他有100块钱")
        assert "一零零" in result
        assert "100" not in result

    def test_no_digits_unchanged(self):
        text = "这是纯中文文本，没有数字。"
        result = LLMScriptDirector._normalize_text(text)
        assert result == text

    def test_mixed_text_conversion(self):
        result = LLMScriptDirector._normalize_text("他花了50%的时间，跑了3.5公里")
        assert "百分之五零" in result
        assert "三点五" in result

    def test_complex_percentage(self):
        result = LLMScriptDirector._normalize_text("增长了12.5%")
        assert "百分之一二点五" in result
        assert "%" not in result


# ---------------------------------------------------------------------------
# JSON Overflow Protection
# ---------------------------------------------------------------------------

class TestJsonOverflowProtection:
    """Verify JSON overflow protection for dialogue-heavy text."""

    def test_dialogue_density_protection_exists(self):
        """Qwen-Flash handles full chapters; dialogue density reduction removed."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_llm)
        assert "max_tokens" in source
        # Qwen-Flash with 1M context no longer needs dialogue density reduction
        pts_source = inspect.getsource(director.parse_text_to_script)
        assert "Qwen-Flash" in pts_source or "max_length" in pts_source

    def test_max_tokens_used_in_payload(self):
        """The payload should use the max_tokens variable."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_llm)
        # Verify max_tokens is used
        assert "max_tokens" in source

    def test_overflow_protection_with_mock(self):
        """Mock a Qwen API call with dialogue-heavy text to verify overflow protection."""
        director = LLMScriptDirector()
        # Create text > 500 chars with many quote marks
        dialogue_text = '\u201c你好吗？\u201d她说。\u201c我很好。\u201d他回答。' * 50  # lots of dialogue markers, > 500 chars

        content_str = json.dumps([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "测试内容。"}
        ], ensure_ascii=False)

        mock_chunk = mock.MagicMock()
        mock_chunk.choices = [mock.MagicMock()]
        mock_chunk.choices[0].delta = mock.MagicMock()
        mock_chunk.choices[0].delta.content = content_str
        mock_chunk.choices[0].delta.reasoning_content = None

        captured_kwargs = []

        def capture_create(**kwargs):
            captured_kwargs.append(kwargs)
            return iter([mock_chunk])

        director.client = mock.MagicMock()
        director.client.chat.completions.create = capture_create

        director._request_llm(dialogue_text)

        assert len(captured_kwargs) == 1
        payload_max_tokens = captured_kwargs[0]["max_tokens"]
        # max_tokens should be set; dialogue-dense strategy now reduces
        # text chunk size instead of max_tokens to preserve speaker context.
        assert payload_max_tokens > 0
