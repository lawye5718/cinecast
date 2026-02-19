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
        """The _request_ollama method should reference archetype prompt injection."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_ollama)
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
    """Verify emotion field uses emotion+acoustic dual descriptions."""

    def test_emotion_dual_description_instruction(self):
        """system_prompt should instruct emotion+acoustic dual description."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_ollama)
        assert "情感与音色双重标签" in source or "音色描述" in source

    def test_emotion_examples_present(self):
        """Example emotion descriptions should be in the prompt."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_ollama)
        assert "High-pitched" in source or "Pitch" in source


# ---------------------------------------------------------------------------
# Punctuation & Interjection Preservation
# ---------------------------------------------------------------------------

class TestPunctuationPreservation:
    """Verify instructions to preserve interjections and punctuation."""

    def test_interjection_instruction_in_prompt(self):
        """system_prompt should instruct preservation of interjections."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_ollama)
        assert "语气词" in source
        assert "嗯" in source or "啊" in source

    def test_punctuation_preservation_instruction(self):
        """system_prompt should mention long pause punctuation like '...'."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_ollama)
        assert "TTS Prosody" in source or "标点" in source


# ---------------------------------------------------------------------------
# JSON Anti-Truncation Output Order
# ---------------------------------------------------------------------------

class TestAntiTruncationInstruction:
    """Verify JSON output ordering instruction to reduce truncation errors."""

    def test_output_order_instruction_in_prompt(self):
        """system_prompt should instruct character list before JSON output."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_ollama)
        assert "Anti-Truncation" in source or "输出顺序纪律" in source


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
        # Mock _request_ollama to return script with a speaker
        script_chunk = [
            {"type": "dialogue", "speaker": "老渔夫", "gender": "male",
             "emotion": "Sad, low volume, shaky voice", "content": "你相信命运吗？"},
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "老渔夫说道。"},
        ]
        director._request_ollama = lambda text_chunk, context=None: script_chunk
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

        director._request_ollama = mock_request
        director._chunk_text_for_llm = lambda text, max_length=800: ["chunk1", "chunk2"]
        director.parse_text_to_script("test text")
        # First emotion should be preserved
        assert director._local_session_cast["英雄"] == "Angry, high pitch"

    def test_voice_lock_in_source(self):
        """The _request_ollama method should contain Voice Lock injection logic."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_ollama)
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

    def test_dialogue_heavy_text_reduces_num_ctx(self):
        """Dialogue-heavy text >500 chars with many quotes should reduce num_ctx."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_ollama)
        assert "num_ctx" in source
        # Verify the overflow protection logic exists
        assert "对话密集型" in source or "dialogue_markers" in source

    def test_num_ctx_used_in_payload(self):
        """The payload should use the dynamic num_ctx variable."""
        director = LLMScriptDirector()
        source = inspect.getsource(director._request_ollama)
        # Verify dynamic num_ctx is used (not hardcoded 8192)
        assert '"num_ctx": num_ctx' in source

    def test_overflow_protection_with_mock(self):
        """Mock an Ollama call with dialogue-heavy text to verify overflow protection."""
        director = LLMScriptDirector()
        # Create text > 500 chars with many quote marks
        dialogue_text = '"你好吗？"她说。"我很好。"他回答。' * 50  # lots of dialogue markers, > 500 chars

        fake_resp = mock.MagicMock()
        fake_resp.status_code = 200
        fake_resp.raise_for_status = mock.MagicMock()
        fake_resp.json.return_value = {
            "message": {"content": json.dumps([
                {"type": "narration", "speaker": "narrator", "gender": "male",
                 "emotion": "平静", "content": "测试内容。"}
            ], ensure_ascii=False)}
        }

        captured_payloads = []
        original_post = mock.MagicMock(return_value=fake_resp)

        def capture_post(url, json=None, timeout=None):
            captured_payloads.append(json)
            return fake_resp

        with mock.patch("modules.llm_director.requests.post", side_effect=capture_post):
            director._request_ollama(dialogue_text)

        assert len(captured_payloads) == 1
        payload_num_ctx = captured_payloads[0]["options"]["num_ctx"]
        # With heavy dialogue, should be reduced from 8192
        assert payload_num_ctx < 8192
