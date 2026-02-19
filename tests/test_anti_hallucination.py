#!/usr/bin/env python3
"""
Tests for anti-hallucination hardening in llm_director.py.

Covers:
- EMOTION_SET constraint for Qwen3-TTS emotion keywords
- Simplified data-converter system prompt
- Few-shot example anchoring
- Forceful anti-hallucination user prompt
- Model parameter adjustments (temperature, top_p, num_predict)
- ASCII quote preprocessing to avoid JSON structure conflicts
"""

import inspect
import json
import os
import sys
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import LLMScriptDirector


# ---------------------------------------------------------------------------
# EMOTION_SET Constraint
# ---------------------------------------------------------------------------

class TestEmotionSetConstraint:
    """Verify EMOTION_SET constrains emotion keywords to Qwen3-TTS supported set."""

    def test_emotion_set_defined_in_source(self):
        """_request_ollama should define EMOTION_SET with allowed emotion keywords."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert "EMOTION_SET" in source

    def test_emotion_set_contains_all_required_emotions(self):
        """EMOTION_SET should contain all 9 Qwen3-TTS core emotions."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        required_emotions = ["平静", "愤怒", "悲伤", "喜悦", "恐惧", "惊讶", "沧桑", "柔和", "激动"]
        for emotion in required_emotions:
            assert emotion in source, f"Missing emotion: {emotion}"

    def test_emotion_constraint_instruction_in_prompt(self):
        """The system prompt should instruct emotion selection from EMOTION_SET only."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert "仅限从以下词汇中选择" in source


# ---------------------------------------------------------------------------
# Simplified Data-Converter System Prompt
# ---------------------------------------------------------------------------

class TestSimplifiedSystemPrompt:
    """Verify the system prompt uses simplified data-converter role."""

    def test_data_converter_role(self):
        """System prompt should define the model as a data format converter."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert "数据格式转换工具" in source

    def test_no_director_terminology(self):
        """System prompt should not use complex director terminology."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert "有声书导演" not in source

    def test_anti_merge_instruction(self):
        """System prompt should forbid merging multiple sentences into one content."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert "严禁将多句话合并" in source

    def test_flat_array_enforcement(self):
        """System prompt should enforce flat JSON array output."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert "平铺的 JSON 数组" in source
        assert "严禁输出字典" in source


# ---------------------------------------------------------------------------
# Few-Shot Example Anchoring
# ---------------------------------------------------------------------------

class TestFewShotAnchoring:
    """Verify few-shot example is injected for sentence-by-sentence decomposition."""

    def test_one_shot_example_defined(self):
        """_request_ollama should define a one_shot_example."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert "one_shot_example" in source

    def test_one_shot_example_injected_in_messages(self):
        """The one_shot_example should be injected into the system message."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert "示例参考" in source

    def test_few_shot_shows_sentence_decomposition(self):
        """The few-shot example should demonstrate splitting into multiple objects."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        # Should show dialogue and narration being split
        assert '"type": "dialogue"' in source
        assert '"type": "narration"' in source


# ---------------------------------------------------------------------------
# Model Parameter Adjustments
# ---------------------------------------------------------------------------

class TestModelParameterAdjustments:
    """Verify model parameters are adjusted for anti-hallucination."""

    def test_temperature_lowered(self):
        """Temperature should be set to 0.1 for reduced randomness."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert '"temperature": 0.1' in source

    def test_num_predict_set(self):
        """num_predict should be set to 2048 for sufficient output length."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert '"num_predict": 2048' in source

    def test_parameters_in_mock_payload(self):
        """Verify the actual payload sent to Ollama contains correct parameters."""
        director = LLMScriptDirector()
        fake_resp = mock.MagicMock()
        fake_resp.status_code = 200
        fake_resp.raise_for_status = mock.MagicMock()
        fake_resp.json.return_value = {
            "message": {"content": json.dumps([
                {"type": "narration", "speaker": "narrator", "gender": "male",
                 "emotion": "平静", "content": "测试。"}
            ], ensure_ascii=False)}
        }

        captured_payloads = []

        def capture_post(url, json=None, timeout=None):
            captured_payloads.append(json)
            return fake_resp

        with mock.patch("modules.llm_director.requests.post", side_effect=capture_post):
            director._request_ollama("测试文本。")

        assert len(captured_payloads) == 1
        options = captured_payloads[0]["options"]
        assert options["temperature"] == 0.1
        assert options["top_p"] == 0.1
        assert options["num_predict"] == 2048
        assert options["num_ctx"] == 8192


# ---------------------------------------------------------------------------
# ASCII Quote Preprocessing
# ---------------------------------------------------------------------------

class TestQuotePreprocessing:
    """Verify ASCII double quotes are replaced with Chinese quotes."""

    def test_quote_preprocessing_in_source(self):
        """_request_ollama should contain quote preprocessing logic."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert "双引号" in source or "\\u201c" in source

    def test_quotes_replaced_in_payload(self):
        """ASCII double quotes in input text should be replaced in the payload."""
        director = LLMScriptDirector()
        fake_resp = mock.MagicMock()
        fake_resp.status_code = 200
        fake_resp.raise_for_status = mock.MagicMock()
        fake_resp.json.return_value = {
            "message": {"content": json.dumps([
                {"type": "narration", "speaker": "narrator", "gender": "male",
                 "emotion": "平静", "content": "测试。"}
            ], ensure_ascii=False)}
        }

        captured_payloads = []

        def capture_post(url, json=None, timeout=None):
            captured_payloads.append(json)
            return fake_resp

        input_text = '"你好，"他说。'

        with mock.patch("modules.llm_director.requests.post", side_effect=capture_post):
            director._request_ollama(input_text)

        assert len(captured_payloads) == 1
        user_content = captured_payloads[0]["messages"][1]["content"]
        # The processed text portion should not contain ASCII double quotes
        # (they are replaced with Chinese quotes to avoid JSON conflicts)
        assert '\u201c' in user_content or '\u201d' in user_content


# ---------------------------------------------------------------------------
# Forceful User Prompt
# ---------------------------------------------------------------------------

class TestForcefulUserPrompt:
    """Verify the user prompt uses forceful anti-hallucination language."""

    def test_user_prompt_contains_warning(self):
        """User prompt should contain explicit warning against deletion."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert "禁止删减" in source

    def test_user_prompt_contains_repeater_role(self):
        """User prompt should assign 'repeater converter' role."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert "复读机转换器" in source

    def test_user_prompt_contains_count_hint(self):
        """User prompt should contain line count hint."""
        source = inspect.getsource(LLMScriptDirector._request_ollama)
        assert "多少句话" in source or "多少个元素" in source
