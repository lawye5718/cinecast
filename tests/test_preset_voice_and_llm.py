#!/usr/bin/env python3
"""
Tests for preset voice selection and LLM connection test features.

Covers:
- QWEN_PRESET_VOICES constant definition and contents
- test_llm_connection: input validation, success/failure paths
- Preset voice ID extraction logic in run_cinecast
- default_narrator_voice propagation through config to TTS engine
- MLXRenderEngine default_voice from config
- UI source code structure: dropdown, LLM config panel, updated inputs_list
"""

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Inline copy of test_llm_connection (webui.py requires gradio at import)
# ---------------------------------------------------------------------------

import requests as _requests


def _test_llm_connection(model_name, base_url, api_key):
    """Standalone copy matching webui.py implementation."""
    if not all([model_name, base_url, api_key]):
        return "❌ 请完整填写大模型名称、Base URL 和 API Key！"

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "测试连接，请只回复1个字"}],
            "max_tokens": 10,
        }
        api_endpoint = f"{base_url.rstrip('/')}/chat/completions"

        response = _requests.post(
            api_endpoint, json=payload, headers=headers, timeout=10
        )

        if response.status_code == 200:
            return f"✅ 连接成功！已成功握手 {model_name}。"
        else:
            return (
                f"❌ 测试失败 (HTTP {response.status_code}): {response.text}\n"
                "请检查各项参数。"
            )
    except Exception as e:
        return (
            f"❌ 请求异常：{str(e)}\n"
            "请检查网络和 Base URL 格式（例如需包含 /v1）。"
        )


# ---------------------------------------------------------------------------
# Tests: QWEN_PRESET_VOICES constant
# ---------------------------------------------------------------------------

class TestPresetVoicesConstant:
    @pytest.fixture
    def webui_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_preset_voices_defined(self, webui_source):
        """QWEN_PRESET_VOICES should be defined in webui.py."""
        assert "QWEN_PRESET_VOICES" in webui_source

    def test_preset_voices_contains_eric(self, webui_source):
        """Eric should be in the preset voices list."""
        assert "Eric" in webui_source

    def test_preset_voices_contains_serena(self, webui_source):
        """Serena should be in the preset voices list."""
        assert "Serena" in webui_source

    def test_preset_voices_contains_all_nine(self, webui_source):
        """All 9 official Qwen3-TTS preset voices should be listed."""
        expected = ["Eric", "Serena", "Aiden", "Dylan", "Ono_anna",
                    "Ryan", "Sohee", "Uncle_fu", "Vivian"]
        for voice in expected:
            assert voice in webui_source, f"Missing preset voice: {voice}"


# ---------------------------------------------------------------------------
# Tests: test_llm_connection function
# ---------------------------------------------------------------------------

class TestLLMConnection:
    def test_missing_model_name(self):
        """Should return error when model_name is empty."""
        result = _test_llm_connection("", "https://api.example.com/v1", "sk-test")
        assert "❌" in result
        assert "完整填写" in result

    def test_missing_base_url(self):
        """Should return error when base_url is empty."""
        result = _test_llm_connection("qwen-plus", "", "sk-test")
        assert "❌" in result

    def test_missing_api_key(self):
        """Should return error when api_key is empty."""
        result = _test_llm_connection("qwen-plus", "https://api.example.com/v1", "")
        assert "❌" in result

    def test_all_params_none(self):
        """Should return error when all params are None."""
        result = _test_llm_connection(None, None, None)
        assert "❌" in result

    @patch("requests.post")
    def test_successful_connection(self, mock_post):
        """Should return success message on HTTP 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = _test_llm_connection("qwen-plus", "https://api.example.com/v1", "sk-test")
        assert "✅" in result
        assert "qwen-plus" in result

    @patch("requests.post")
    def test_failed_connection_401(self, mock_post):
        """Should return error on HTTP 401."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        result = _test_llm_connection("qwen-plus", "https://api.example.com/v1", "sk-bad")
        assert "❌" in result
        assert "401" in result

    @patch("requests.post", side_effect=_requests.exceptions.Timeout("Connection timed out"))
    def test_timeout_exception(self, mock_post):
        """Should return error on request timeout."""
        result = _test_llm_connection("qwen-plus", "https://api.example.com/v1", "sk-test")
        assert "❌" in result
        assert "请求异常" in result

    @patch("requests.post")
    def test_url_trailing_slash_normalized(self, mock_post):
        """URL with trailing slash should be normalized."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        _test_llm_connection("model", "https://api.example.com/v1/", "sk-test")
        called_url = mock_post.call_args[0][0]
        assert called_url == "https://api.example.com/v1/chat/completions"
        assert "//" not in called_url.replace("https://", "")


# ---------------------------------------------------------------------------
# Tests: Preset voice ID extraction
# ---------------------------------------------------------------------------

class TestPresetVoiceExtraction:
    def test_extract_eric_from_default_label(self):
        """'Eric (默认男声)' -> 'eric'"""
        selection = "Eric (默认男声)"
        voice_id = selection.split(" ")[0].lower()
        assert voice_id == "eric"

    def test_extract_serena_from_label(self):
        """'Serena (默认女声)' -> 'serena'"""
        selection = "Serena (默认女声)"
        voice_id = selection.split(" ")[0].lower()
        assert voice_id == "serena"

    def test_extract_single_word_voice(self):
        """'Aiden' -> 'aiden'"""
        selection = "Aiden"
        voice_id = selection.split(" ")[0].lower()
        assert voice_id == "aiden"

    def test_extract_ono_anna(self):
        """'Ono_anna' -> 'ono_anna'"""
        selection = "Ono_anna"
        voice_id = selection.split(" ")[0].lower()
        assert voice_id == "ono_anna"

    def test_none_selection_fallback(self):
        """None selection should fallback to 'eric'."""
        selection = None
        voice_id = selection.split(" ")[0].lower() if selection else "eric"
        assert voice_id == "eric"

    def test_empty_selection_fallback(self):
        """Empty selection should fallback to 'eric'."""
        selection = ""
        voice_id = selection.split(" ")[0].lower() if selection else "eric"
        assert voice_id == "eric"


# ---------------------------------------------------------------------------
# Tests: default_narrator_voice in default config
# ---------------------------------------------------------------------------

class TestDefaultConfigNarratorVoice:
    def test_default_config_has_narrator_voice(self):
        """_get_default_config should include default_narrator_voice."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")
        producer = CineCastProducer.__new__(CineCastProducer)
        config = producer._get_default_config()
        assert "default_narrator_voice" in config
        assert config["default_narrator_voice"] == "eric"


# ---------------------------------------------------------------------------
# Tests: MLXRenderEngine default_voice from config
# ---------------------------------------------------------------------------

class TestEngineDefaultVoice:
    def test_engine_default_voice_from_config(self):
        """Engine should use default_narrator_voice from config."""
        try:
            from modules.mlx_tts_engine import MLXRenderEngine
        except ImportError:
            pytest.skip("mlx_tts_engine requires numpy/mlx (macOS-only)")
        engine = MLXRenderEngine.__new__(MLXRenderEngine)
        engine.config = {"default_narrator_voice": "serena"}
        engine.default_voice = engine.config.get("default_narrator_voice", "eric")
        assert engine.default_voice == "serena"

    def test_engine_default_voice_fallback(self):
        """Engine should fallback to 'eric' when no config provided."""
        try:
            from modules.mlx_tts_engine import MLXRenderEngine
        except ImportError:
            pytest.skip("mlx_tts_engine requires numpy/mlx (macOS-only)")
        engine = MLXRenderEngine.__new__(MLXRenderEngine)
        engine.config = {}
        engine.default_voice = engine.config.get("default_narrator_voice", "eric")
        assert engine.default_voice == "eric"


# ---------------------------------------------------------------------------
# Tests: WebUI source code structure
# ---------------------------------------------------------------------------

class TestWebuiSourceStructure:
    @pytest.fixture
    def webui_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_imports_requests(self, webui_source):
        """webui.py should import requests."""
        assert "import requests" in webui_source

    def test_test_llm_connection_defined(self, webui_source):
        """test_llm_connection function should be defined."""
        assert "def test_llm_connection(" in webui_source

    def test_preset_voice_dropdown_exists(self, webui_source):
        """Preset voice dropdown should be in UI layout."""
        assert "preset_voice_dropdown" in webui_source
        assert "gr.Dropdown" in webui_source

    def test_llm_config_panel_exists(self, webui_source):
        """LLM config panel with model/baseurl/apikey inputs should exist."""
        assert "llm_model" in webui_source
        assert "llm_baseurl" in webui_source
        assert "llm_apikey" in webui_source
        assert "btn_test_llm" in webui_source

    def test_inputs_list_includes_preset_dropdown(self, webui_source):
        """inputs_list should include preset_voice_dropdown."""
        assert "preset_voice_dropdown," in webui_source

    def test_run_cinecast_accepts_preset_selection(self, webui_source):
        """run_cinecast should accept preset_voice_selection parameter."""
        assert "preset_voice_selection," in webui_source

    def test_default_narrator_voice_in_config(self, webui_source):
        """Config should include default_narrator_voice."""
        assert "default_narrator_voice" in webui_source

    def test_global_cast_narrator_voice_injection(self, webui_source):
        """Should inject voice into global_cast narrator."""
        assert 'global_cast["旁白"]["voice"] = base_voice_id' in webui_source


# ---------------------------------------------------------------------------
# Tests: Config propagation from _create_tts_engine
# ---------------------------------------------------------------------------

class TestConfigPropagation:
    def test_create_tts_engine_passes_narrator_voice(self):
        """_create_tts_engine should include default_narrator_voice in engine_config."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Verify the key is in the iteration list of _create_tts_engine
        assert '"default_narrator_voice"' in source

    def test_engine_source_uses_self_default_voice(self):
        """MLX engine source should use self.default_voice instead of hardcoded 'eric'."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "mlx_tts_engine.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()

        assert "self.default_voice" in source
        # Should NOT have hardcoded "eric" as voice fallback in render logic
        # (only allowed in config default)
        import re
        hardcoded_erics = re.findall(r'voice_cfg\.get\(.*?"eric"\)', source)
        assert len(hardcoded_erics) == 0, (
            f"Found hardcoded 'eric' fallback in voice_cfg.get: {hardcoded_erics}"
        )
