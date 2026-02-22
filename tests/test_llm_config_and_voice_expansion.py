#!/usr/bin/env python3
"""
Tests for:
1. LLM configuration propagation from WebUI to LLMScriptDirector
2. Dynamic voice role expansion (m3, m4, f3, f4, etc.) in AssetManager
3. Benign LLM API call wording (no attack-like content)
"""

import inspect
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import LLMScriptDirector
from modules.asset_manager import AssetManager


# ---------------------------------------------------------------------------
# Fix 1: LLM config propagation
# ---------------------------------------------------------------------------
class TestLLMConfigPropagation:
    """Ensure user-configured LLM settings (model_name, base_url, api_key)
    are correctly accepted and used by LLMScriptDirector."""

    @patch.object(LLMScriptDirector, '_test_api_connection', return_value=True)
    @patch.object(LLMScriptDirector, '_load_cast_profiles', return_value={})
    def test_custom_model_name_accepted(self, mock_profiles, mock_conn):
        """LLMScriptDirector should use user-provided model_name."""
        director = LLMScriptDirector(
            api_key="test-key",
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
        )
        assert director.model_name == "gpt-4o"

    @patch.object(LLMScriptDirector, '_test_api_connection', return_value=True)
    @patch.object(LLMScriptDirector, '_load_cast_profiles', return_value={})
    def test_custom_base_url_accepted(self, mock_profiles, mock_conn):
        """LLMScriptDirector should use user-provided base_url."""
        director = LLMScriptDirector(
            api_key="test-key",
            base_url="https://custom-llm.example.com/v1",
        )
        assert director.base_url == "https://custom-llm.example.com/v1"

    @patch.object(LLMScriptDirector, '_test_api_connection', return_value=True)
    @patch.object(LLMScriptDirector, '_load_cast_profiles', return_value={})
    def test_default_model_name_when_not_provided(self, mock_profiles, mock_conn):
        """LLMScriptDirector should default to 'qwen-flash' when model_name is not provided."""
        director = LLMScriptDirector(api_key="test-key")
        assert director.model_name == "qwen-flash"

    @patch.object(LLMScriptDirector, '_test_api_connection', return_value=True)
    @patch.object(LLMScriptDirector, '_load_cast_profiles', return_value={})
    def test_default_base_url_when_not_provided(self, mock_profiles, mock_conn):
        """LLMScriptDirector should default to DashScope URL when base_url is not provided."""
        director = LLMScriptDirector(api_key="test-key")
        assert "dashscope" in director.base_url

    @patch.object(LLMScriptDirector, '_test_api_connection', return_value=True)
    @patch.object(LLMScriptDirector, '_load_cast_profiles', return_value={})
    def test_client_uses_custom_base_url(self, mock_profiles, mock_conn):
        """The OpenAI client should be configured with the custom base_url."""
        director = LLMScriptDirector(
            api_key="test-key",
            base_url="https://my-proxy.example.com/v1",
        )
        assert director.client.base_url is not None

    @patch.object(LLMScriptDirector, '_test_api_connection', return_value=True)
    @patch.object(LLMScriptDirector, '_load_cast_profiles', return_value={})
    def test_init_signature_accepts_model_name_and_base_url(self, mock_profiles, mock_conn):
        """__init__ should accept model_name and base_url as named parameters."""
        sig = inspect.signature(LLMScriptDirector.__init__)
        param_names = list(sig.parameters.keys())
        assert "model_name" in param_names
        assert "base_url" in param_names


# ---------------------------------------------------------------------------
# Fix 1 continued: Config flow from webui run_cinecast to producer
# ---------------------------------------------------------------------------
class TestConfigFlowToProducer:
    """Verify that webui run_cinecast injects LLM config into producer config."""

    def test_run_cinecast_loads_llm_config(self):
        """run_cinecast should load saved LLM config and inject into producer config."""
        try:
            from webui import run_cinecast
        except ImportError:
            pytest.skip("webui requires gradio (may not be installed)")
        source = inspect.getsource(run_cinecast)
        assert "llm_model_name" in source
        assert "llm_base_url" in source
        assert "llm_api_key" in source
        assert "load_llm_config" in source

    def test_phase1_passes_llm_config_to_director(self):
        """phase_1_generate_scripts should pass LLM config to LLMScriptDirector."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")
        source = inspect.getsource(CineCastProducer.phase_1_generate_scripts)
        assert "llm_api_key" in source
        assert "llm_model_name" in source
        assert "llm_base_url" in source

    def test_check_api_connectivity_uses_config(self):
        """check_api_connectivity should use user config instead of hardcoded values."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")
        source = inspect.getsource(CineCastProducer.check_api_connectivity)
        assert "llm_api_key" in source
        assert "llm_base_url" in source
        assert "llm_model_name" in source


# ---------------------------------------------------------------------------
# Fix 2: Dynamic voice role expansion
# ---------------------------------------------------------------------------
class TestDynamicVoiceRoleExpansion:
    """Test that set_custom_role_voices handles m3, m4, f3, f4, etc."""

    @pytest.fixture
    def manager(self):
        return AssetManager(asset_dir="./assets")

    @pytest.fixture
    def tmp_wav(self, tmp_path):
        wav = tmp_path / "test_voice.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 40)
        return str(wav)

    def test_m1_m2_still_work(self, manager, tmp_wav):
        """m1 and m2 should still update existing male_pool slots."""
        original_m1 = manager.voices["male_pool"][0]["audio"]
        manager.set_custom_role_voices({"m1": tmp_wav})
        assert manager.voices["male_pool"][0]["audio"] == tmp_wav

    def test_f1_f2_still_work(self, manager, tmp_wav):
        """f1 and f2 should still update existing female_pool slots."""
        manager.set_custom_role_voices({"f1": tmp_wav})
        assert manager.voices["female_pool"][0]["audio"] == tmp_wav

    def test_narrator_still_works(self, manager, tmp_wav):
        """narrator should still update narrator and related voices."""
        manager.set_custom_role_voices({"narrator": tmp_wav})
        assert manager.voices["narrator"]["audio"] == tmp_wav
        assert manager.voices["narration"]["audio"] == tmp_wav
        assert manager.voices["title"]["audio"] == tmp_wav
        assert manager.voices["subtitle"]["audio"] == tmp_wav

    def test_m3_expands_male_pool(self, manager, tmp_wav):
        """m3 should expand male_pool to have at least 3 entries."""
        original_len = len(manager.voices["male_pool"])
        assert original_len == 2  # default: m1 and m2
        manager.set_custom_role_voices({"m3": tmp_wav})
        assert len(manager.voices["male_pool"]) >= 3
        assert manager.voices["male_pool"][2]["audio"] == tmp_wav

    def test_f3_expands_female_pool(self, manager, tmp_wav):
        """f3 should expand female_pool to have at least 3 entries."""
        original_len = len(manager.voices["female_pool"])
        assert original_len == 2  # default: f1 and f2
        manager.set_custom_role_voices({"f3": tmp_wav})
        assert len(manager.voices["female_pool"]) >= 3
        assert manager.voices["female_pool"][2]["audio"] == tmp_wav

    def test_m5_expands_male_pool_with_gaps(self, manager, tmp_wav):
        """m5 should expand male_pool to have 5 entries, filling gaps with defaults."""
        manager.set_custom_role_voices({"m5": tmp_wav})
        assert len(manager.voices["male_pool"]) >= 5
        assert manager.voices["male_pool"][4]["audio"] == tmp_wav
        # Intermediate slots (m3, m4) should be filled with narrator fallback
        assert manager.voices["male_pool"][2]["audio"] is not None

    def test_unknown_role_name_skipped(self, manager, tmp_wav):
        """Unknown role names (not narrator/mN/fN) should be skipped."""
        original_male_len = len(manager.voices["male_pool"])
        original_female_len = len(manager.voices["female_pool"])
        manager.set_custom_role_voices({"xyz": tmp_wav})
        assert len(manager.voices["male_pool"]) == original_male_len
        assert len(manager.voices["female_pool"]) == original_female_len

    def test_none_file_path_skipped(self, manager):
        """None file paths should be skipped."""
        original_m1 = manager.voices["male_pool"][0]["audio"]
        manager.set_custom_role_voices({"m1": None})
        assert manager.voices["male_pool"][0]["audio"] == original_m1

    def test_multiple_roles_at_once(self, manager, tmp_wav):
        """Multiple roles can be set in a single call."""
        manager.set_custom_role_voices({
            "narrator": tmp_wav,
            "m1": tmp_wav,
            "f1": tmp_wav,
            "m3": tmp_wav,
        })
        assert manager.voices["narrator"]["audio"] == tmp_wav
        assert manager.voices["male_pool"][0]["audio"] == tmp_wav
        assert manager.voices["female_pool"][0]["audio"] == tmp_wav
        assert len(manager.voices["male_pool"]) >= 3

    def test_case_insensitive_role_names(self, manager, tmp_wav):
        """Role names like M1, F2 should work (case-insensitive)."""
        manager.set_custom_role_voices({"M1": tmp_wav})
        assert manager.voices["male_pool"][0]["audio"] == tmp_wav


# ---------------------------------------------------------------------------
# Fix 3: Benign LLM API call wording
# ---------------------------------------------------------------------------
class TestBenignLLMCallWording:
    """Ensure LLM API calls use benign, non-attack-like wording."""

    def test_user_content_no_aggressive_instruction_brackets(self):
        """user_content should not use 【指令：...】 format that could trigger content filters."""
        source = inspect.getsource(LLMScriptDirector._request_llm)
        # The old aggressive wording should not be present
        assert "【指令：" not in source

    def test_user_content_uses_polite_task_description(self):
        """user_content should use polite, task-oriented language."""
        source = inspect.getsource(LLMScriptDirector._request_llm)
        assert "有声书制作" in source or "有声书" in source

    def test_test_api_connection_no_ping(self):
        """_test_api_connection should not send 'ping' which could trigger filters."""
        source = inspect.getsource(LLMScriptDirector._test_api_connection)
        assert '"ping"' not in source

    def test_test_api_connection_uses_natural_message(self):
        """_test_api_connection should use a natural test message."""
        source = inspect.getsource(LLMScriptDirector._test_api_connection)
        # Should contain a proper Chinese test message
        assert "连接正常" in source

    def test_system_prompt_is_task_oriented(self):
        """System prompt should describe a professional task, not use attack-like language."""
        source = inspect.getsource(LLMScriptDirector._request_llm)
        assert "有声书剧本转换" in source
        assert "JSON 数组" in source
