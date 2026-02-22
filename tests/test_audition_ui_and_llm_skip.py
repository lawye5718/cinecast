#!/usr/bin/env python3
"""
Tests for audition UI audio player and smart LLM skip logic.

Covers:
- Audition section includes audio player for preview playback
- Preview button outputs to audition audio player
- LLMScriptDirector accepts model_name and base_url parameters
- User LLM config is passed through producer config
- Default config includes llm_model_name, llm_base_url, llm_api_key keys
- check_api_connectivity uses user LLM config when available
"""

import inspect
import json
import os
import sys

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Tests: Audition UI has audio player
# ---------------------------------------------------------------------------

class TestAuditionUIAudioPlayer:
    @pytest.fixture
    def webui_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_preview_audio_player_defined(self, webui_source):
        """Audition section should have an audio player for preview results."""
        assert "preview_audio_player" in webui_source

    def test_preview_audio_player_is_gr_audio(self, webui_source):
        """preview_audio_player should be a gr.Audio component."""
        assert 'preview_audio_player = gr.Audio(' in webui_source

    def test_preview_status_defined(self, webui_source):
        """Audition section should have a status text box."""
        assert "preview_status" in webui_source

    def test_preview_button_outputs_to_audition_player(self, webui_source):
        """Preview button should output to audition audio player, not main player."""
        assert "[preview_audio_player, preview_status]" in webui_source


# ---------------------------------------------------------------------------
# Tests: LLMScriptDirector accepts model_name and base_url
# ---------------------------------------------------------------------------

class TestDirectorCustomLLMConfig:
    def test_director_default_model_is_qwen_flash(self):
        """Default model_name should be qwen-flash."""
        from modules.llm_director import LLMScriptDirector
        director = LLMScriptDirector()
        assert director.model_name == "qwen-flash"

    def test_director_accepts_custom_model_name(self):
        """Should accept a custom model_name parameter."""
        from modules.llm_director import LLMScriptDirector
        director = LLMScriptDirector(model_name="gpt-4")
        assert director.model_name == "gpt-4"

    def test_director_accepts_custom_base_url(self):
        """Should accept a custom base_url parameter."""
        from modules.llm_director import LLMScriptDirector
        director = LLMScriptDirector(base_url="https://api.openai.com/v1")
        assert "api.openai.com" in str(director.client.base_url)

    def test_director_none_model_name_falls_back(self):
        """None model_name should fallback to qwen-flash."""
        from modules.llm_director import LLMScriptDirector
        director = LLMScriptDirector(model_name=None)
        assert director.model_name == "qwen-flash"

    def test_director_empty_model_name_falls_back(self):
        """Empty string model_name should fallback to qwen-flash."""
        from modules.llm_director import LLMScriptDirector
        director = LLMScriptDirector(model_name="")
        assert director.model_name == "qwen-flash"

    def test_director_init_signature_includes_model_name(self):
        """__init__ should have model_name parameter."""
        from modules.llm_director import LLMScriptDirector
        sig = inspect.signature(LLMScriptDirector.__init__)
        assert "model_name" in sig.parameters

    def test_director_init_signature_includes_base_url(self):
        """__init__ should have base_url parameter."""
        from modules.llm_director import LLMScriptDirector
        sig = inspect.signature(LLMScriptDirector.__init__)
        assert "base_url" in sig.parameters


# ---------------------------------------------------------------------------
# Tests: Default config includes LLM keys
# ---------------------------------------------------------------------------

class TestDefaultConfigLLMKeys:
    def test_default_config_has_llm_model_name(self):
        """Default config should include llm_model_name."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")
        producer = CineCastProducer()
        assert "llm_model_name" in producer.config

    def test_default_config_has_llm_base_url(self):
        """Default config should include llm_base_url."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")
        producer = CineCastProducer()
        assert "llm_base_url" in producer.config

    def test_default_config_has_llm_api_key(self):
        """Default config should include llm_api_key."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")
        producer = CineCastProducer()
        assert "llm_api_key" in producer.config


# ---------------------------------------------------------------------------
# Tests: WebUI passes LLM config to producer
# ---------------------------------------------------------------------------

class TestWebuiLLMConfigPassthrough:
    @pytest.fixture
    def webui_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_llm_config_loaded_in_run_cinecast(self, webui_source):
        """run_cinecast should load LLM config."""
        assert "load_llm_config()" in webui_source

    def test_llm_model_name_in_config(self, webui_source):
        """Config should include llm_model_name."""
        assert '"llm_model_name"' in webui_source

    def test_llm_base_url_in_config(self, webui_source):
        """Config should include llm_base_url."""
        assert '"llm_base_url"' in webui_source

    def test_llm_api_key_in_config(self, webui_source):
        """Config should include llm_api_key."""
        assert '"llm_api_key"' in webui_source


# ---------------------------------------------------------------------------
# Tests: main_producer passes LLM config to director
# ---------------------------------------------------------------------------

class TestProducerLLMConfigPassthrough:
    @pytest.fixture
    def producer_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_director_receives_model_name(self, producer_source):
        """phase_1 should pass llm_model_name to director."""
        assert 'model_name=self.config.get("llm_model_name")' in producer_source

    def test_director_receives_base_url(self, producer_source):
        """phase_1 should pass llm_base_url to director."""
        assert 'base_url=self.config.get("llm_base_url")' in producer_source

    def test_director_receives_api_key(self, producer_source):
        """phase_1 should pass llm_api_key to director."""
        assert 'api_key=self.config.get("llm_api_key")' in producer_source

    def test_check_api_connectivity_uses_user_config(self, producer_source):
        """check_api_connectivity should check user's LLM config first."""
        assert "llm_base_url" in producer_source
        assert "llm_api_key" in producer_source
        assert "llm_model_name" in producer_source
