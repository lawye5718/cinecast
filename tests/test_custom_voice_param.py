#!/usr/bin/env python3
"""
Tests for CustomVoice 'voice' parameter fix.

The CustomVoice model requires a 'voice' parameter (speaker name) to be passed
in generate_kwargs. Previously the code used 'speaker' as the key name, which
the MLX library does not recognize, and default voice configs omitted the key
entirely, causing all preset-mode renders to fail.

These tests verify:
1. The engine source passes 'voice' (not 'speaker') to model.generate()
2. A default voice name is used when voice_cfg has no explicit voice/speaker
3. The asset_manager uses 'voice' in build_voice_profile for preset mode
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MLX_ENGINE_PATH = os.path.join(_PROJECT_ROOT, "modules", "mlx_tts_engine.py")
_ASSET_MANAGER_PATH = os.path.join(_PROJECT_ROOT, "modules", "asset_manager.py")


def _read_engine_source():
    with open(_MLX_ENGINE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _read_asset_manager_source():
    with open(_ASSET_MANAGER_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 1. Engine passes 'voice' parameter to model.generate in preset mode
# ---------------------------------------------------------------------------

class TestCustomVoiceParameter:
    """Verify the engine uses 'voice' key for CustomVoice model."""

    def test_generate_kwargs_uses_voice_key(self):
        """generate_kwargs must use 'voice' (not 'speaker') for CustomVoice."""
        source = _read_engine_source()
        # Find the preset mode block
        preset_idx = source.index("# 传统 Preset 模式")
        # Get a reasonable block after the comment
        block = source[preset_idx:preset_idx + 600]
        assert 'generate_kwargs["voice"]' in block

    def test_no_speaker_key_in_generate_kwargs(self):
        """generate_kwargs must NOT use 'speaker' as key passed to generate()."""
        source = _read_engine_source()
        preset_idx = source.index("# 传统 Preset 模式")
        block = source[preset_idx:preset_idx + 600]
        assert 'generate_kwargs["speaker"]' not in block

    def test_default_voice_provided(self):
        """A default voice name must be set when voice_cfg has no voice key."""
        source = _read_engine_source()
        # DEFAULT_CUSTOM_VOICE constant should be defined at module level
        assert "DEFAULT_CUSTOM_VOICE" in source
        preset_idx = source.index("# 传统 Preset 模式")
        block = source[preset_idx:preset_idx + 600]
        assert "DEFAULT_CUSTOM_VOICE" in block

    def test_supports_legacy_speaker_field(self):
        """voice_cfg with legacy 'speaker' field should still be read."""
        source = _read_engine_source()
        preset_idx = source.index("# 传统 Preset 模式")
        block = source[preset_idx:preset_idx + 600]
        assert 'voice_cfg.get("speaker")' in block

    def test_supports_voice_field(self):
        """voice_cfg with 'voice' field should be read."""
        source = _read_engine_source()
        preset_idx = source.index("# 传统 Preset 模式")
        block = source[preset_idx:preset_idx + 600]
        assert 'voice_cfg.get("voice")' in block


# ---------------------------------------------------------------------------
# 2. Asset manager uses 'voice' key in build_voice_profile
# ---------------------------------------------------------------------------

class TestAssetManagerVoiceKey:
    """Verify asset_manager uses 'voice' in preset profile."""

    def test_build_voice_profile_uses_voice_key(self):
        """build_voice_profile must use 'voice' key for preset mode."""
        source = _read_asset_manager_source()
        profile_idx = source.index("def build_voice_profile")
        # Find the end of build_voice_profile method
        next_def = source.find("\n    def ", profile_idx + 30)
        block = source[profile_idx:next_def] if next_def != -1 else source[profile_idx:]
        assert '"voice": speaker_id' in block

    def test_build_voice_profile_no_speaker_key(self):
        """build_voice_profile must NOT use 'speaker' key in the profile dict."""
        source = _read_asset_manager_source()
        profile_idx = source.index("def build_voice_profile")
        next_def = source.find("\n    def ", profile_idx + 30)
        block = source[profile_idx:next_def] if next_def != -1 else source[profile_idx:]
        assert '"speaker": speaker_id' not in block
