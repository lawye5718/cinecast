#!/usr/bin/env python3
"""
Tests for the base voice parameter fix in MLX TTS engine.

Verifies that:
- CustomVoice model receives 'voice' instead of 'speaker' in preset mode
- Default fallback 'Ethan' is used when neither 'voice' nor 'speaker' is set
- Clone mode defensively passes 'voice' when available
- Clone mode uses .get() for safe key access on ref_audio/ref_text
- Old 'speaker' field is no longer passed directly to generate()
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MLX_ENGINE_PATH = os.path.join(_PROJECT_ROOT, "modules", "mlx_tts_engine.py")


def _read_source():
    with open(_MLX_ENGINE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _get_render_func_body():
    """Extract the render_dry_chunk method body from source."""
    source = _read_source()
    start = source.index("def render_dry_chunk")
    return source[start:]


# ---------------------------------------------------------------------------
# 1. Preset mode: voice parameter mapping
# ---------------------------------------------------------------------------

class TestPresetVoiceMapping:
    """Verify preset mode maps speaker→voice and has default fallback."""

    def test_voice_get_with_speaker_fallback(self):
        """Preset mode should use voice_cfg.get('voice', voice_cfg.get('speaker', 'Ethan'))."""
        body = _get_render_func_body()
        assert 'voice_cfg.get("voice", voice_cfg.get("speaker", "Ethan"))' in body

    def test_generate_kwargs_uses_voice_key(self):
        """Preset mode should set generate_kwargs['voice'], not generate_kwargs['speaker']."""
        body = _get_render_func_body()
        assert 'generate_kwargs["voice"] = target_voice' in body

    def test_no_speaker_passed_to_generate(self):
        """The old generate_kwargs['speaker'] pattern must be removed."""
        body = _get_render_func_body()
        assert 'generate_kwargs["speaker"]' not in body

    def test_default_ethan_fallback(self):
        """Default voice fallback 'Ethan' must be present for safety."""
        body = _get_render_func_body()
        assert '"Ethan"' in body

    def test_ref_audio_optional_in_preset(self):
        """Preset mode should only add ref_audio if 'audio' key exists and is truthy."""
        body = _get_render_func_body()
        assert '"audio" in voice_cfg and voice_cfg["audio"]' in body

    def test_ref_text_optional_in_preset(self):
        """Preset mode should only add ref_text if 'text' key exists and is truthy."""
        body = _get_render_func_body()
        assert '"text" in voice_cfg and voice_cfg["text"]' in body


# ---------------------------------------------------------------------------
# 2. Clone mode: defensive voice parameter
# ---------------------------------------------------------------------------

class TestCloneVoiceDefense:
    """Verify clone mode defensively passes voice when available."""

    def test_clone_uses_get_for_ref_audio(self):
        """Clone mode should use .get() for ref_audio with fallback to 'audio' key."""
        body = _get_render_func_body()
        assert 'voice_cfg.get("ref_audio", voice_cfg.get("audio", ""))' in body

    def test_clone_uses_get_for_ref_text(self):
        """Clone mode should use .get() for ref_text with fallback to 'text' key."""
        body = _get_render_func_body()
        assert 'voice_cfg.get("ref_text", voice_cfg.get("text", ""))' in body

    def test_clone_adds_voice_when_speaker_present(self):
        """Clone mode should add voice param if speaker or voice is in voice_cfg."""
        body = _get_render_func_body()
        assert '"speaker" in voice_cfg or "voice" in voice_cfg' in body

    def test_clone_voice_fallback(self):
        """Clone mode voice fallback should also use Ethan as default."""
        # Find the clone-specific block
        body = _get_render_func_body()
        clone_idx = body.index('mode == "clone"')
        # Find the next elif/else to bound the clone block
        design_idx = body.index('mode == "design"', clone_idx)
        clone_block = body[clone_idx:design_idx]
        assert 'voice_cfg.get("voice", voice_cfg.get("speaker", "Ethan"))' in clone_block
