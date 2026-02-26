#!/usr/bin/env python3
"""
Tests for:
1. Clone voice persistence: _persist_clone_ref_audio copies ref audio to assets/voices/
   and save_role_voice persists clone configs so they survive across sessions.
2. LLM UI config passthrough: run_cinecast accepts live UI LLM fields and
   prioritises them over saved file values. inputs_list includes llm_model,
   llm_baseurl, llm_apikey.
"""

import json
import os
import sys
import shutil
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Standalone helpers (avoid importing webui.py which requires gradio)
# ---------------------------------------------------------------------------

DEPRECATED_VOICES = {"eric", "serena"}


def _load_role_voices(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"narrator": {"mode": "preset", "voice": "aiden"}}


def _save_role_voice(filepath, role, voice_cfg):
    if role not in ["m1", "f1", "m2", "f2", "narrator"]:
        return
    voice_id = voice_cfg.get("voice", "")
    if isinstance(voice_id, str) and voice_id.lower() in DEPRECATED_VOICES:
        return
    voices = _load_role_voices(filepath)
    voices[role] = voice_cfg
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(voices, f, ensure_ascii=False, indent=2)


def _persist_clone_ref_audio(voice_cfg, role, assets_dir):
    """Standalone version matching webui._persist_clone_ref_audio."""
    if voice_cfg.get("mode") != "clone":
        return
    ref_audio = voice_cfg.get("ref_audio", "")
    if not ref_audio or not os.path.exists(ref_audio):
        return
    persistent_dir = os.path.join(assets_dir, "voices")
    os.makedirs(persistent_dir, exist_ok=True)
    ext = os.path.splitext(ref_audio)[1] or ".wav"
    persistent_path = os.path.join(persistent_dir, f"role_{role}{ext}")
    shutil.copy(ref_audio, persistent_path)
    voice_cfg["ref_audio"] = persistent_path


# ---------------------------------------------------------------------------
# Issue 1: Clone voice persistence
# ---------------------------------------------------------------------------

class TestPersistCloneRefAudio:
    def test_clone_audio_copied_to_assets(self, tmp_path):
        """Clone ref audio should be copied to assets/voices/role_<role>.wav."""
        # Create a fake temp ref audio
        src = tmp_path / "gradio_tmp" / "upload_abc123.wav"
        src.parent.mkdir()
        src.write_bytes(b"RIFF" + b"\x00" * 40)

        assets_dir = str(tmp_path / "assets")
        voice_cfg = {"mode": "clone", "ref_audio": str(src), "ref_text": ""}
        _persist_clone_ref_audio(voice_cfg, "m1", assets_dir)

        expected = os.path.join(assets_dir, "voices", "role_m1.wav")
        assert os.path.exists(expected)
        assert voice_cfg["ref_audio"] == expected

    def test_preset_mode_not_affected(self, tmp_path):
        """Preset voice configs should not trigger file copy."""
        assets_dir = str(tmp_path / "assets")
        voice_cfg = {"mode": "preset", "voice": "aiden"}
        _persist_clone_ref_audio(voice_cfg, "f1", assets_dir)
        assert voice_cfg == {"mode": "preset", "voice": "aiden"}
        assert not os.path.exists(os.path.join(assets_dir, "voices", "role_f1.wav"))

    def test_design_mode_not_affected(self, tmp_path):
        """Design voice configs should not trigger file copy."""
        assets_dir = str(tmp_path / "assets")
        voice_cfg = {"mode": "design", "instruct": "Deep male voice"}
        _persist_clone_ref_audio(voice_cfg, "m2", assets_dir)
        assert "ref_audio" not in voice_cfg

    def test_missing_ref_audio_file_skipped(self, tmp_path):
        """If ref_audio file does not exist, skip silently."""
        assets_dir = str(tmp_path / "assets")
        voice_cfg = {"mode": "clone", "ref_audio": "/nonexistent/path.wav", "ref_text": ""}
        _persist_clone_ref_audio(voice_cfg, "narrator", assets_dir)
        assert voice_cfg["ref_audio"] == "/nonexistent/path.wav"

    def test_empty_ref_audio_skipped(self, tmp_path):
        """Empty ref_audio should be skipped."""
        assets_dir = str(tmp_path / "assets")
        voice_cfg = {"mode": "clone", "ref_audio": "", "ref_text": ""}
        _persist_clone_ref_audio(voice_cfg, "m1", assets_dir)
        assert voice_cfg["ref_audio"] == ""


class TestSaveRoleVoiceCloneMode:
    def test_clone_voice_cfg_persisted(self, tmp_path):
        """Clone voice configs should be saved in role voices file."""
        filepath = str(tmp_path / "voices.json")
        cfg = {"mode": "clone", "ref_audio": "./assets/voices/role_m1.wav", "ref_text": ""}
        _save_role_voice(filepath, "m1", cfg)
        loaded = _load_role_voices(filepath)
        assert loaded["m1"]["mode"] == "clone"
        assert loaded["m1"]["ref_audio"] == "./assets/voices/role_m1.wav"

    def test_design_voice_cfg_persisted(self, tmp_path):
        """Design voice configs should be saved in role voices file."""
        filepath = str(tmp_path / "voices.json")
        cfg = {"mode": "design", "instruct": "Warm female voice"}
        _save_role_voice(filepath, "f1", cfg)
        loaded = _load_role_voices(filepath)
        assert loaded["f1"]["mode"] == "design"
        assert loaded["f1"]["instruct"] == "Warm female voice"

    def test_clone_roundtrip_across_sessions(self, tmp_path):
        """Simulates save → restart → load cycle for clone voice."""
        filepath = str(tmp_path / "voices.json")
        ref_path = str(tmp_path / "assets" / "voices" / "role_narrator.wav")
        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        with open(ref_path, 'wb') as f:
            f.write(b"RIFF" + b"\x00" * 40)

        cfg = {"mode": "clone", "ref_audio": ref_path, "ref_text": ""}
        _save_role_voice(filepath, "narrator", cfg)

        # Simulate restart: load from file
        loaded = _load_role_voices(filepath)
        assert loaded["narrator"]["mode"] == "clone"
        assert os.path.exists(loaded["narrator"]["ref_audio"])


# ---------------------------------------------------------------------------
# Issue 2: LLM UI config passthrough
# ---------------------------------------------------------------------------

class TestLLMUIConfigPassthrough:
    """Verify that webui.py passes LLM UI fields to run_cinecast."""

    @pytest.fixture
    def webui_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_run_cinecast_accepts_llm_params(self, webui_source):
        """run_cinecast should accept llm_model_name, llm_base_url, llm_api_key."""
        assert 'llm_model_name=""' in webui_source or "llm_model_name=" in webui_source
        assert 'llm_base_url=""' in webui_source or "llm_base_url=" in webui_source
        assert 'llm_api_key=""' in webui_source or "llm_api_key=" in webui_source

    def test_inputs_list_includes_llm_fields(self, webui_source):
        """inputs_list should include llm_model, llm_baseurl, llm_apikey."""
        assert "llm_model," in webui_source
        assert "llm_baseurl," in webui_source
        assert "llm_apikey," in webui_source

    def test_active_llm_values_prioritised(self, webui_source):
        """run_cinecast should prioritise UI values over saved config."""
        assert "active_llm_model" in webui_source
        assert "active_llm_base_url" in webui_source
        assert "active_llm_api_key" in webui_source

    def test_llm_config_saved_at_runtime(self, webui_source):
        """run_cinecast should save LLM config when running (not just on test)."""
        # Find run_cinecast function body
        func_start = webui_source.index('def run_cinecast(')
        next_def = webui_source.find('\ndef ', func_start + 1)
        func_body = webui_source[func_start:next_def] if next_def != -1 else webui_source[func_start:]
        assert 'save_llm_config(' in func_body


class TestPersistCloneRefAudioInWebUI:
    """Verify that _persist_clone_ref_audio exists and is called in update_cast_voice_cfg."""

    @pytest.fixture
    def webui_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_persist_function_defined(self, webui_source):
        """_persist_clone_ref_audio should be defined in webui.py."""
        assert "def _persist_clone_ref_audio(" in webui_source

    def test_persist_called_in_update(self, webui_source):
        """_persist_clone_ref_audio should be called in update_cast_voice_cfg."""
        func_start = webui_source.index('def update_cast_voice_cfg(')
        next_def = webui_source.find('\ndef ', func_start + 1)
        func_body = webui_source[func_start:next_def] if next_def != -1 else webui_source[func_start:]
        assert '_persist_clone_ref_audio(' in func_body
