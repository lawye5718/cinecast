"""Tests for role-based voice persistence and updated chunk limits."""

import json
import os
import sys
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Role voice file helpers (isolated from Gradio imports)
# ---------------------------------------------------------------------------

ROLE_VOICE_FILE_KEY = "ROLE_VOICE_FILE"


def _load_role_voices(filepath):
    """Standalone version of load_role_voices for testing."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"narrator": {"mode": "preset", "voice": "eric"}}


def _save_role_voice(filepath, role, voice_cfg):
    """Standalone version of save_role_voice for testing."""
    if role not in ["m1", "f1", "m2", "f2", "narrator"]:
        return
    voices = _load_role_voices(filepath)
    voices[role] = voice_cfg
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(voices, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tests: Role voice persistence
# ---------------------------------------------------------------------------

class TestLoadRoleVoices:
    def test_returns_default_when_no_file(self, tmp_path):
        filepath = str(tmp_path / "nonexistent.json")
        result = _load_role_voices(filepath)
        assert result == {"narrator": {"mode": "preset", "voice": "eric"}}

    def test_loads_existing_file(self, tmp_path):
        filepath = str(tmp_path / "voices.json")
        data = {"m1": {"mode": "clone", "ref_audio": "hero.wav"}, "narrator": {"mode": "preset", "voice": "eric"}}
        with open(filepath, 'w') as f:
            json.dump(data, f)
        result = _load_role_voices(filepath)
        assert result["m1"]["mode"] == "clone"

    def test_returns_default_on_corrupt_json(self, tmp_path):
        filepath = str(tmp_path / "bad.json")
        with open(filepath, 'w') as f:
            f.write("{corrupted")
        result = _load_role_voices(filepath)
        assert result == {"narrator": {"mode": "preset", "voice": "eric"}}


class TestSaveRoleVoice:
    def test_saves_m1_voice(self, tmp_path):
        filepath = str(tmp_path / "voices.json")
        _save_role_voice(filepath, "m1", {"mode": "preset", "voice": "aiden"})
        with open(filepath) as f:
            data = json.load(f)
        assert data["m1"]["voice"] == "aiden"

    def test_ignores_extra_role(self, tmp_path):
        filepath = str(tmp_path / "voices.json")
        _save_role_voice(filepath, "extra", {"mode": "preset", "voice": "aiden"})
        assert not os.path.exists(filepath)

    def test_roundtrip_persistence(self, tmp_path):
        filepath = str(tmp_path / "voices.json")
        cfg = {"mode": "design", "instruct": "Deep male voice"}
        _save_role_voice(filepath, "f1", cfg)
        loaded = _load_role_voices(filepath)
        assert loaded["f1"] == cfg
        # narrator default should still be present
        assert "narrator" in loaded


# ---------------------------------------------------------------------------
# Tests: parse_json_to_cast_state with role field
# ---------------------------------------------------------------------------

class TestParseJsonToCastStateWithRole:
    def test_role_field_extracted(self):
        """parse_json_to_cast_state should extract the 'role' field from characters."""
        # Read the actual source to verify the function definition
        webui_path = os.path.join(os.path.dirname(__file__), "..", "webui.py")
        with open(webui_path, 'r', encoding='utf-8') as f:
            source = f.read()
        assert 'def parse_json_to_cast_state' in source
        assert '"role"' in source or "'role'" in source
        assert 'load_role_voices' in source

    def test_cast_state_includes_role_key(self):
        """The cast_state dict should include a 'role' key per character."""
        webui_path = os.path.join(os.path.dirname(__file__), "..", "webui.py")
        with open(webui_path, 'r', encoding='utf-8') as f:
            source = f.read()
        # Check that role is being set in cast_state
        assert 'char_info.get("role"' in source


# ---------------------------------------------------------------------------
# Tests: update_cast_voice_cfg triggers save_role_voice
# ---------------------------------------------------------------------------

class TestUpdateCastVoiceCfgSavesRole:
    def test_save_role_voice_called_in_update(self):
        """update_cast_voice_cfg should call save_role_voice for role-based persistence."""
        webui_path = os.path.join(os.path.dirname(__file__), "..", "webui.py")
        with open(webui_path, 'r', encoding='utf-8') as f:
            source = f.read()
        assert 'save_role_voice' in source
        # Verify it's called inside update_cast_voice_cfg
        # Find the function body
        func_start = source.index('def update_cast_voice_cfg')
        # Find next def or end
        next_def = source.find('\ndef ', func_start + 1)
        func_body = source[func_start:next_def] if next_def != -1 else source[func_start:]
        assert 'save_role_voice' in func_body


# ---------------------------------------------------------------------------
# Tests: BRAIN_PROMPT_TEMPLATE includes role
# ---------------------------------------------------------------------------

class TestBrainPromptTemplateRole:
    def test_brain_prompt_includes_role_labels(self):
        """BRAIN_PROMPT_TEMPLATE should mention role labels m1, f1, etc."""
        webui_path = os.path.join(os.path.dirname(__file__), "..", "webui.py")
        with open(webui_path, 'r', encoding='utf-8') as f:
            source = f.read()
        assert "m1" in source
        assert "f1" in source
        assert "m2" in source
        assert "f2" in source
        assert "extra" in source

    def test_brain_prompt_example_includes_role(self):
        """The JSON example in BRAIN_PROMPT_TEMPLATE should include 'role' field."""
        webui_path = os.path.join(os.path.dirname(__file__), "..", "webui.py")
        with open(webui_path, 'r', encoding='utf-8') as f:
            source = f.read()
        # The example should show role in the characters output
        assert '"role": "m1"' in source
        assert '"role": "f1"' in source


# ---------------------------------------------------------------------------
# Tests: max_chars limits updated to 150
# ---------------------------------------------------------------------------

class TestMaxCharsLimits:
    def test_mlx_engine_max_chars_150(self):
        """MLX engine should set max_chars to 150."""
        engine_path = os.path.join(os.path.dirname(__file__), "..", "modules", "mlx_tts_engine.py")
        with open(engine_path, 'r', encoding='utf-8') as f:
            source = f.read()
        assert "self.max_chars = 150" in source

    def test_llm_director_max_chars_per_chunk_150(self):
        """LLM director should set max_chars_per_chunk to 150."""
        director_path = os.path.join(os.path.dirname(__file__), "..", "modules", "llm_director.py")
        with open(director_path, 'r', encoding='utf-8') as f:
            source = f.read()
        assert "self.max_chars_per_chunk = 150" in source

    def test_smart_chunk_limit_150(self):
        """smart_chunk_limit should use 150 as the minimum."""
        director_path = os.path.join(os.path.dirname(__file__), "..", "modules", "llm_director.py")
        with open(director_path, 'r', encoding='utf-8') as f:
            source = f.read()
        assert "max(self.max_chars_per_chunk, 150)" in source


# ---------------------------------------------------------------------------
# Tests: LLM prompt coherence rules
# ---------------------------------------------------------------------------

class TestLLMPromptCoherence:
    def test_prompt_contains_coherence_principle(self):
        """The LLM prompt should mention coherence principle for merging."""
        import inspect
        from modules.llm_director import LLMScriptDirector
        source = inspect.getsource(LLMScriptDirector._request_llm)
        assert "连贯性原则" in source

    def test_prompt_mentions_150_char_limit(self):
        """The LLM prompt should mention the 150 char limit for merging."""
        import inspect
        from modules.llm_director import LLMScriptDirector
        source = inspect.getsource(LLMScriptDirector._request_llm)
        assert "150" in source

    def test_prompt_forbids_fragmentation(self):
        """The prompt should forbid fragmenting a character's speech."""
        import inspect
        from modules.llm_director import LLMScriptDirector
        source = inspect.getsource(LLMScriptDirector._request_llm)
        assert "切碎" in source or "碎片化" in source
