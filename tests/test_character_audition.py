#!/usr/bin/env python3
"""
Tests for the Character Audition Console (角色试音与定妆室).

Covers:
- parse_json_to_cast_state: parsing Master JSON into cast state dict
- build_voice_cfg_from_ui: assembling voice_cfg from UI selections
- update_cast_voice_cfg: locking character voice and updating state
- inject_cast_state_into_global_cast: merging locked configs into global_cast
- WebUI source structure: audition panel, @gr.render, cast_state
"""

import json
import os
import sys

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Inline copies of functions from webui.py (avoid heavy import chain: mlx etc.)
# ---------------------------------------------------------------------------

def parse_json_to_cast_state(json_str):
    """Standalone copy matching webui.py implementation."""
    try:
        data = json.loads(json_str)
        characters = data.get("characters", {})
    except Exception:
        return {}

    cast_state = {}
    for char_name, char_info in characters.items():
        if not isinstance(char_info, dict):
            continue
        cast_state[char_name] = {
            "gender": char_info.get("gender", "unknown"),
            "emotion": char_info.get("emotion", "平静"),
            "locked": False,
            "voice_cfg": {
                "mode": "preset",
                "voice": "eric",
            },
        }
    return cast_state


def build_voice_cfg_from_ui(mode, preset_voice, clone_file, design_text):
    """Standalone copy matching webui.py implementation."""
    voice_cfg = {"mode": "preset", "voice": "eric"}

    if mode == "预设基底":
        voice_id = preset_voice.split(" ")[0].lower() if preset_voice else "eric"
        voice_cfg = {"mode": "preset", "voice": voice_id}
    elif mode == "声音克隆" and clone_file is not None:
        ref_path = clone_file if isinstance(clone_file, str) else getattr(clone_file, "name", "")
        voice_cfg = {"mode": "clone", "ref_audio": ref_path, "ref_text": ""}
    elif mode == "文本设计" and design_text:
        voice_cfg = {"mode": "design", "instruct": design_text}

    return voice_cfg


def update_cast_voice_cfg(cast_state, char_name, mode, preset_voice, clone_file, design_text):
    """Standalone copy matching webui.py implementation."""
    if not cast_state or char_name not in cast_state:
        return cast_state

    voice_cfg = build_voice_cfg_from_ui(mode, preset_voice, clone_file, design_text)
    cast_state[char_name]["voice_cfg"] = voice_cfg
    cast_state[char_name]["locked"] = True
    return cast_state


def unlock_cast_voice_cfg(cast_state, char_name):
    """Standalone copy matching webui.py implementation."""
    if not cast_state or char_name not in cast_state:
        return cast_state

    cast_state[char_name]["locked"] = False
    return cast_state


def inject_cast_state_into_global_cast(global_cast, cast_state):
    """Standalone copy matching webui.py implementation."""
    if not cast_state:
        return global_cast
    for char_name, info in cast_state.items():
        if info.get("locked") and char_name in global_cast:
            global_cast[char_name]["voice_cfg"] = info["voice_cfg"]
    return global_cast


# ---------------------------------------------------------------------------
# Tests: parse_json_to_cast_state
# ---------------------------------------------------------------------------

class TestParseJsonToCastState:
    def test_valid_json_extracts_characters(self):
        """Should extract characters with locked=False and default voice_cfg."""
        master = json.dumps({
            "characters": {
                "旁白": {"gender": "male", "emotion": "平静"},
                "老渔夫": {"gender": "male", "emotion": "沧桑"},
            },
            "recaps": {}
        })
        state = parse_json_to_cast_state(master)
        assert "旁白" in state
        assert "老渔夫" in state
        assert state["老渔夫"]["locked"] is False
        assert state["老渔夫"]["gender"] == "male"
        assert state["老渔夫"]["emotion"] == "沧桑"
        assert state["老渔夫"]["voice_cfg"]["mode"] == "preset"

    def test_empty_characters(self):
        """Should return empty dict when characters is empty."""
        master = json.dumps({"characters": {}, "recaps": {}})
        state = parse_json_to_cast_state(master)
        assert state == {}

    def test_invalid_json_returns_empty(self):
        """Should return empty dict on invalid JSON."""
        state = parse_json_to_cast_state("not valid json {{{")
        assert state == {}

    def test_missing_characters_key(self):
        """Should return empty dict when 'characters' key is missing."""
        master = json.dumps({"recaps": {}})
        state = parse_json_to_cast_state(master)
        assert state == {}

    def test_empty_string_returns_empty(self):
        """Should return empty dict for empty string."""
        state = parse_json_to_cast_state("")
        assert state == {}

    def test_none_returns_empty(self):
        """Should return empty dict for None input."""
        state = parse_json_to_cast_state(None)
        assert state == {}

    def test_non_dict_character_info_skipped(self):
        """Should skip characters whose info is not a dict."""
        master = json.dumps({
            "characters": {
                "旁白": {"gender": "male", "emotion": "平静"},
                "invalid": "not a dict",
            },
            "recaps": {}
        })
        state = parse_json_to_cast_state(master)
        assert "旁白" in state
        assert "invalid" not in state

    def test_default_gender_and_emotion(self):
        """Should use defaults when gender/emotion are missing."""
        master = json.dumps({
            "characters": {"角色A": {}},
            "recaps": {}
        })
        state = parse_json_to_cast_state(master)
        assert state["角色A"]["gender"] == "unknown"
        assert state["角色A"]["emotion"] == "平静"

    def test_multiple_characters(self):
        """Should handle many characters."""
        chars = {f"角色{i}": {"gender": "male", "emotion": "平静"} for i in range(20)}
        master = json.dumps({"characters": chars, "recaps": {}})
        state = parse_json_to_cast_state(master)
        assert len(state) == 20


# ---------------------------------------------------------------------------
# Tests: build_voice_cfg_from_ui
# ---------------------------------------------------------------------------

class TestBuildVoiceCfgFromUI:
    def test_preset_mode_extracts_voice_id(self):
        """Preset mode should extract lowercase voice ID from dropdown."""
        cfg = build_voice_cfg_from_ui("预设基底", "Serena (默认女声)", None, "")
        assert cfg["mode"] == "preset"
        assert cfg["voice"] == "serena"

    def test_preset_mode_single_word(self):
        """Preset mode with single-word voice name."""
        cfg = build_voice_cfg_from_ui("预设基底", "Aiden", None, "")
        assert cfg["voice"] == "aiden"

    def test_preset_mode_none_dropdown(self):
        """Preset mode with None dropdown should fallback to eric."""
        cfg = build_voice_cfg_from_ui("预设基底", None, None, "")
        assert cfg["voice"] == "eric"

    def test_clone_mode_with_file_path(self):
        """Clone mode should use file path string."""
        cfg = build_voice_cfg_from_ui("声音克隆", "Eric (默认男声)", "/path/to/ref.wav", "")
        assert cfg["mode"] == "clone"
        assert cfg["ref_audio"] == "/path/to/ref.wav"

    def test_clone_mode_with_file_object(self):
        """Clone mode should extract .name from file object."""
        class FakeFile:
            name = "/tmp/uploaded.wav"
        cfg = build_voice_cfg_from_ui("声音克隆", "Eric (默认男声)", FakeFile(), "")
        assert cfg["mode"] == "clone"
        assert cfg["ref_audio"] == "/tmp/uploaded.wav"

    def test_clone_mode_without_file_falls_back(self):
        """Clone mode without file should fallback to preset."""
        cfg = build_voice_cfg_from_ui("声音克隆", "Eric (默认男声)", None, "")
        assert cfg["mode"] == "preset"

    def test_design_mode_with_prompt(self):
        """Design mode should use instruct text."""
        cfg = build_voice_cfg_from_ui("文本设计", "Eric (默认男声)", None, "Deep male voice")
        assert cfg["mode"] == "design"
        assert cfg["instruct"] == "Deep male voice"

    def test_design_mode_without_prompt_falls_back(self):
        """Design mode without prompt should fallback to preset."""
        cfg = build_voice_cfg_from_ui("文本设计", "Eric (默认男声)", None, "")
        assert cfg["mode"] == "preset"


# ---------------------------------------------------------------------------
# Tests: update_cast_voice_cfg
# ---------------------------------------------------------------------------

class TestUpdateCastVoiceCfg:
    def _make_state(self):
        return {
            "老渔夫": {
                "gender": "male", "emotion": "沧桑",
                "locked": False,
                "voice_cfg": {"mode": "preset", "voice": "eric"},
            }
        }

    def test_locks_character(self):
        """Should set locked=True for the character."""
        state = self._make_state()
        updated = update_cast_voice_cfg(state, "老渔夫", "预设基底", "Serena (默认女声)", None, "")
        assert updated["老渔夫"]["locked"] is True

    def test_updates_voice_cfg(self):
        """Should update voice_cfg with new selection."""
        state = self._make_state()
        updated = update_cast_voice_cfg(state, "老渔夫", "预设基底", "Ryan", None, "")
        assert updated["老渔夫"]["voice_cfg"]["voice"] == "ryan"

    def test_unknown_character_no_change(self):
        """Should not crash when character not in state."""
        state = self._make_state()
        updated = update_cast_voice_cfg(state, "不存在的角色", "预设基底", "Eric (默认男声)", None, "")
        assert updated == state

    def test_empty_state_no_crash(self):
        """Should handle empty state gracefully."""
        updated = update_cast_voice_cfg({}, "角色", "预设基底", "Eric (默认男声)", None, "")
        assert updated == {}

    def test_none_state_no_crash(self):
        """Should handle None state gracefully."""
        updated = update_cast_voice_cfg(None, "角色", "预设基底", "Eric (默认男声)", None, "")
        assert updated is None


# ---------------------------------------------------------------------------
# Tests: unlock_cast_voice_cfg
# ---------------------------------------------------------------------------

class TestUnlockCastVoiceCfg:
    def _make_locked_state(self):
        return {
            "老渔夫": {
                "gender": "male", "emotion": "沧桑",
                "locked": True,
                "voice_cfg": {"mode": "preset", "voice": "ryan"},
            }
        }

    def test_unlocks_character(self):
        """Should set locked=False for the character."""
        state = self._make_locked_state()
        updated = unlock_cast_voice_cfg(state, "老渔夫")
        assert updated["老渔夫"]["locked"] is False

    def test_preserves_voice_cfg_after_unlock(self):
        """Should preserve voice_cfg when unlocking."""
        state = self._make_locked_state()
        updated = unlock_cast_voice_cfg(state, "老渔夫")
        assert updated["老渔夫"]["voice_cfg"]["voice"] == "ryan"

    def test_unlock_then_relock(self):
        """Should allow re-locking with different voice after unlock."""
        state = self._make_locked_state()
        state = unlock_cast_voice_cfg(state, "老渔夫")
        assert state["老渔夫"]["locked"] is False
        state = update_cast_voice_cfg(state, "老渔夫", "预设基底", "Serena (默认女声)", None, "")
        assert state["老渔夫"]["locked"] is True
        assert state["老渔夫"]["voice_cfg"]["voice"] == "serena"

    def test_unlock_clone_voice_preserves_ref_audio(self):
        """Should preserve clone ref_audio path after unlock."""
        state = {
            "角色A": {
                "gender": "female", "emotion": "活泼",
                "locked": True,
                "voice_cfg": {"mode": "clone", "ref_audio": "/path/to/voice.wav", "ref_text": ""},
            }
        }
        updated = unlock_cast_voice_cfg(state, "角色A")
        assert updated["角色A"]["locked"] is False
        assert updated["角色A"]["voice_cfg"]["ref_audio"] == "/path/to/voice.wav"

    def test_unknown_character_no_change(self):
        """Should not crash when character not in state."""
        state = self._make_locked_state()
        updated = unlock_cast_voice_cfg(state, "不存在的角色")
        assert updated == state

    def test_empty_state_no_crash(self):
        """Should handle empty state gracefully."""
        updated = unlock_cast_voice_cfg({}, "角色")
        assert updated == {}

    def test_none_state_no_crash(self):
        """Should handle None state gracefully."""
        updated = unlock_cast_voice_cfg(None, "角色")
        assert updated is None


# ---------------------------------------------------------------------------
# Tests: inject_cast_state_into_global_cast
# ---------------------------------------------------------------------------

class TestInjectCastState:
    def test_injects_locked_voice_cfg(self):
        """Should inject voice_cfg for locked characters."""
        global_cast = {
            "老渔夫": {"gender": "male", "emotion": "沧桑"},
            "旁白": {"gender": "male", "emotion": "平静"},
        }
        cast_state = {
            "老渔夫": {
                "locked": True,
                "voice_cfg": {"mode": "preset", "voice": "ryan"},
            },
            "旁白": {
                "locked": False,
                "voice_cfg": {"mode": "preset", "voice": "eric"},
            },
        }
        result = inject_cast_state_into_global_cast(global_cast, cast_state)
        assert result["老渔夫"]["voice_cfg"]["voice"] == "ryan"
        assert "voice_cfg" not in result["旁白"]

    def test_empty_cast_state_no_change(self):
        """Should return global_cast unchanged when cast_state is empty."""
        global_cast = {"旁白": {"gender": "male"}}
        result = inject_cast_state_into_global_cast(global_cast, {})
        assert result == global_cast

    def test_none_cast_state_no_change(self):
        """Should return global_cast unchanged when cast_state is None."""
        global_cast = {"旁白": {"gender": "male"}}
        result = inject_cast_state_into_global_cast(global_cast, None)
        assert result == global_cast

    def test_character_not_in_global_cast_skipped(self):
        """Should skip characters in cast_state but not in global_cast."""
        global_cast = {"旁白": {"gender": "male"}}
        cast_state = {
            "不存在": {
                "locked": True,
                "voice_cfg": {"mode": "preset", "voice": "ryan"},
            }
        }
        result = inject_cast_state_into_global_cast(global_cast, cast_state)
        assert "不存在" not in result


# ---------------------------------------------------------------------------
# Tests: WebUI source code structure for audition console
# ---------------------------------------------------------------------------

class TestAuditionConsoleSourceStructure:
    @pytest.fixture
    def webui_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_cast_state_defined(self, webui_source):
        """cast_state gr.State should be defined."""
        assert "cast_state = gr.State" in webui_source

    def test_audition_panel_defined(self, webui_source):
        """Audition panel Accordion should be defined."""
        assert "audition_panel" in webui_source
        assert "角色试音与定妆室" in webui_source

    def test_gr_render_used(self, webui_source):
        """@gr.render should be used for dynamic cards."""
        assert "@gr.render" in webui_source
        assert "render_character_cards" in webui_source

    def test_parse_json_to_cast_state_defined(self, webui_source):
        """parse_json_to_cast_state function should be defined."""
        assert "def parse_json_to_cast_state(" in webui_source

    def test_build_voice_cfg_from_ui_defined(self, webui_source):
        """build_voice_cfg_from_ui function should be defined."""
        assert "def build_voice_cfg_from_ui(" in webui_source

    def test_test_single_voice_defined(self, webui_source):
        """test_single_voice function should be defined."""
        assert "def test_single_voice(" in webui_source

    def test_update_cast_voice_cfg_defined(self, webui_source):
        """update_cast_voice_cfg function should be defined."""
        assert "def update_cast_voice_cfg(" in webui_source

    def test_inject_cast_state_defined(self, webui_source):
        """inject_cast_state_into_global_cast should be defined."""
        assert "def inject_cast_state_into_global_cast(" in webui_source

    def test_three_mode_radio(self, webui_source):
        """Three voice modes should be in the Radio choices."""
        assert "预设基底" in webui_source
        assert "声音克隆" in webui_source
        assert "文本设计" in webui_source

    def test_lock_button_exists(self, webui_source):
        """Lock button should exist in the UI."""
        assert "确认使用此音色" in webui_source

    def test_cast_state_in_full_production(self, webui_source):
        """cast_state should be passed to run_cinecast for full production."""
        assert "cast_state=args[-1]" in webui_source

    def test_run_cinecast_accepts_cast_state(self, webui_source):
        """run_cinecast should accept cast_state parameter."""
        assert "cast_state=None" in webui_source

    def test_inject_called_in_run_cinecast(self, webui_source):
        """inject_cast_state_into_global_cast should be called in run_cinecast."""
        assert "inject_cast_state_into_global_cast(global_cast, cast_state)" in webui_source

    def test_audition_panel_visibility_synced_with_mode(self, webui_source):
        """Audition panel should toggle with mode change."""
        assert "audition_panel" in webui_source
        assert "[brain_panel, audition_panel]" in webui_source

    def test_uuid_import(self, webui_source):
        """webui.py should import uuid for test audio filename generation."""
        assert "import uuid" in webui_source

    def test_unlock_cast_voice_cfg_defined(self, webui_source):
        """unlock_cast_voice_cfg function should be defined."""
        assert "def unlock_cast_voice_cfg(" in webui_source

    def test_unlock_button_text_exists(self, webui_source):
        """Unlock button text should exist in the UI for locked characters."""
        assert "解锁修改" in webui_source

    def test_toggle_lock_logic_exists(self, webui_source):
        """Toggle lock/unlock logic should exist in the UI."""
        assert "_toggle_lock" in webui_source

    def test_voice_cfg_restored_in_render(self, webui_source):
        """Render should restore voice_cfg values for display."""
        assert "saved_mode" in webui_source
        assert "mode_default" in webui_source
        assert "preset_default" in webui_source
