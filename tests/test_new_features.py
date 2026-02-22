#!/usr/bin/env python3
"""
Tests for new features:
1. LLM API config persistence (load/save)
2. Preview text extraction (first 10 sentences)
3. Character list persistence & eric/serena deprecation
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers: standalone copies of functions (avoid heavy Gradio import chain)
# ---------------------------------------------------------------------------

LLM_CONFIG_FILE_KEY = "LLM_CONFIG_FILE"

DEPRECATED_VOICES = {"eric", "serena"}
DEFAULT_VOICE_ORDER = ["aiden", "dylan", "ryan", "uncle_fu", "ono_anna", "sohee", "vivian"]


def _load_llm_config(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"model_name": "qwen-plus", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": ""}


def _save_llm_config(filepath, model_name, base_url, api_key):
    config = {"model_name": model_name, "base_url": base_url, "api_key": api_key}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


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


def _parse_json_to_cast_state(json_str, role_voices):
    try:
        data = json.loads(json_str)
        characters = data.get("characters", {})
    except Exception:
        return {}

    cast_state = {}
    voice_idx = 0

    for char_name, char_info in characters.items():
        if not isinstance(char_info, dict):
            continue
        role = char_info.get("role", "extra")
        if role in role_voices:
            default_voice = role_voices[role]
        else:
            assigned_voice = DEFAULT_VOICE_ORDER[voice_idx % len(DEFAULT_VOICE_ORDER)]
            voice_idx += 1
            default_voice = {"mode": "preset", "voice": assigned_voice}

        cast_state[char_name] = {
            "role": role,
            "gender": char_info.get("gender", "unknown"),
            "emotion": char_info.get("emotion", "平静"),
            "locked": False,
            "voice_cfg": default_voice,
        }
    return cast_state


# ---------------------------------------------------------------------------
# Tests: Requirement 1 – LLM API config persistence
# ---------------------------------------------------------------------------

class TestLLMConfigPersistence:
    def test_default_config_when_no_file(self, tmp_path):
        filepath = str(tmp_path / "nonexistent.json")
        config = _load_llm_config(filepath)
        assert config["model_name"] == "qwen-plus"
        assert "dashscope" in config["base_url"]
        assert config["api_key"] == ""

    def test_save_and_load_roundtrip(self, tmp_path):
        filepath = str(tmp_path / "llm_config.json")
        _save_llm_config(filepath, "gpt-4", "https://api.openai.com/v1", "sk-test123")
        config = _load_llm_config(filepath)
        assert config["model_name"] == "gpt-4"
        assert config["base_url"] == "https://api.openai.com/v1"
        assert config["api_key"] == "sk-test123"

    def test_corrupt_file_returns_default(self, tmp_path):
        filepath = str(tmp_path / "bad.json")
        with open(filepath, 'w') as f:
            f.write("{{corrupted")
        config = _load_llm_config(filepath)
        assert config["model_name"] == "qwen-plus"

    def test_overwrite_existing_config(self, tmp_path):
        filepath = str(tmp_path / "llm_config.json")
        _save_llm_config(filepath, "model-a", "https://a.com/v1", "key-a")
        _save_llm_config(filepath, "model-b", "https://b.com/v1", "key-b")
        config = _load_llm_config(filepath)
        assert config["model_name"] == "model-b"
        assert config["api_key"] == "key-b"


class TestLLMConfigInWebUI:
    @pytest.fixture
    def webui_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_llm_config_file_defined(self, webui_source):
        assert "LLM_CONFIG_FILE" in webui_source

    def test_load_llm_config_defined(self, webui_source):
        assert "def load_llm_config(" in webui_source

    def test_save_llm_config_defined(self, webui_source):
        assert "def save_llm_config(" in webui_source

    def test_save_called_on_success(self, webui_source):
        assert "save_llm_config(model_name, base_url, api_key)" in webui_source

    def test_saved_llm_loaded_at_startup(self, webui_source):
        assert "saved_llm = load_llm_config()" in webui_source

    def test_llm_fields_use_saved_config(self, webui_source):
        assert 'saved_llm.get("model_name"' in webui_source
        assert 'saved_llm.get("base_url"' in webui_source
        assert 'saved_llm.get("api_key"' in webui_source


# ---------------------------------------------------------------------------
# Tests: Requirement 2 – Preview text extraction
# ---------------------------------------------------------------------------

class TestExtractPreviewSentences:
    @pytest.fixture
    def webui_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_extract_function_defined(self, webui_source):
        assert "def extract_preview_sentences(" in webui_source

    def test_preview_text_in_ui(self, webui_source):
        assert "preview_text" in webui_source

    def test_extract_button_exists(self, webui_source):
        assert "提取首章前10句" in webui_source

    def test_preview_text_editable(self, webui_source):
        assert "试听文本 (可自由编辑)" in webui_source

    def test_run_cinecast_accepts_preview_text(self, webui_source):
        assert "preview_text=None" in webui_source

    def test_preview_text_passed_to_preview(self, webui_source):
        assert "preview_text=args[-1]" in webui_source

    def test_plain_text_extraction(self, tmp_path):
        """Test extracting sentences from a plain text file."""
        txt_file = tmp_path / "novel.txt"
        txt_file.write_text(
            "第一句话。第二句话。第三句话！第四句话？"
            "第五句话。第六句话。第七句话。第八句话。"
            "第九句话。第十句话。第十一句话不该出现。",
            encoding="utf-8",
        )
        import re as _re
        with open(str(txt_file), "r", encoding="utf-8") as f:
            text = f.read()
        sentences = _re.split(r'(?<=[。！？!?\n])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        selected = sentences[:10]
        assert len(selected) == 10
        assert "第十一句话" not in "\n".join(selected)


class TestPreviewModeAcceptsText:
    def test_run_preview_mode_signature(self):
        """run_preview_mode should accept preview_text parameter."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "def run_preview_mode(self, input_source: str, preview_text: str = None)" in source

    def test_preview_text_builds_script(self):
        """When preview_text is provided, it should build narration script."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert '"type": "narration"' in source
        assert '"speaker": "narrator"' in source
        assert "使用用户编辑的试听文本" in source


# ---------------------------------------------------------------------------
# Tests: Requirement 3 – eric/serena deprecation & character persistence
# ---------------------------------------------------------------------------

class TestDeprecatedVoices:
    @pytest.fixture
    def webui_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_deprecated_voices_defined(self, webui_source):
        assert "DEPRECATED_VOICES" in webui_source
        assert '"eric"' in webui_source
        assert '"serena"' in webui_source

    def test_default_voice_order_defined(self, webui_source):
        assert "DEFAULT_VOICE_ORDER" in webui_source

    def test_default_narrator_is_not_eric(self, webui_source):
        # The default load_role_voices should return aiden, not eric
        assert '"voice": "aiden"' in webui_source

    def test_default_preset_dropdown_not_eric(self, webui_source):
        # The character card preset dropdown default should not be Eric
        assert 'value="Aiden"' in webui_source


class TestEricSerenaNotPersisted:
    def test_eric_voice_not_saved(self, tmp_path):
        filepath = str(tmp_path / "voices.json")
        _save_role_voice(filepath, "m1", {"mode": "preset", "voice": "eric"})
        assert not os.path.exists(filepath)

    def test_serena_voice_not_saved(self, tmp_path):
        filepath = str(tmp_path / "voices.json")
        _save_role_voice(filepath, "f1", {"mode": "preset", "voice": "serena"})
        assert not os.path.exists(filepath)

    def test_aiden_voice_saved(self, tmp_path):
        filepath = str(tmp_path / "voices.json")
        _save_role_voice(filepath, "m1", {"mode": "preset", "voice": "aiden"})
        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert data["m1"]["voice"] == "aiden"

    def test_clone_mode_eric_ref_still_saved(self, tmp_path):
        """Clone mode with an eric reference name doesn't have 'voice' key,
        so it should still be saved normally."""
        filepath = str(tmp_path / "voices.json")
        _save_role_voice(filepath, "m1", {"mode": "clone", "ref_audio": "eric.wav"})
        assert os.path.exists(filepath)

    def test_design_mode_still_saved(self, tmp_path):
        filepath = str(tmp_path / "voices.json")
        _save_role_voice(filepath, "f1", {"mode": "design", "instruct": "Soft voice"})
        assert os.path.exists(filepath)


class TestCastStateDefaultVoices:
    def test_extra_roles_use_default_order(self):
        """Characters with role 'extra' should cycle through DEFAULT_VOICE_ORDER."""
        master = json.dumps({
            "characters": {
                "路人A": {"role": "extra", "gender": "male", "emotion": "平静"},
                "路人B": {"role": "extra", "gender": "female", "emotion": "活泼"},
                "路人C": {"role": "extra", "gender": "male", "emotion": "沧桑"},
            },
            "recaps": {}
        })
        role_voices = {"narrator": {"mode": "preset", "voice": "aiden"}}
        state = _parse_json_to_cast_state(master, role_voices)
        voices = [state[k]["voice_cfg"]["voice"] for k in state]
        assert voices[0] == "aiden"  # first in DEFAULT_VOICE_ORDER
        assert voices[1] == "dylan"  # second
        assert voices[2] == "ryan"   # third

    def test_known_roles_use_saved_config(self):
        """Characters with known roles (m1, f1) should use saved config."""
        master = json.dumps({
            "characters": {
                "主角": {"role": "m1", "gender": "male", "emotion": "热血"},
                "女主": {"role": "f1", "gender": "female", "emotion": "温柔"},
            },
            "recaps": {}
        })
        role_voices = {
            "narrator": {"mode": "preset", "voice": "aiden"},
            "m1": {"mode": "preset", "voice": "dylan"},
            "f1": {"mode": "preset", "voice": "vivian"},
        }
        state = _parse_json_to_cast_state(master, role_voices)
        assert state["主角"]["voice_cfg"]["voice"] == "dylan"
        assert state["女主"]["voice_cfg"]["voice"] == "vivian"

    def test_no_eric_in_defaults(self):
        """Default voice assignment should never use eric or serena."""
        master = json.dumps({
            "characters": {
                f"角色{i}": {"role": "extra", "gender": "male", "emotion": "平静"}
                for i in range(20)
            },
            "recaps": {}
        })
        role_voices = {"narrator": {"mode": "preset", "voice": "aiden"}}
        state = _parse_json_to_cast_state(master, role_voices)
        for char_info in state.values():
            voice = char_info["voice_cfg"].get("voice", "")
            assert voice not in DEPRECATED_VOICES


class TestMainProducerDefaultVoice:
    def test_default_narrator_voice_not_eric(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert '"default_narrator_voice": "aiden"' in source
