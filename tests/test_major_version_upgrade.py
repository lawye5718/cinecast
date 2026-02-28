#!/usr/bin/env python3
"""
Tests for the major version upgrade modules:
- RhythmManager (rhythm_manager.py)
- RoleManager (role_manager.py)
- AudiobookOrchestrator (audiobook_orchestrator.py)
- CinecastMLXEngine (mlx_tts_engine.py)
- production_pipeline (main_producer.py)
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.rhythm_manager import RhythmManager
from modules.role_manager import RoleManager
from modules.audiobook_orchestrator import (
    AudiobookOrchestrator,
    parse_script_line,
    parse_script,
    LANGUAGE_MAP,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MLX_ENGINE_PATH = os.path.join(_PROJECT_ROOT, "modules", "mlx_tts_engine.py")
_MAIN_PRODUCER_PATH = os.path.join(_PROJECT_ROOT, "main_producer.py")


# ===========================================================================
# RhythmManager Tests
# ===========================================================================

class TestRhythmManagerInit:
    """Test RhythmManager initialization and configuration."""

    def test_default_pauses(self):
        rm = RhythmManager()
        assert rm.pauses["comma"] == 0.2
        assert rm.pauses["period"] == 0.5
        assert rm.pauses["question"] == 0.6
        assert rm.pauses["newline"] == 0.8

    def test_custom_config_overrides(self):
        rm = RhythmManager(config={"comma": 0.5, "period": 1.0})
        assert rm.pauses["comma"] == 0.5
        assert rm.pauses["period"] == 1.0
        # Non-overridden keys retain defaults
        assert rm.pauses["question"] == 0.6

    def test_update_config(self):
        rm = RhythmManager()
        rm.update_config({"comma": 0.3})
        assert rm.pauses["comma"] == 0.3

    def test_get_pause_duration(self):
        rm = RhythmManager()
        assert rm.get_pause_duration("comma") == 0.2
        assert rm.get_pause_duration("unknown") == 0.0


class TestRhythmManagerProcessText:
    """Test text segmentation with pause metadata."""

    def test_empty_text(self):
        rm = RhythmManager()
        assert rm.process_text_with_metadata("") == []
        assert rm.process_text_with_metadata("  ") == []

    def test_single_sentence_chinese(self):
        rm = RhythmManager()
        segments = rm.process_text_with_metadata("你好，世界。")
        assert len(segments) >= 1
        # Should have at least some segments with pauses
        texts = [s["text"] for s in segments]
        assert any("你好" in t for t in texts)

    def test_period_pause(self):
        rm = RhythmManager()
        segments = rm.process_text_with_metadata("第一句。第二句。")
        # At least one segment should have a period pause
        period_pauses = [s for s in segments if s["pause"] == rm.pauses["period"]]
        assert len(period_pauses) >= 1

    def test_question_pause(self):
        rm = RhythmManager()
        segments = rm.process_text_with_metadata("你好吗？")
        pauses = [s["pause"] for s in segments]
        assert rm.pauses["question"] in pauses

    def test_returns_list_of_dicts(self):
        rm = RhythmManager()
        segments = rm.process_text_with_metadata("简单测试。")
        for seg in segments:
            assert "text" in seg
            assert "pause" in seg
            assert isinstance(seg["pause"], float)


class TestRhythmManagerInjectPauses:
    """Test pause marker injection into text."""

    def test_inject_chinese_comma(self):
        rm = RhythmManager()
        result = rm.inject_pauses("你好，世界")
        assert "[pause=" in result

    def test_inject_period(self):
        rm = RhythmManager()
        result = rm.inject_pauses("结束。")
        assert f"[pause={rm.pauses['period']}]" in result

    def test_empty_text_unchanged(self):
        rm = RhythmManager()
        assert rm.inject_pauses("") == ""
        assert rm.inject_pauses(None) is None


class TestRhythmManagerSilence:
    """Test silence frame creation."""

    def test_create_silence_frames(self):
        rm = RhythmManager()
        silence = rm.create_silence_frames(0.5, 24000)
        assert len(silence) == 12000
        assert silence.dtype == np.float32
        assert np.all(silence == 0)

    def test_create_silence_custom_sr(self):
        rm = RhythmManager()
        silence = rm.create_silence_frames(1.0, 22050)
        assert len(silence) == 22050

    def test_zero_duration(self):
        rm = RhythmManager()
        silence = rm.create_silence_frames(0.0)
        assert len(silence) == 0


# ===========================================================================
# RoleManager Tests
# ===========================================================================

class TestRoleManagerSaveLoad:
    """Test voice feature save/load with NPZ format."""

    def test_save_and_load_feature(self, tmp_path):
        roles_dir = str(tmp_path / "roles")
        feature = {"spk_emb": np.random.randn(256).astype(np.float32)}
        RoleManager.save_voice_feature(feature, "hero", roles_dir)

        loaded = RoleManager.load_voice_feature("hero", roles_dir)
        assert loaded is not None
        assert "spk_emb" in loaded
        np.testing.assert_array_almost_equal(loaded["spk_emb"], feature["spk_emb"])

    def test_save_with_metadata(self, tmp_path):
        roles_dir = str(tmp_path / "roles")
        feature = {"ref_code": np.zeros(128, dtype=np.float32)}
        metadata = {"description": "深沉男声", "language": "zh"}
        RoleManager.save_voice_feature(feature, "narrator", roles_dir, metadata)

        meta = RoleManager.load_voice_metadata("narrator", roles_dir)
        assert meta is not None
        assert meta["description"] == "深沉男声"

    def test_load_nonexistent_returns_none(self, tmp_path):
        roles_dir = str(tmp_path / "empty")
        os.makedirs(roles_dir)
        assert RoleManager.load_voice_feature("ghost", roles_dir) is None

    def test_load_metadata_nonexistent_returns_none(self, tmp_path):
        assert RoleManager.load_voice_metadata("ghost", str(tmp_path)) is None


class TestRoleManagerBank:
    """Test role bank loading."""

    def test_load_role_bank(self, tmp_path):
        roles_dir = str(tmp_path / "bank")
        for name in ["hero", "villain"]:
            feature = {"emb": np.ones(64, dtype=np.float32) * (1 if name == "hero" else -1)}
            RoleManager.save_voice_feature(feature, name, roles_dir)

        rm = RoleManager(roles_dir)
        bank = rm.load_role_bank(["hero", "villain"])
        assert len(bank) == 2
        assert "hero" in bank
        assert "villain" in bank

    def test_load_role_bank_auto_scan(self, tmp_path):
        roles_dir = str(tmp_path / "auto")
        RoleManager.save_voice_feature(
            {"v": np.zeros(32, dtype=np.float32)}, "char1", roles_dir
        )
        RoleManager.save_voice_feature(
            {"v": np.ones(32, dtype=np.float32)}, "char2", roles_dir
        )

        rm = RoleManager(roles_dir)
        bank = rm.load_role_bank()  # Auto-scan
        assert len(bank) == 2

    def test_load_empty_bank(self, tmp_path):
        roles_dir = str(tmp_path / "empty_bank")
        os.makedirs(roles_dir)
        rm = RoleManager(roles_dir)
        bank = rm.load_role_bank()
        assert bank == {}

    def test_load_bank_nonexistent_dir(self, tmp_path):
        rm = RoleManager(str(tmp_path / "nonexistent"))
        bank = rm.load_role_bank()
        assert bank == {}


class TestRoleManagerList:
    """Test listing and deleting roles."""

    def test_list_roles(self, tmp_path):
        roles_dir = str(tmp_path / "list")
        RoleManager.save_voice_feature(
            {"emb": np.zeros(32, dtype=np.float32)}, "hero", roles_dir,
            metadata={"description": "英雄"}
        )
        rm = RoleManager(roles_dir)
        roles = rm.list_roles()
        assert len(roles) == 1
        assert roles[0]["name"] == "hero"
        assert roles[0]["has_metadata"] is True

    def test_delete_role(self, tmp_path):
        roles_dir = str(tmp_path / "del")
        RoleManager.save_voice_feature(
            {"emb": np.zeros(32, dtype=np.float32)}, "temp", roles_dir
        )
        rm = RoleManager(roles_dir)
        assert rm.delete_role("temp") is True
        assert RoleManager.load_voice_feature("temp", roles_dir) is None

    def test_delete_nonexistent_role(self, tmp_path):
        rm = RoleManager(str(tmp_path))
        assert rm.delete_role("ghost") is False


class TestRoleManagerVoiceCard:
    """Test Voice Card export/import."""

    def test_export_and_import(self, tmp_path):
        roles_dir = str(tmp_path / "src")
        export_dir = str(tmp_path / "export")
        dest_dir = str(tmp_path / "dest")

        feature = {"emb": np.random.randn(64).astype(np.float32)}
        RoleManager.save_voice_feature(feature, "warrior", roles_dir,
                                       metadata={"lang": "zh"})

        # Export
        src_rm = RoleManager(roles_dir)
        card_path = src_rm.export_voice_card("warrior", export_dir)
        assert card_path is not None
        assert os.path.exists(card_path)

        # Import
        dest_rm = RoleManager(dest_dir)
        name = dest_rm.import_voice_card(card_path)
        assert name == "warrior"
        loaded = RoleManager.load_voice_feature("warrior", dest_dir)
        assert loaded is not None

    def test_export_nonexistent(self, tmp_path):
        rm = RoleManager(str(tmp_path))
        assert rm.export_voice_card("ghost", str(tmp_path / "out")) is None

    def test_import_invalid_path(self, tmp_path):
        rm = RoleManager(str(tmp_path))
        assert rm.import_voice_card("/nonexistent/file.npz") is None
        assert rm.import_voice_card(str(tmp_path / "not_npz.txt")) is None


# ===========================================================================
# AudiobookOrchestrator Tests
# ===========================================================================

class TestScriptParser:
    """Test script line parsing."""

    def test_parse_chinese_colon(self):
        role, text = parse_script_line("老渔夫：你相信命运吗？")
        assert role == "老渔夫"
        assert text == "你相信命运吗？"

    def test_parse_english_colon(self):
        role, text = parse_script_line("Hero: I believe in fate.")
        assert role == "Hero"
        assert text == "I believe in fate."

    def test_parse_narration(self):
        role, text = parse_script_line("夜幕降临港口。")
        assert role is None
        assert text == "夜幕降临港口。"

    def test_parse_empty_line(self):
        role, text = parse_script_line("")
        assert role is None
        assert text == ""

    def test_parse_multiline_script(self):
        script_text = "老渔夫：你相信命运吗？\n年轻人：我不信。\n夜幕降临了。"
        result = parse_script(script_text)
        assert len(result) == 3
        assert result[0] == ("老渔夫", "你相信命运吗？")
        assert result[1] == ("年轻人", "我不信。")
        assert result[2] == (None, "夜幕降临了。")

    def test_long_role_name_ignored(self):
        """Role names longer than 20 chars are treated as narration."""
        role, text = parse_script_line("这是一个非常非常非常非常非常非常长的角色名字: 内容")
        assert role is None


class TestLanguageMap:
    """Test language mapping."""

    def test_chinese_variants(self):
        assert LANGUAGE_MAP["Chinese"] == "zh"
        assert LANGUAGE_MAP["中文"] == "zh"
        assert LANGUAGE_MAP["zh"] == "zh"

    def test_english_variants(self):
        assert LANGUAGE_MAP["English"] == "en"
        assert LANGUAGE_MAP["en"] == "en"

    def test_japanese(self):
        assert LANGUAGE_MAP["Japanese"] == "jp"


class TestAudiobookOrchestrator:
    """Test orchestrator initialization and basic functionality."""

    def test_init_defaults(self):
        orch = AudiobookOrchestrator()
        assert orch.engine is None
        assert isinstance(orch.rhythm, RhythmManager)
        assert isinstance(orch.rm, RoleManager)
        assert orch.sample_rate == 24000

    def test_init_custom_rhythm(self):
        orch = AudiobookOrchestrator(rhythm_config={"comma": 0.5})
        assert orch.rhythm.pauses["comma"] == 0.5

    def test_process_chapter_no_engine(self):
        """Without engine, should return empty audio (no crash)."""
        orch = AudiobookOrchestrator()
        script = [("narrator", "测试文本。")]
        audio = orch.process_chapter(script)
        assert isinstance(audio, np.ndarray)

    def test_process_chapter_from_text(self):
        orch = AudiobookOrchestrator()
        audio = orch.process_chapter_from_text("旁白：测试。\n角色：你好。")
        assert isinstance(audio, np.ndarray)

    def test_clear_memory_no_crash(self):
        orch = AudiobookOrchestrator()
        orch.clear_memory()  # Should not raise


# ===========================================================================
# CinecastMLXEngine Source Code Tests
# ===========================================================================

class TestCinecastMLXEngineSource:
    """Verify CinecastMLXEngine class exists and has required methods."""

    def _read_source(self):
        with open(_MLX_ENGINE_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_class_exists(self):
        source = self._read_source()
        assert "class CinecastMLXEngine" in source

    def test_generate_method(self):
        source = self._read_source()
        assert "def generate(self, text" in source

    def test_generate_voice_design_method(self):
        source = self._read_source()
        assert "def generate_voice_design(self, text" in source

    def test_generate_voice_clone_method(self):
        source = self._read_source()
        assert "def generate_voice_clone(self, text" in source

    def test_unload_model_method(self):
        source = self._read_source()
        assert "def unload_model(self" in source

    def test_design_mode_in_generate(self):
        source = self._read_source()
        assert '"design"' in source

    def test_clone_mode_in_generate(self):
        source = self._read_source()
        assert '"clone"' in source

    def test_metal_clear_cache(self):
        source = self._read_source()
        assert "mx.metal.clear_cache()" in source or "mx.clear_cache()" in source

    def test_gc_collect_in_unload(self):
        source = self._read_source()
        # Verify gc.collect is used in unload
        assert "gc.collect()" in source

    def test_sample_rate_default_24000(self):
        source = self._read_source()
        assert "self.sample_rate = 24000" in source


# ===========================================================================
# main_producer production_pipeline Tests
# ===========================================================================

class TestProductionPipelineSource:
    """Verify production_pipeline function exists in main_producer.py."""

    def _read_source(self):
        with open(_MAIN_PRODUCER_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_function_exists(self):
        source = self._read_source()
        assert "def production_pipeline(" in source

    def test_imports_role_manager(self):
        source = self._read_source()
        assert "RoleManager" in source

    def test_imports_orchestrator(self):
        source = self._read_source()
        assert "AudiobookOrchestrator" in source

    def test_imports_parse_script(self):
        source = self._read_source()
        assert "parse_script" in source

    def test_loads_role_bank(self):
        source = self._read_source()
        assert "load_role_bank" in source

    def test_clear_memory(self):
        source = self._read_source()
        assert "clear_memory" in source


class TestProductionPipelineImport:
    """Verify production_pipeline function exists in source."""

    def _read_source(self):
        with open(_MAIN_PRODUCER_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_function_callable_in_source(self):
        source = self._read_source()
        assert "def production_pipeline(" in source


# ===========================================================================
# Module Import Tests
# ===========================================================================

class TestModuleImports:
    """Verify all new modules can be imported without errors."""

    def test_import_rhythm_manager(self):
        from modules.rhythm_manager import RhythmManager
        assert RhythmManager is not None

    def test_import_role_manager(self):
        from modules.role_manager import RoleManager
        assert RoleManager is not None

    def test_import_audiobook_orchestrator(self):
        from modules.audiobook_orchestrator import AudiobookOrchestrator
        assert AudiobookOrchestrator is not None

    def test_import_cinecast_mlx_engine(self):
        """CinecastMLXEngine exists in source (mlx only available on Apple Silicon)."""
        with open(_MLX_ENGINE_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        assert "class CinecastMLXEngine" in source

    def test_import_parse_script(self):
        from modules.audiobook_orchestrator import parse_script, parse_script_line
        assert callable(parse_script)
        assert callable(parse_script_line)
