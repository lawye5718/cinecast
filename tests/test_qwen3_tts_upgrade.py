#!/usr/bin/env python3
"""
Tests for Qwen3-TTS 1.7B upgrade.

Covers:
- MLXRenderEngine Model Pool (mode switching, config, warmup, fallback, destroy)
- AssetManager VoiceProfile (clone/design/preset modes, Clones/ scanning)
- CinematicPackager sample_rate and crossfade_ms upgrade
- main_producer config paths
- audio_assets_config.json updates
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.asset_manager import AssetManager
from modules.cinematic_packager import CinematicPackager

# ---------------------------------------------------------------------------
# Source code path helpers
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MLX_ENGINE_PATH = os.path.join(_PROJECT_ROOT, "modules", "mlx_tts_engine.py")
_ASSET_MANAGER_PATH = os.path.join(_PROJECT_ROOT, "modules", "asset_manager.py")
_MAIN_PRODUCER_PATH = os.path.join(_PROJECT_ROOT, "main_producer.py")
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "audio_assets_config.json")


# ---------------------------------------------------------------------------
# MLXRenderEngine source code guards
# ---------------------------------------------------------------------------

class TestMLXEngineModelPool:
    """Verify MLXRenderEngine supports Model Pool architecture."""

    def _read_source(self):
        with open(_MLX_ENGINE_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_init_accepts_config_param(self):
        source = self._read_source()
        assert "config=None" in source or "config=" in source

    def test_model_paths_dict_in_init(self):
        source = self._read_source()
        assert "_model_paths" in source

    def test_load_mode_method_exists(self):
        source = self._read_source()
        assert "def _load_mode(self, mode)" in source

    def test_warmup_method_exists(self):
        source = self._read_source()
        assert "def warmup(self" in source

    def test_fallback_path_in_init(self):
        source = self._read_source()
        assert "_fallback_path" in source

    def test_sample_rate_24000(self):
        source = self._read_source()
        assert "self.sample_rate = 24000" in source

    def test_fallback_sample_rate_22050(self):
        source = self._read_source()
        assert "self.sample_rate = 22050" in source

    def test_render_supports_clone_mode(self):
        source = self._read_source()
        assert '"clone"' in source
        assert "ref_audio" in source

    def test_render_supports_design_mode(self):
        source = self._read_source()
        assert '"design"' in source
        assert "instruct" in source

    def test_render_supports_preset_mode(self):
        source = self._read_source()
        assert '"preset"' in source

    def test_destroy_clears_model(self):
        source = self._read_source()
        assert "self.model = None" in source
        assert "self.current_mode = None" in source

    def test_do_load_method_exists(self):
        source = self._read_source()
        assert "def _do_load(self, path" in source

    def test_do_load_gc_cleanup(self):
        """_do_load should set self.model = None and call gc.collect() before mx.clear_cache()."""
        source = self._read_source()
        assert "self.model = None" in source
        assert "gc.collect()" in source


# ---------------------------------------------------------------------------
# AssetManager VoiceProfile tests
# ---------------------------------------------------------------------------

class TestAssetManagerVoiceProfile:
    """Verify AssetManager supports VoiceProfile system."""

    @pytest.fixture
    def manager(self, tmp_path):
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        for name in ("narrator.wav", "m1.wav", "m2.wav", "f1.wav", "f2.wav"):
            (voices_dir / name).write_bytes(b"RIFF" + b"\x00" * 40)
        return AssetManager(asset_dir=str(tmp_path))

    def test_target_sr_24000(self, manager):
        assert manager.target_sr == 24000

    def test_build_voice_profile_clone(self, manager, tmp_path):
        ref_audio = tmp_path / "voices" / "hero.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 40)
        profile = manager.build_voice_profile(
            "Hero", ref_audio=str(ref_audio), ref_text="test ref"
        )
        assert profile["mode"] == "clone"
        assert profile["ref_audio"] == str(ref_audio)
        assert profile["ref_text"] == "test ref"

    def test_build_voice_profile_design(self, manager):
        profile = manager.build_voice_profile(
            "OldMan", description="Deep male voice, like a pirate"
        )
        assert profile["mode"] == "design"
        assert "pirate" in profile["instruct"]

    def test_build_voice_profile_preset(self, manager):
        profile = manager.build_voice_profile(
            "NPC", speaker_id="Male_01"
        )
        assert profile["mode"] == "preset"
        assert profile["speaker"] == "Male_01"

    def test_build_voice_profile_fallback(self, manager):
        """When no mode params provided, falls back to default voice."""
        profile = manager.build_voice_profile("Unknown")
        assert "audio" in profile or "mode" in profile

    def test_clone_profile_remembered(self, manager, tmp_path):
        ref_audio = tmp_path / "voices" / "hero.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 40)
        manager.build_voice_profile("Hero", ref_audio=str(ref_audio))
        voice = manager.get_voice_for_role("dialogue", "Hero")
        assert voice["mode"] == "clone"

    def test_design_profile_remembered(self, manager):
        manager.build_voice_profile("Elder", description="Old gravelly voice")
        voice = manager.get_voice_for_role("dialogue", "Elder")
        assert voice["mode"] == "design"

    def test_build_profile_missing_ref_audio(self, manager):
        """If ref_audio path doesn't exist, should not use clone mode."""
        profile = manager.build_voice_profile(
            "Ghost", ref_audio="/nonexistent/path.wav"
        )
        assert profile.get("mode") != "clone"


class TestClonesScan:
    """Verify AssetManager scans Assets/Clones/ directory."""

    def test_scan_clone_voices(self, tmp_path):
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        for name in ("narrator.wav", "m1.wav", "m2.wav", "f1.wav", "f2.wav"):
            (voices_dir / name).write_bytes(b"RIFF" + b"\x00" * 40)

        clones_dir = tmp_path / "Clones"
        clones_dir.mkdir()
        (clones_dir / "hero.wav").write_bytes(b"RIFF" + b"\x00" * 40)
        (clones_dir / "villain.mp3").write_bytes(b"\x00" * 40)

        manager = AssetManager(asset_dir=str(tmp_path))
        assert "hero" in manager.role_voice_map
        assert "villain" in manager.role_voice_map
        assert manager.role_voice_map["hero"]["mode"] == "clone"

    def test_no_clones_dir_no_error(self, tmp_path):
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        for name in ("narrator.wav", "m1.wav", "m2.wav", "f1.wav", "f2.wav"):
            (voices_dir / name).write_bytes(b"RIFF" + b"\x00" * 40)
        manager = AssetManager(asset_dir=str(tmp_path))
        # Should not raise, just no clone voices


class TestAssetManagerSourceGuards:
    """Verify AssetManager source includes new methods."""

    def _read_source(self):
        with open(_ASSET_MANAGER_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_scan_clone_voices_exists(self):
        source = self._read_source()
        assert "def _scan_clone_voices(self)" in source

    def test_build_voice_profile_exists(self):
        source = self._read_source()
        assert "def build_voice_profile(self" in source

    def test_target_sr_24000_in_source(self):
        source = self._read_source()
        assert "self.target_sr = 24000" in source


# ---------------------------------------------------------------------------
# CinematicPackager upgrade tests
# ---------------------------------------------------------------------------

class TestCinematicPackagerUpgrade:
    """Verify CinematicPackager sample_rate and crossfade_ms upgrades."""

    def test_sample_rate_24000(self):
        p = CinematicPackager("/tmp/test_output")
        assert p.sample_rate == 24000

    def test_crossfade_ms_in_range(self):
        p = CinematicPackager("/tmp/test_output")
        assert hasattr(p, "crossfade_ms")
        assert 15 <= p.crossfade_ms <= 20


# ---------------------------------------------------------------------------
# main_producer config tests
# ---------------------------------------------------------------------------

class TestMainProducerConfig:
    """Verify main_producer.py has been updated for 1.7B model paths."""

    def _read_source(self):
        with open(_MAIN_PRODUCER_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_model_path_base_in_config(self):
        source = self._read_source()
        assert "model_path_base" in source

    def test_model_path_design_in_config(self):
        source = self._read_source()
        assert "model_path_design" in source

    def test_model_path_custom_in_config(self):
        source = self._read_source()
        assert "model_path_custom" in source

    def test_model_path_fallback_in_config(self):
        source = self._read_source()
        assert "model_path_fallback" in source

    def test_engine_config_passthrough(self):
        source = self._read_source()
        assert "engine_config" in source

    def test_warmup_called_in_phase2(self):
        source = self._read_source()
        assert "engine.warmup" in source


# ---------------------------------------------------------------------------
# audio_assets_config.json tests
# ---------------------------------------------------------------------------

class TestAudioAssetsConfig:
    """Verify audio_assets_config.json has been updated."""

    def _load_config(self):
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_target_sample_rate_24000(self):
        config = self._load_config()
        assert config["audio_processing"]["target_sample_rate"] == 24000

    def test_crossfade_ms_in_config(self):
        config = self._load_config()
        assert "crossfade_ms" in config["audio_processing"]
        assert 15 <= config["audio_processing"]["crossfade_ms"] <= 20

    def test_model_config_section_exists(self):
        config = self._load_config()
        assert "model_config" in config

    def test_model_config_has_paths(self):
        config = self._load_config()
        mc = config["model_config"]
        assert "model_path_base" in mc
        assert "model_path_design" in mc
        assert "model_path_custom" in mc
        assert "model_path_fallback" in mc
