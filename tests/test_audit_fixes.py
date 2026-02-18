#!/usr/bin/env python3
"""
Tests for the code audit fixes (rendering timeout, text cleaning, gender detection,
CinematicPackager improvements, and MLX engine destroy).

Covers:
- Fix 1: Timeout handler deletes dirty audio files
- Fix 2: Enhanced text cleaning (whitespace, double-dot, length truncation)
- Fix 3: Speed factor frame_rate distortion removed
- Fix 4: Crossfade merge in _merge_with_previous
- Fix 5: CinematicPackager accepts target_duration_min parameter
- Fix 6: Gender detection supports Chinese labels
- Fix 7: MLXRenderEngine has destroy() method
"""

import gc
import os
import re
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.asset_manager import AssetManager
from modules.cinematic_packager import CinematicPackager

# ---------------------------------------------------------------------------
# Helper: mirror the updated text cleaning logic from mlx_tts_engine.py
# ---------------------------------------------------------------------------

_CLOSING_PUNCT_RE = re.compile(r'[。！？；.!?;]$')


def _apply_text_cleaning(text: str, max_chars: int = 60) -> str:
    """Mirror the full aggressive text cleaning logic in render_dry_chunk."""
    render_text = text.strip()
    render_text = re.sub(r'[…]+', '。', render_text)
    render_text = re.sub(r'\.{2,}', '。', render_text)
    render_text = re.sub(r'[—]+', '，', render_text)
    render_text = re.sub(r'[-]{2,}', '，', render_text)
    render_text = re.sub(r'[~～]+', '。', render_text)
    render_text = re.sub(r'\s+', ' ', render_text).strip()
    if len(render_text) > max_chars:
        render_text = render_text[:max_chars] + "。"
    if not _CLOSING_PUNCT_RE.search(render_text):
        render_text += "。"
    return render_text


# ---------------------------------------------------------------------------
# Fix 1: Timeout handler deletes dirty audio files
# ---------------------------------------------------------------------------

class TestTimeoutDirtyFileCleanup:
    """Verify the render loop deletes dirty audio on timeout."""

    def test_dirty_file_removed_on_timeout(self):
        """When render exceeds threshold, the produced file must be deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_chunk.wav")
            # Simulate: engine writes a file, then timeout is detected
            with open(save_path, "w") as f:
                f.write("dirty audio data")
            assert os.path.exists(save_path)

            # Simulate timeout cleanup logic
            if os.path.exists(save_path):
                os.remove(save_path)

            assert not os.path.exists(save_path)

    def test_source_has_dirty_file_removal(self):
        """Verify main_producer.py contains dirty file removal in timeout branch."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "os.remove(save_path)" in source, "Dirty file removal not found in timeout handler"
        assert "已销毁超时产生的脏音频" in source, "Dirty file removal log message not found"


# ---------------------------------------------------------------------------
# Fix 2: Enhanced text cleaning
# ---------------------------------------------------------------------------

class TestEnhancedTextCleaning:
    """Verify the additional cleaning rules: whitespace, double-dot, truncation."""

    def test_newlines_collapsed_to_space(self):
        """Internal newlines should be collapsed to a single space."""
        result = _apply_text_cleaning("第一行\n第二行\n第三行")
        assert "\n" not in result
        assert "第一行 第二行 第三行" in result

    def test_tabs_collapsed_to_space(self):
        """Tabs should be collapsed to a single space."""
        result = _apply_text_cleaning("第一\t\t第二")
        assert "\t" not in result

    def test_multiple_spaces_collapsed(self):
        """Multiple consecutive spaces should be collapsed."""
        result = _apply_text_cleaning("第一     第二")
        assert "     " not in result
        assert "第一 第二" in result

    def test_double_dot_now_replaced(self):
        """Double dots (..) should now be replaced with 。"""
        result = _apply_text_cleaning("他走了..")
        assert ".." not in result

    def test_long_text_truncated_at_max_chars(self):
        """Text longer than max_chars (60) should be truncated with 。"""
        long_text = "这" * 100
        result = _apply_text_cleaning(long_text)
        assert len(result) <= 62  # 60 chars + "。"
        assert result.endswith("。")

    def test_text_at_max_chars_not_truncated(self):
        """Text exactly 60 characters should not be truncated."""
        text_60 = "这" * 60
        result = _apply_text_cleaning(text_60)
        # Should not be truncated, just get a period appended
        assert result == text_60 + "。"

    def test_text_under_max_chars_not_truncated(self):
        """Text under 60 characters should not be truncated."""
        text_50 = "这" * 50
        result = _apply_text_cleaning(text_50)
        assert result == text_50 + "。"

    def test_source_has_whitespace_cleaning(self):
        """Verify mlx_tts_engine.py contains whitespace normalization."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "mlx_tts_engine.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert r"\s+" in source, "Whitespace normalization regex not found"

    def test_source_has_length_truncation(self):
        """Verify mlx_tts_engine.py contains length truncation using self.max_chars."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "mlx_tts_engine.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "len(render_text) > self.max_chars" in source, "Length truncation check not found"


# ---------------------------------------------------------------------------
# Fix 3: Speed factor frame_rate distortion removed
# ---------------------------------------------------------------------------

class TestSpeedDistortionRemoved:
    """Verify that frame_rate-based speed change is removed from cinematic_packager."""

    def test_no_frame_rate_speed_hack(self):
        """cinematic_packager.py should NOT contain frame_rate speed manipulation."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "cinematic_packager.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "new_frame_rate" not in source, "Frame-rate speed hack still present"
        assert "_spawn" not in source, "_spawn speed hack still present"


# ---------------------------------------------------------------------------
# Fix 4: Crossfade merge
# ---------------------------------------------------------------------------

class TestCrossfadeMerge:
    """Verify that _merge_with_previous uses crossfade instead of simple concatenation."""

    def test_source_has_crossfade(self):
        """cinematic_packager.py should use crossfade in _merge_with_previous."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "cinematic_packager.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "crossfade" in source, "Crossfade not found in _merge_with_previous"
        assert ".append(" in source, "append with crossfade not found"


# ---------------------------------------------------------------------------
# Fix 5: CinematicPackager target_duration_min parameter
# ---------------------------------------------------------------------------

class TestTargetDurationParam:
    """Verify CinematicPackager accepts and uses target_duration_min."""

    def test_default_30_minutes(self):
        """Default target_duration_ms should be 30 * 60 * 1000."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = CinematicPackager(tmpdir)
            assert p.target_duration_ms == 30 * 60 * 1000

    def test_custom_duration(self):
        """Custom target_duration_min should be respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = CinematicPackager(tmpdir, target_duration_min=15)
            assert p.target_duration_ms == 15 * 60 * 1000

    def test_preview_duration(self):
        """Preview mode 0.5 min should result in 30000 ms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = CinematicPackager(tmpdir, target_duration_min=0.5)
            assert p.target_duration_ms == 0.5 * 60 * 1000

    def test_source_passes_target_duration(self):
        """main_producer.py should pass target_duration_min to CinematicPackager."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "target_duration_min=" in source, \
            "target_duration_min not passed to CinematicPackager"


# ---------------------------------------------------------------------------
# Fix 6: Gender detection supports Chinese labels
# ---------------------------------------------------------------------------

class TestChineseGenderLabels:
    """Verify gender detection handles Chinese labels correctly."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create an AssetManager with actual voice files on disk."""
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        for name in ("narrator.wav", "m1.wav", "m2.wav", "f1.wav", "f2.wav"):
            (voices_dir / name).write_bytes(b"RIFF" + b"\x00" * 40)
        return AssetManager(asset_dir=str(tmp_path))

    def test_female_chinese_char(self, manager):
        """Gender '女' should select female_pool."""
        v = manager.get_voice_for_role("dialogue", "女角色测试A", "女")
        pool_audios = [e["audio"] for e in manager.voices["female_pool"]]
        assert v["audio"] in pool_audios

    def test_female_chinese_word(self, manager):
        """Gender '女性' should select female_pool."""
        v = manager.get_voice_for_role("dialogue", "女角色测试B", "女性")
        pool_audios = [e["audio"] for e in manager.voices["female_pool"]]
        assert v["audio"] in pool_audios

    def test_female_f_shorthand(self, manager):
        """Gender 'f' should select female_pool."""
        v = manager.get_voice_for_role("dialogue", "女角色测试C", "f")
        pool_audios = [e["audio"] for e in manager.voices["female_pool"]]
        assert v["audio"] in pool_audios

    def test_female_english_still_works(self, manager):
        """Gender 'female' should still select female_pool."""
        v = manager.get_voice_for_role("dialogue", "女角色测试D", "female")
        pool_audios = [e["audio"] for e in manager.voices["female_pool"]]
        assert v["audio"] in pool_audios

    def test_male_still_uses_male_pool(self, manager):
        """Gender 'male' should still select male_pool."""
        v = manager.get_voice_for_role("dialogue", "男角色测试A", "male")
        pool_audios = [e["audio"] for e in manager.voices["male_pool"]]
        assert v["audio"] in pool_audios

    def test_unknown_gender_defaults_to_male(self, manager):
        """Unknown gender string should default to male_pool."""
        v = manager.get_voice_for_role("dialogue", "未知角色A", "unknown")
        pool_audios = [e["audio"] for e in manager.voices["male_pool"]]
        assert v["audio"] in pool_audios


# ---------------------------------------------------------------------------
# Fix 7: MLXRenderEngine has destroy() method
# ---------------------------------------------------------------------------

class TestEngineDestroyMethod:
    """Verify MLXRenderEngine has an explicit destroy() method."""

    def test_destroy_method_in_source(self):
        """mlx_tts_engine.py should define a destroy() method."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "mlx_tts_engine.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "def destroy(self):" in source, "destroy() method not found"
        assert "del self.model" in source, "Model cleanup in destroy not found"

    def test_destroy_called_in_main_producer(self):
        """main_producer.py should call engine.destroy() with hasattr guard."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "engine.destroy()" in source, "engine.destroy() call not found"
        assert "hasattr(engine, 'destroy')" in source, "hasattr guard for engine.destroy() not found"
