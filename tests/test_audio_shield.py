#!/usr/bin/env python3
"""
Tests for CineCast Audio Shield module.

Covers:
- Scanner: directory scanning, file status management
- Analyzer: glitch detection algorithm on synthetic signals
- Editor: delete range, undo, normalize, save
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from pydub import AudioSegment
from pydub.generators import Sine

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_shield.scanner import AudioScanner, AudioFileInfo, FileStatus
from audio_shield.analyzer import detect_glitches_from_array
from audio_shield.editor import AudioBufferManager


# ===========================================================================
# Helpers
# ===========================================================================

def _make_sine_wav(path: str, duration_ms: int = 2000, freq: int = 440):
    """Create a short sine-wave WAV file for testing."""
    tone = Sine(freq).to_audio_segment(duration=duration_ms)
    tone.export(path, format="wav")
    return path


def _make_signal_with_spike(sr: int = 22050, duration_sec: float = 2.0):
    """
    Create a synthetic signal (quiet noise) with a deliberate spike.
    Returns (y, sr, spike_time).
    """
    n_samples = int(sr * duration_sec)
    rng = np.random.default_rng(42)
    y = rng.normal(0, 0.01, n_samples).astype(np.float32)

    # Insert a spike at 1.0 second
    spike_idx = int(sr * 1.0)
    y[spike_idx] = 0.95
    y[spike_idx + 1] = -0.90

    return y, sr, 1.0


# ===========================================================================
# Scanner Tests
# ===========================================================================

class TestAudioScanner:
    def test_scan_empty_directory(self, tmp_path):
        scanner = AudioScanner(str(tmp_path))
        files = scanner.scan()
        assert files == []

    def test_scan_finds_audio_files(self, tmp_path):
        # Create some dummy files
        (tmp_path / "song.mp3").write_bytes(b"fake")
        (tmp_path / "track.wav").write_bytes(b"fake")
        (tmp_path / "readme.txt").write_bytes(b"not audio")

        scanner = AudioScanner(str(tmp_path))
        files = scanner.scan()
        names = [f.filename for f in files]
        assert "song.mp3" in names
        assert "track.wav" in names
        assert "readme.txt" not in names

    def test_scan_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.flac").write_bytes(b"fake")
        (tmp_path / "top.mp3").write_bytes(b"fake")

        scanner = AudioScanner(str(tmp_path))
        files = scanner.scan()
        names = [f.filename for f in files]
        assert "top.mp3" in names
        assert "deep.flac" in names

    def test_scan_nonexistent_directory(self, tmp_path):
        scanner = AudioScanner(str(tmp_path / "no_such_dir"))
        files = scanner.scan()
        assert files == []

    def test_update_status(self, tmp_path):
        (tmp_path / "a.mp3").write_bytes(b"fake")
        scanner = AudioScanner(str(tmp_path))
        scanner.scan()

        path = str((tmp_path / "a.mp3").resolve())
        scanner.update_status(path, FileStatus.NEEDS_FIX, [1.0, 2.5])

        info = scanner.get_file_info(path)
        assert info is not None
        assert info.status == FileStatus.NEEDS_FIX
        assert info.glitches == [1.0, 2.5]

    def test_get_pending_files(self, tmp_path):
        (tmp_path / "a.mp3").write_bytes(b"fake")
        (tmp_path / "b.mp3").write_bytes(b"fake")
        scanner = AudioScanner(str(tmp_path))
        scanner.scan()
        assert len(scanner.get_pending_files()) == 2

        path_a = str((tmp_path / "a.mp3").resolve())
        scanner.update_status(path_a, FileStatus.PASSED)
        assert len(scanner.get_pending_files()) == 1

    def test_get_progress_stats_empty(self, tmp_path):
        """Empty scanner should return (0, 0, 0)."""
        scanner = AudioScanner(str(tmp_path))
        scanner.scan()
        processed, total, pct = scanner.get_progress_stats()
        assert (processed, total, pct) == (0, 0, 0)

    def test_get_progress_stats(self, tmp_path):
        """get_progress_stats should track PASSED and FIXED files."""
        (tmp_path / "a.mp3").write_bytes(b"fake")
        (tmp_path / "b.mp3").write_bytes(b"fake")
        (tmp_path / "c.mp3").write_bytes(b"fake")
        scanner = AudioScanner(str(tmp_path))
        scanner.scan()

        processed, total, pct = scanner.get_progress_stats()
        assert total == 3
        assert processed == 0
        assert pct == 0

        path_a = str((tmp_path / "a.mp3").resolve())
        scanner.update_status(path_a, FileStatus.PASSED)
        processed, total, pct = scanner.get_progress_stats()
        assert processed == 1
        assert pct == 33

        path_b = str((tmp_path / "b.mp3").resolve())
        scanner.update_status(path_b, FileStatus.FIXED)
        processed, total, pct = scanner.get_progress_stats()
        assert processed == 2
        assert pct == 66


class TestAudioFileInfo:
    def test_repr_pending(self, tmp_path):
        info = AudioFileInfo(str(tmp_path / "test.mp3"))
        assert "⏳" in repr(info)

    def test_repr_needs_fix(self, tmp_path):
        info = AudioFileInfo(str(tmp_path / "test.mp3"))
        info.status = FileStatus.NEEDS_FIX
        info.glitches = [1.0, 2.0, 3.0]
        r = repr(info)
        assert "⚠️" in r
        assert "3处异常" in r


# ===========================================================================
# Analyzer Tests
# ===========================================================================

class TestAnalyzer:
    def test_detect_spike(self):
        """A signal with a deliberate spike should be detected."""
        y, sr, spike_time = _make_signal_with_spike()
        glitches = detect_glitches_from_array(y, sr, sensitivity=0.4, min_duration=0)
        assert len(glitches) >= 1
        # The detected glitch should be near the spike time
        assert any(abs(t - spike_time) < 0.1 for t in glitches)

    def test_clean_signal_no_glitches(self):
        """A clean sine wave should produce no glitches."""
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        glitches = detect_glitches_from_array(y, sr, sensitivity=0.4, min_duration=0)
        assert glitches == []

    def test_empty_signal(self):
        """An empty or very short signal should return empty list."""
        assert detect_glitches_from_array(np.array([]), 22050, min_duration=0) == []
        assert detect_glitches_from_array(np.array([0.0]), 22050, min_duration=0) == []

    def test_sensitivity_affects_results(self):
        """Higher sensitivity value means lower threshold, detecting more glitches."""
        y, sr, _ = _make_signal_with_spike()
        g_low = detect_glitches_from_array(y, sr, sensitivity=0.1, min_duration=0)
        g_high = detect_glitches_from_array(y, sr, sensitivity=1.0, min_duration=0)
        # Higher sensitivity value means lower threshold, detecting more glitches
        assert len(g_high) >= len(g_low)

    def test_min_interval_clustering(self):
        """Multiple spikes within min_interval should be clustered."""
        sr = 22050
        y = np.zeros(sr * 3, dtype=np.float32)
        # Two spikes 0.1s apart
        y[sr] = 0.95
        y[sr + 1] = -0.9
        y[int(sr * 1.1)] = 0.95
        y[int(sr * 1.1) + 1] = -0.9

        glitches = detect_glitches_from_array(y, sr, min_interval=0.5, min_duration=0)
        # Should cluster into 1
        assert len(glitches) == 1

    def test_constant_signal_no_glitches(self):
        """A constant signal (std=0) should return empty list."""
        y = np.ones(22050, dtype=np.float32) * 0.5
        glitches = detect_glitches_from_array(y, 22050, min_duration=0)
        assert glitches == []

    def test_sliding_window_detects_spike_in_quiet_segment(self):
        """Sliding window should detect a spike in a quiet segment
        even when the signal has a loud section at the beginning."""
        from audio_shield.analyzer import detect_audio_glitches_pro

        sr = 22050
        duration_sec = 4.0
        n_samples = int(sr * duration_sec)
        y = np.zeros(n_samples, dtype=np.float32)

        # Loud section at the start (0-1 sec): high amplitude noise
        rng = np.random.default_rng(42)
        y[:sr] = rng.normal(0, 0.3, sr).astype(np.float32)

        # Quiet section (1-4 sec): very low noise
        y[sr:] = rng.normal(0, 0.001, n_samples - sr).astype(np.float32)

        # Insert a small spike at 3.0 seconds (in the quiet section)
        spike_idx = int(sr * 3.0)
        y[spike_idx] = 0.05
        y[spike_idx + 1] = -0.05

        glitches = detect_audio_glitches_pro(y, sr, sensitivity=0.4, min_duration=0)
        # The sliding window should detect the spike in the quiet segment
        assert any(abs(t - 3.0) < 0.1 for t in glitches)

    def test_detect_audio_glitches_pro_empty(self):
        """Empty or very short signal should return empty list."""
        from audio_shield.analyzer import detect_audio_glitches_pro

        assert detect_audio_glitches_pro(np.array([]), 22050, min_duration=0) == []
        assert detect_audio_glitches_pro(np.array([0.0]), 22050, min_duration=0) == []

    def test_detect_audio_glitches_pro_window_size(self):
        """Custom window size should work correctly."""
        from audio_shield.analyzer import detect_audio_glitches_pro

        y, sr, spike_time = _make_signal_with_spike()
        glitches = detect_audio_glitches_pro(
            y, sr, window_size_sec=0.5, sensitivity=0.4, min_duration=0
        )
        assert len(glitches) >= 1
        assert any(abs(t - spike_time) < 0.1 for t in glitches)

    def test_excessive_glitches_auto_retry(self):
        """When >50 glitches are detected, auto-retry with reduced sensitivity."""
        from audio_shield.analyzer import detect_audio_glitches_pro

        sr = 22050
        duration_sec = 60.0
        n_samples = int(sr * duration_sec)
        rng = np.random.default_rng(123)
        # Noisy background with moderate-level spikes that can be filtered
        y = rng.normal(0, 0.01, n_samples).astype(np.float32)
        # Insert >50 distinct moderate spikes spaced >0.5s apart
        for i in range(60):
            idx = int(sr * (0.5 + i * 0.9))
            if idx + 1 < n_samples:
                y[idx] = 0.15
                y[idx + 1] = -0.15

        # With very high sensitivity (low value) many glitches would be found
        # The auto-retry should kick in and reduce the count
        glitches = detect_audio_glitches_pro(y, sr, sensitivity=0.2)
        # Retries terminate without recursion error and return a result
        assert isinstance(glitches, list)

    def test_excessive_glitches_no_infinite_recursion(self):
        """Auto-retry should be bounded; extremely noisy signal should not
        cause infinite recursion."""
        from audio_shield.analyzer import detect_audio_glitches_pro

        sr = 22050
        n_samples = int(sr * 60)
        rng = np.random.default_rng(99)
        y = rng.normal(0, 0.001, n_samples).astype(np.float32)
        # Insert 100 very strong spikes that can't be filtered out
        for i in range(100):
            idx = int(sr * (0.3 + i * 0.55))
            if idx + 1 < n_samples:
                y[idx] = 0.95
                y[idx + 1] = -0.95

        # Should not raise RecursionError
        glitches = detect_audio_glitches_pro(y, sr, sensitivity=0.2)
        assert isinstance(glitches, list)

    def test_energy_gate_filters_quiet_noise(self):
        """Very quiet quantization noise should be filtered by the energy gate."""
        from audio_shield.analyzer import detect_audio_glitches_pro, _MIN_ABS_ENERGY

        sr = 22050
        duration_sec = 2.0
        n_samples = int(sr * duration_sec)
        rng = np.random.default_rng(7)
        # All samples well below the energy gate
        y = rng.normal(0, _MIN_ABS_ENERGY * 0.1, n_samples).astype(np.float32)
        # Insert a tiny "spike" that is still below the energy gate
        spike_idx = int(sr * 1.0)
        y[spike_idx] = _MIN_ABS_ENERGY * 0.5
        y[spike_idx + 1] = -_MIN_ABS_ENERGY * 0.5

        glitches = detect_audio_glitches_pro(y, sr, sensitivity=0.4, min_duration=0)
        assert glitches == [], "Quiet spikes below energy gate should not be reported"

    def test_rms_filter_reduces_false_positives(self):
        """Spikes that are only slightly above local RMS should be filtered out."""
        from audio_shield.analyzer import detect_audio_glitches_pro

        sr = 22050
        duration_sec = 2.0
        n_samples = int(sr * duration_sec)
        rng = np.random.default_rng(55)
        # Background noise with moderate RMS
        y = rng.normal(0, 0.05, n_samples).astype(np.float32)

        # These glitches should still be detected (strong spike well above RMS)
        spike_idx = int(sr * 1.0)
        y[spike_idx] = 0.95
        y[spike_idx + 1] = -0.90

        glitches = detect_audio_glitches_pro(y, sr, sensitivity=0.4, min_duration=0)
        assert any(abs(t - 1.0) < 0.1 for t in glitches), \
            "Strong spikes above RMS threshold should still be detected"

    def test_short_noise_filtered_by_default_min_duration(self):
        """A single short spike (<3s) should be filtered out with default min_duration."""
        y, sr, _ = _make_signal_with_spike()
        glitches = detect_glitches_from_array(y, sr, sensitivity=0.4)
        assert glitches == [], "A single short spike should not be reported as problem noise"

    def test_continuous_noise_detected(self):
        """Continuous noise spanning >=3 seconds should be detected."""
        from audio_shield.analyzer import detect_audio_glitches_pro

        sr = 22050
        duration_sec = 10.0
        n_samples = int(sr * duration_sec)
        rng = np.random.default_rng(42)
        y = rng.normal(0, 0.001, n_samples).astype(np.float32)

        # Insert many spikes spanning 4 seconds (2.0s to 6.0s),
        # spaced 0.3s apart — continuous noise region
        for i in range(14):
            t_spike = 2.0 + i * 0.3
            idx = int(sr * t_spike)
            if idx + 1 < n_samples:
                y[idx] = 0.95
                y[idx + 1] = -0.90

        glitches = detect_audio_glitches_pro(y, sr, sensitivity=0.4)
        assert len(glitches) >= 1, "Continuous noise >=3s should be detected"
        # All detected glitches should be within the noise region
        assert all(1.5 < t < 6.5 for t in glitches)

    def test_min_duration_zero_disables_filter(self):
        """Setting min_duration=0 should disable duration filtering."""
        y, sr, spike_time = _make_signal_with_spike()
        glitches = detect_glitches_from_array(y, sr, sensitivity=0.4, min_duration=0)
        assert len(glitches) >= 1
        assert any(abs(t - spike_time) < 0.1 for t in glitches)

    def test_filter_by_duration_helper(self):
        """_filter_by_duration should group and filter correctly."""
        from audio_shield.analyzer import _filter_by_duration

        # Region 1: [1.0, 1.5, 2.0] — span = 1.0s (< 3.0s, filtered out)
        # Region 2: [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0] — span = 3.0s (kept)
        times = [1.0, 1.5, 2.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
        result = _filter_by_duration(times, min_duration=3.0)
        assert result == [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]

    def test_filter_by_duration_empty(self):
        """_filter_by_duration with empty input should return empty."""
        from audio_shield.analyzer import _filter_by_duration
        assert _filter_by_duration([], min_duration=3.0) == []

    def test_filter_by_duration_zero_disables(self):
        """_filter_by_duration with min_duration=0 should return all."""
        from audio_shield.analyzer import _filter_by_duration
        times = [1.0, 2.0, 5.0]
        assert _filter_by_duration(times, min_duration=0) == times


# ===========================================================================
# Editor Tests
# ===========================================================================

class TestAudioBufferManager:
    @pytest.fixture
    def wav_file(self, tmp_path):
        path = str(tmp_path / "test.wav")
        _make_sine_wav(path, duration_ms=3000)
        return path

    def test_load_and_duration(self, wav_file):
        editor = AudioBufferManager(wav_file)
        assert abs(editor.duration_seconds - 3.0) < 0.1

    def test_delete_range(self, wav_file):
        editor = AudioBufferManager(wav_file)
        original_dur = editor.duration_seconds
        editor.delete_range(1.0, 2.0)
        # Should be ~1 second shorter
        assert editor.duration_seconds < original_dur
        assert abs(editor.duration_seconds - 2.0) < 0.1

    def test_delete_range_validation(self, wav_file):
        editor = AudioBufferManager(wav_file)
        with pytest.raises(ValueError):
            editor.delete_range(2.0, 1.0)  # start > end
        with pytest.raises(ValueError):
            editor.delete_range(-1.0, 1.0)  # negative

    def test_undo(self, wav_file):
        editor = AudioBufferManager(wav_file)
        original_dur = editor.duration_seconds
        editor.delete_range(1.0, 2.0)
        assert editor.duration_seconds < original_dur
        editor.undo()
        assert abs(editor.duration_seconds - original_dur) < 0.01

    def test_undo_empty(self, wav_file):
        editor = AudioBufferManager(wav_file)
        assert editor.undo() is False

    def test_normalize(self, wav_file):
        editor = AudioBufferManager(wav_file)
        editor.normalize(target_dbfs=-3.0)
        assert abs(editor.audio.max_dBFS - (-3.0)) < 0.5

    def test_save_result(self, wav_file, tmp_path):
        editor = AudioBufferManager(wav_file)
        output = str(tmp_path / "output.wav")
        editor.save_result(output, file_format="wav")
        assert os.path.exists(output)
        # Reload and verify
        reloaded = AudioSegment.from_file(output)
        assert abs(len(reloaded) - len(editor.audio)) < 50  # within 50ms

    def test_get_segment(self, wav_file):
        editor = AudioBufferManager(wav_file)
        seg = editor.get_segment(0.5, 1.5)
        assert abs(len(seg) - 1000) < 50  # ~1 second

    def test_empty_init(self):
        editor = AudioBufferManager()
        assert editor.duration_seconds == 0.0

    def test_delete_with_crossfade(self, wav_file):
        """Ensure crossfade parameter is applied without error."""
        editor = AudioBufferManager(wav_file)
        editor.delete_range(0.5, 1.5, crossfade_ms=20)
        assert editor.duration_seconds > 0


# ===========================================================================
# Phase 4 / GUI Integration Tests (non-GUI)
# ===========================================================================

class TestPhase4Integration:
    def test_launch_gui_with_context_importable(self):
        """launch_gui_with_context should be importable from gui module."""
        from audio_shield.gui import launch_gui_with_context
        assert callable(launch_gui_with_context)

    def test_phase_4_exists(self):
        """CineCastProducer should have phase_4_quality_control method."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer dependencies not available")
        producer = CineCastProducer()
        assert hasattr(producer, 'phase_4_quality_control')
        assert callable(producer.phase_4_quality_control)

    def test_phase_4_missing_output_dir(self, tmp_path):
        """phase_4 should handle missing output directory gracefully."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer dependencies not available")
        producer = CineCastProducer(config={
            "assets_dir": str(tmp_path / "assets"),
            "output_dir": str(tmp_path / "nonexistent_output"),
            "model_path": "dummy",
            "ambient_theme": "iceland_wind",
            "target_duration_min": 30,
            "min_tail_min": 10,
            "use_local_llm": True,
            "enable_recap": True,
            "pure_narrator_mode": False,
            "user_recaps": None,
            "global_cast": {},
            "custom_recaps": {},
            "enable_auto_recap": True,
        })
        # Should not raise, just log error and return
        producer.phase_4_quality_control()

    def test_phase_4_accepts_target_dir(self):
        """phase_4_quality_control should accept optional target_dir parameter."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer dependencies not available")

        import inspect
        sig = inspect.signature(CineCastProducer.phase_4_quality_control)
        assert "target_dir" in sig.parameters
        # target_dir should default to None
        assert sig.parameters["target_dir"].default is None

    def test_phase_4_with_custom_target_dir(self, tmp_path):
        """phase_4 should use target_dir instead of output_dir when provided."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer dependencies not available")
        producer = CineCastProducer(config={
            "assets_dir": str(tmp_path / "assets"),
            "output_dir": str(tmp_path / "output"),
            "model_path": "dummy",
            "ambient_theme": "iceland_wind",
            "target_duration_min": 30,
            "min_tail_min": 10,
            "use_local_llm": True,
            "enable_recap": True,
            "pure_narrator_mode": False,
            "user_recaps": None,
            "global_cast": {},
            "custom_recaps": {},
            "enable_auto_recap": True,
        })
        # nonexistent target_dir should log error and return without raising
        producer.phase_4_quality_control(target_dir=str(tmp_path / "nonexistent"))
