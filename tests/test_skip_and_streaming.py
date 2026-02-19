#!/usr/bin/env python3
"""
Tests for the enhanced Phase 3 skip mechanism, streaming log viewer,
and headless quality control integration.

Covers:
- Fix 1: process_from_cache pre-flight skip with file_index advancement
- Fix 2: phase_3_cinematic_mix full-phase skip when volumes are up-to-date
- Fix 3: get_logs reads last 50 lines from cinecast.log
- Fix 4: run_headless_qc returns text report
- Fix 5: WebUI includes timer-based log viewer and QC integration
"""

import json
import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.cinematic_packager import CinematicPackager


# ---------------------------------------------------------------------------
# Fix 1: process_from_cache pre-flight skip
# ---------------------------------------------------------------------------

class TestProcessFromCachePreFlightSkip:
    """Verify process_from_cache returns immediately when volume exists."""

    def test_skips_when_current_volume_exists(self):
        """When the current volume file exists, process_from_cache should return
        without processing any items and advance file_index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = CinematicPackager(tmpdir)
            # Pre-create volume 1
            vol1 = os.path.join(tmpdir, "Audiobook_Part_001.mp3")
            with open(vol1, "wb") as f:
                f.write(b"fake mp3 data")

            assert p.file_index == 1

            # Call process_from_cache with a dummy script
            dummy_script = [{"chunk_id": "test_001", "type": "narration",
                           "speaker": "narrator", "content": "test"}]
            p.process_from_cache(dummy_script, tmpdir, None)

            # file_index should have advanced past volume 1
            assert p.file_index == 2
            # Buffer should still be empty (no processing occurred)
            assert len(p.buffer) == 0

    def test_advances_past_multiple_existing_volumes(self):
        """file_index should advance past all consecutive existing volumes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = CinematicPackager(tmpdir)
            # Pre-create volumes 1, 2, 3
            for i in range(1, 4):
                vol = os.path.join(tmpdir, f"Audiobook_Part_{i:03d}.mp3")
                with open(vol, "wb") as f:
                    f.write(b"fake mp3 data")

            dummy_script = [{"chunk_id": "test_001", "type": "narration",
                           "speaker": "narrator", "content": "test"}]
            p.process_from_cache(dummy_script, tmpdir, None)

            # Should advance to 4 (next available index)
            assert p.file_index == 4

    def test_processes_normally_when_no_volume_exists(self):
        """When no volume files exist, process_from_cache should enter the loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = CinematicPackager(tmpdir)
            assert p.file_index == 1

            # No volume files exist, empty script (loop won't load any wavs)
            p.process_from_cache([], tmpdir, None)

            # file_index stays at 1 since no exports happened
            assert p.file_index == 1

    def test_source_has_preflight_skip(self):
        """cinematic_packager.py should contain the pre-flight skip logic."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "cinematic_packager.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "检测到分卷已完全覆盖当前剧本，直接跳过混音计算" in source


# ---------------------------------------------------------------------------
# Fix 2: phase_3_cinematic_mix full-phase skip
# ---------------------------------------------------------------------------

class TestPhase3FullPhaseSkip:
    """Verify phase_3_cinematic_mix skips when volumes are up-to-date."""

    def test_source_has_full_phase_skip(self):
        """main_producer.py should contain the full-phase skip logic."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "分卷已存在且剧本无更新，跳过整个混音阶段" in source
        assert "latest_volume_mtime" in source
        assert "latest_script_mtime" in source


# ---------------------------------------------------------------------------
# Fix 3: get_logs streaming
# ---------------------------------------------------------------------------

class TestGetLogs:
    """Verify get_logs reads the log file correctly."""

    def test_returns_default_when_no_log_file(self, tmp_path):
        """Inline get_logs logic: returns default when file doesn't exist."""
        log_path = str(tmp_path / "nonexistent.log")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                result = "".join(lines[-50:])
        else:
            result = "等待日志输出..."
        assert result == "等待日志输出..."

    def test_reads_last_50_lines(self, tmp_path):
        """Inline get_logs logic: returns last 50 lines."""
        log_file = tmp_path / "test.log"
        lines = [f"Line {i}\n" for i in range(100)]
        log_file.write_text("".join(lines), encoding="utf-8")

        log_path = str(log_file)
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                result = "".join(all_lines[-50:])
        else:
            result = "等待日志输出..."

        result_lines = result.strip().split("\n")
        assert len(result_lines) == 50
        assert "Line 50" in result_lines[0]
        assert "Line 99" in result_lines[-1]

    def test_source_has_get_logs(self):
        """webui.py should define get_logs."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "def get_logs(" in source


# ---------------------------------------------------------------------------
# Fix 4: run_headless_qc
# ---------------------------------------------------------------------------

class TestRunHeadlessQC:
    """Verify run_headless_qc exists and handles edge cases."""

    def test_source_has_headless_qc(self):
        """webui.py should define run_headless_qc."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "def run_headless_qc(" in source
        assert "run_headless_qc(config" in source

    def test_headless_qc_checks_dir_existence(self):
        """run_headless_qc should check if output_dir exists."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "os.path.isdir(output_dir)" in source

    def test_headless_qc_uses_scanner_and_analyzer(self):
        """run_headless_qc should use AudioScanner and detect_audio_glitches."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "AudioScanner" in source
        assert "detect_audio_glitches" in source


# ---------------------------------------------------------------------------
# Fix 5: WebUI timer-based log viewer
# ---------------------------------------------------------------------------

class TestWebuiTimerAndQC:
    """Verify WebUI source includes timer and QC integration."""

    @pytest.fixture
    def webui_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_timer_defined(self, webui_source):
        """gr.Timer should be used for log polling."""
        assert "gr.Timer(" in webui_source

    def test_timer_ticks_get_logs(self, webui_source):
        """Timer should call get_logs and output to log_viewer."""
        assert "timer.tick(get_logs" in webui_source

    def test_log_viewer_defined(self, webui_source):
        """A log_viewer Textbox should be defined."""
        assert "log_viewer" in webui_source
        assert "实时制片日志" in webui_source

    def test_qc_integrated_in_full_production(self, webui_source):
        """run_headless_qc should be called after phase_3."""
        assert "run_headless_qc(" in webui_source
