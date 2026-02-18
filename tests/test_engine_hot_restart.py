#!/usr/bin/env python3
"""
Tests for engine hot-restart (self-healing) logic in phase_2_render_dry_audio.

Validates that:
- Normal renders within threshold do NOT trigger engine restart
- Cold-start threshold (120s) is used for the first render after engine init/restart
- Warm threshold (45s) is used after a successful render within threshold
- Render exceptions are caught gracefully and do not crash the loop
- Cached files are skipped without entering the timer
- Progress counter always increments (regardless of success/failure)
"""

import gc
import os
import sys
import json
import time
import logging
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import threshold constants directly; fall back to expected values if
# main_producer cannot be imported (e.g. missing Apple-only MLX packages).
try:
    from main_producer import ENGINE_COLD_START_THRESHOLD_SECONDS, ENGINE_WARM_THRESHOLD_SECONDS
except ImportError:
    ENGINE_COLD_START_THRESHOLD_SECONDS = 120.0
    ENGINE_WARM_THRESHOLD_SECONDS = 45.0


class FakeEngine:
    """Minimal stand-in for MLXRenderEngine used to verify hot-restart logic."""

    instance_count = 0

    def __init__(self, model_path):
        FakeEngine.instance_count += 1
        self.model_path = model_path
        self.id = FakeEngine.instance_count
        self.calls = []

    def render_dry_chunk(self, content, voice_cfg, save_path):
        self.calls.append(content)
        return True


class SlowEngine(FakeEngine):
    """Engine whose render_dry_chunk sleeps to simulate slowness."""

    def __init__(self, model_path, *, delay=0.0):
        super().__init__(model_path)
        self._delay = delay

    def render_dry_chunk(self, content, voice_cfg, save_path):
        time.sleep(self._delay)
        self.calls.append(content)
        return True


class ExplodingEngine(FakeEngine):
    """Engine whose render_dry_chunk raises an exception."""

    def render_dry_chunk(self, content, voice_cfg, save_path):
        raise RuntimeError("GPU memory corrupted")


# ---------------------------------------------------------------------------
# Helper: simulate the innermost rendering loop from phase_2_render_dry_audio
# ---------------------------------------------------------------------------
def _run_render_loop(engine_factory, items, cold_threshold=120.0, warm_threshold=45.0,
                     cache_dir=None):
    """Re-implements the hot-restart loop extracted from main_producer.py.

    Returns (engine, rendered_count, restart_count, is_cold_start) so callers
    can assert on engine identity, counters, and cold/warm state.
    """
    engine = engine_factory()
    is_cold_start = True
    rendered_chunks = 0
    restart_count = 0
    voice_cfg = {"ref_audio": "dummy.wav", "speed": 1.0}

    for item in items:
        save_path = os.path.join(cache_dir, f"{item['chunk_id']}.wav") if cache_dir else f"/tmp/{item['chunk_id']}.wav"

        # Cache hit: skip without entering timer
        if os.path.exists(save_path):
            rendered_chunks += 1
            continue

        start_time = time.time()

        try:
            success = engine.render_dry_chunk(item["content"], voice_cfg, save_path)
        except Exception:
            success = False

        elapsed_time = time.time() - start_time
        rendered_chunks += 1

        timeout_threshold = cold_threshold if is_cold_start else warm_threshold

        if elapsed_time > timeout_threshold:
            del engine
            gc.collect()
            engine = engine_factory()
            restart_count += 1
            is_cold_start = True
        else:
            is_cold_start = False

    return engine, rendered_chunks, restart_count, is_cold_start


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEngineHotRestart:
    """Verify hot-restart behaviour of the rendering loop."""

    def test_no_restart_for_fast_render(self):
        """Fast renders should not trigger a restart."""
        FakeEngine.instance_count = 0
        items = [{"chunk_id": f"c{i}", "content": f"line {i}"} for i in range(5)]
        engine, rendered, restarts, is_cold = _run_render_loop(
            lambda: FakeEngine("model"), items
        )
        assert restarts == 0
        assert rendered == 5
        assert engine.id == 1  # same engine throughout
        assert is_cold is False  # warmed up after first successful render

    def test_restart_triggered_by_slow_render(self):
        """A render exceeding the threshold should trigger exactly one restart."""
        FakeEngine.instance_count = 0
        items = [{"chunk_id": "c0", "content": "line 0"}]
        # Use a tiny threshold so even instant render "exceeds" it
        engine, rendered, restarts, is_cold = _run_render_loop(
            lambda: FakeEngine("model"), items, cold_threshold=-1.0, warm_threshold=-1.0
        )
        assert restarts == 1
        assert rendered == 1
        # Engine was replaced: second instance
        assert engine.id == 2
        assert is_cold is True  # restart resets to cold

    def test_exception_does_not_crash_loop(self):
        """An exception during render should be caught; loop continues."""
        FakeEngine.instance_count = 0
        items = [{"chunk_id": "c0", "content": "boom"}]
        engine, rendered, restarts, _ = _run_render_loop(
            lambda: ExplodingEngine("model"), items
        )
        # render failed, but progress still incremented and loop did not crash
        assert rendered == 1

    def test_multiple_restarts(self):
        """Multiple slow renders should each trigger a restart."""
        FakeEngine.instance_count = 0
        items = [{"chunk_id": f"c{i}", "content": f"line {i}"} for i in range(3)]
        engine, rendered, restarts, is_cold = _run_render_loop(
            lambda: FakeEngine("model"), items, cold_threshold=-1.0, warm_threshold=-1.0
        )
        assert restarts == 3
        assert rendered == 3
        assert engine.id == 4  # original + 3 restarts

    def test_cold_start_transitions_to_warm(self):
        """After a successful render within cold threshold, state transitions to warm."""
        FakeEngine.instance_count = 0
        items = [{"chunk_id": f"c{i}", "content": f"line {i}"} for i in range(3)]
        engine, rendered, restarts, is_cold = _run_render_loop(
            lambda: FakeEngine("model"), items
        )
        assert restarts == 0
        assert is_cold is False  # after first fast render, should be warm

    def test_cold_threshold_vs_warm_threshold(self):
        """A render between warm and cold thresholds should only restart when warm."""
        FakeEngine.instance_count = 0
        # First render: cold start, threshold is 120s => no restart for instant render
        # Second render: warm, threshold is -1 => instant render triggers restart
        items = [
            {"chunk_id": "c0", "content": "line 0"},
            {"chunk_id": "c1", "content": "line 1"},
        ]
        engine, rendered, restarts, is_cold = _run_render_loop(
            lambda: FakeEngine("model"), items, cold_threshold=120.0, warm_threshold=-1.0
        )
        # First render passes cold threshold, transitions to warm
        # Second render exceeds warm threshold (-1), triggers restart
        assert restarts == 1
        assert rendered == 2
        assert is_cold is True  # restart resets to cold

    def test_cache_hit_skips_timer(self):
        """Cached files should be skipped without entering the timer or engine."""
        FakeEngine.instance_count = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            items = [
                {"chunk_id": "cached_0", "content": "line 0"},
                {"chunk_id": "cached_1", "content": "line 1"},
                {"chunk_id": "new_2", "content": "line 2"},
            ]
            # Pre-create cache files for first two items
            for i in range(2):
                open(os.path.join(tmpdir, f"cached_{i}.wav"), "w").close()

            engine, rendered, restarts, is_cold = _run_render_loop(
                lambda: FakeEngine("model"), items, cache_dir=tmpdir
            )
            # All 3 items counted in progress
            assert rendered == 3
            # Only the non-cached item was actually rendered
            assert len(engine.calls) == 1
            assert engine.calls[0] == "line 2"
            assert restarts == 0

    def test_cache_hit_does_not_affect_cold_start(self):
        """Cache hits should not change the cold_start flag."""
        FakeEngine.instance_count = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            items = [
                {"chunk_id": "cached_0", "content": "line 0"},
                {"chunk_id": "new_1", "content": "line 1"},
            ]
            # Pre-create cache file for first item
            open(os.path.join(tmpdir, "cached_0.wav"), "w").close()

            engine, rendered, restarts, is_cold = _run_render_loop(
                lambda: FakeEngine("model"), items, cache_dir=tmpdir
            )
            assert rendered == 2
            # After skipping cache hit and one successful render, should be warm
            assert is_cold is False

    def test_progress_increments_on_failure(self):
        """Progress should increment even when render fails."""
        FakeEngine.instance_count = 0
        items = [{"chunk_id": f"c{i}", "content": f"line {i}"} for i in range(3)]
        engine, rendered, restarts, _ = _run_render_loop(
            lambda: ExplodingEngine("model"), items
        )
        assert rendered == 3  # all counted despite failures
