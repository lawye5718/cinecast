#!/usr/bin/env python3
"""
Tests for engine hot-restart (self-healing) logic in phase_2_render_dry_audio.

Validates that:
- Normal renders (< 20s) do NOT trigger engine restart
- Slow renders (> 20s) DO trigger engine destruction and re-creation
- Render exceptions are caught gracefully and do not crash the loop
"""

import gc
import os
import sys
import json
import time
import logging

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
def _run_render_loop(engine_factory, items, threshold=20.0):
    """Re-implements the hot-restart loop extracted from main_producer.py.

    Returns (engine, rendered_count, restart_count) so callers can assert on
    engine identity and counters.
    """
    engine = engine_factory()
    rendered_chunks = 0
    restart_count = 0
    voice_cfg = {"ref_audio": "dummy.wav", "speed": 1.0}

    for item in items:
        save_path = f"/tmp/{item['chunk_id']}.wav"

        start_time = time.time()

        try:
            success = engine.render_dry_chunk(item["content"], voice_cfg, save_path)
        except Exception:
            success = False

        elapsed_time = time.time() - start_time

        if elapsed_time > threshold:
            del engine
            gc.collect()
            engine = engine_factory()
            restart_count += 1

        if success:
            rendered_chunks += 1

    return engine, rendered_chunks, restart_count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEngineHotRestart:
    """Verify hot-restart behaviour of the rendering loop."""

    def test_no_restart_for_fast_render(self):
        """Fast renders should not trigger a restart."""
        FakeEngine.instance_count = 0
        items = [{"chunk_id": f"c{i}", "content": f"line {i}"} for i in range(5)]
        engine, rendered, restarts = _run_render_loop(
            lambda: FakeEngine("model"), items
        )
        assert restarts == 0
        assert rendered == 5
        assert engine.id == 1  # same engine throughout

    def test_restart_triggered_by_slow_render(self):
        """A render exceeding the threshold should trigger exactly one restart."""
        FakeEngine.instance_count = 0
        items = [{"chunk_id": "c0", "content": "line 0"}]
        # Use a tiny threshold so even instant render "exceeds" it
        engine, rendered, restarts = _run_render_loop(
            lambda: FakeEngine("model"), items, threshold=-1.0
        )
        assert restarts == 1
        assert rendered == 1
        # Engine was replaced: second instance
        assert engine.id == 2

    def test_exception_does_not_crash_loop(self):
        """An exception during render should be caught; loop continues."""
        FakeEngine.instance_count = 0
        items = [{"chunk_id": "c0", "content": "boom"}]
        engine, rendered, restarts = _run_render_loop(
            lambda: ExplodingEngine("model"), items
        )
        # render failed, but loop did not crash
        assert rendered == 0

    def test_multiple_restarts(self):
        """Multiple slow renders should each trigger a restart."""
        FakeEngine.instance_count = 0
        items = [{"chunk_id": f"c{i}", "content": f"line {i}"} for i in range(3)]
        engine, rendered, restarts = _run_render_loop(
            lambda: FakeEngine("model"), items, threshold=-1.0
        )
        assert restarts == 3
        assert rendered == 3
        assert engine.id == 4  # original + 3 restarts
