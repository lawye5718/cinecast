#!/usr/bin/env python3
"""
Tests for single TXT file input support.

Covers:
- Single .txt file is read correctly as a single-chapter dict
- Single .md file is read correctly as a single-chapter dict
- Non-UTF-8 encoded TXT file returns False gracefully
- TXT directory mode still works as before
- Preview mode works with a single TXT file
"""

import os
import sys
import tempfile

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_producer(tmpdir):
    """Create a minimal CineCastProducer instance for testing."""
    try:
        from main_producer import CineCastProducer
    except ImportError:
        pytest.skip("main_producer requires mlx (macOS-only)")

    config = {
        "assets_dir": os.path.join(tmpdir, "assets"),
        "output_dir": os.path.join(tmpdir, "output"),
        "model_path": "dummy",
        "ambient_theme": "iceland_wind",
        "target_duration_min": 30,
        "min_tail_min": 10,
        "use_local_llm": True,
        "enable_recap": False,
        "pure_narrator_mode": True,  # Avoid needing Ollama
        "user_recaps": None,
    }
    producer = CineCastProducer.__new__(CineCastProducer)
    producer.config = config
    producer.script_dir = os.path.join(tmpdir, "scripts")
    producer.cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(producer.script_dir, exist_ok=True)
    os.makedirs(producer.cache_dir, exist_ok=True)
    producer.assets = None
    return producer


# ---------------------------------------------------------------------------
# Test: Single TXT file input
# ---------------------------------------------------------------------------

class TestSingleTxtFileInput:
    def test_single_txt_file_generates_script(self):
        """A single .txt file should be treated as one chapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            producer = _make_producer(tmpdir)

            txt_path = os.path.join(tmpdir, "novel.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("第一章 风雪\n夜幕降临港口。" * 20)

            result = producer.phase_1_generate_scripts(txt_path)
            assert result is True

            scripts = [f for f in os.listdir(producer.script_dir) if f.endswith("_micro.json")]
            assert len(scripts) == 1
            # Chapter key should be derived from filename without extension
            assert scripts[0].startswith("novel")

    def test_single_md_file_generates_script(self):
        """A single .md file should also be treated as one chapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            producer = _make_producer(tmpdir)

            md_path = os.path.join(tmpdir, "chapter.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("# 第一章\n这是Markdown格式的章节内容。" * 20)

            result = producer.phase_1_generate_scripts(md_path)
            assert result is True

            scripts = [f for f in os.listdir(producer.script_dir) if f.endswith("_micro.json")]
            assert len(scripts) == 1

    def test_non_utf8_txt_returns_false(self):
        """A TXT file with non-UTF-8 encoding should return False gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            producer = _make_producer(tmpdir)

            txt_path = os.path.join(tmpdir, "bad_encoding.txt")
            # Write raw GBK-encoded bytes that are invalid UTF-8
            with open(txt_path, "wb") as f:
                f.write("中文内容".encode("gbk"))

            result = producer.phase_1_generate_scripts(txt_path)
            assert result is False


# ---------------------------------------------------------------------------
# Test: TXT directory mode still works
# ---------------------------------------------------------------------------

class TestTxtDirectoryModeUnchanged:
    def test_directory_with_txt_files(self):
        """Existing directory-based TXT input should still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            producer = _make_producer(tmpdir)

            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir)
            for i in range(1, 3):
                with open(os.path.join(input_dir, f"ch{i:02d}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"第{i}章\n这是第{i}章的内容。" * 20)

            result = producer.phase_1_generate_scripts(input_dir)
            assert result is True

            scripts = [f for f in os.listdir(producer.script_dir) if f.endswith("_micro.json")]
            assert len(scripts) == 2


# ---------------------------------------------------------------------------
# Test: Preview mode with single TXT file
# ---------------------------------------------------------------------------

class TestPreviewWithSingleTxt:
    def test_preview_mode_single_txt(self):
        """Preview mode should work with a single .txt file (is_preview=True)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            producer = _make_producer(tmpdir)

            txt_path = os.path.join(tmpdir, "test_novel.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("这是一个很长的文本内容用于测试试听模式。" * 100)

            result = producer.phase_1_generate_scripts(txt_path, is_preview=True)
            assert result is True

            scripts = [f for f in os.listdir(producer.script_dir) if f.endswith("_micro.json")]
            assert len(scripts) == 1
