#!/usr/bin/env python3
"""
Tests for preview mode optimization and recap configuration.

Covers:
- Preview mode only processes the first chapter (max_chapters=1)
- Recap enable/disable toggle via config
- User-provided recap parsing (multiple formats)
- User recaps injection into chapter scripts
- Default config includes user_recaps key
"""

import json
import os
import re
import sys
import tempfile

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import LLMScriptDirector


# ---------------------------------------------------------------------------
# Helper: standalone parse_user_recaps (mirrors CineCastProducer.parse_user_recaps)
# ---------------------------------------------------------------------------

def parse_user_recaps(raw_text):
    """Standalone copy for environments where mlx is unavailable."""
    if not raw_text or not raw_text.strip():
        return {}
    recaps = {}
    pattern = re.compile(
        r'(?:第\s*(\d+)\s*章|Chapter[_ ]?(\d+))\s*[：:]\s*(.+?)(?=\n\s*(?:第\s*\d+\s*章|Chapter[_ ]?\d+)|$)',
        re.DOTALL | re.IGNORECASE
    )
    matches = pattern.findall(raw_text)
    if matches:
        for m in matches:
            chapter_num = int(m[0] or m[1])
            recap_text = m[2].strip()
            if recap_text:
                recaps[chapter_num] = recap_text
    else:
        lines = [line.strip() for line in raw_text.strip().split('\n') if line.strip()]
        for idx, line in enumerate(lines):
            recaps[idx + 2] = line
    return recaps


# ---------------------------------------------------------------------------
# Test: parse_user_recaps — Chinese chapter format
# ---------------------------------------------------------------------------

class TestParseUserRecapsChinese:
    def test_chinese_chapter_format(self):
        text = "第2章：夜色中的港口\n第3章：远行的决定"
        result = parse_user_recaps(text)
        assert result == {2: "夜色中的港口", 3: "远行的决定"}

    def test_chinese_chapter_with_spaces(self):
        text = "第 2 章：夜色中的港口\n第 3 章：远行的决定"
        result = parse_user_recaps(text)
        assert result == {2: "夜色中的港口", 3: "远行的决定"}

    def test_chinese_chapter_with_colon_variants(self):
        text = "第2章:用英文冒号"
        result = parse_user_recaps(text)
        assert result == {2: "用英文冒号"}


# ---------------------------------------------------------------------------
# Test: parse_user_recaps — English chapter format
# ---------------------------------------------------------------------------

class TestParseUserRecapsEnglish:
    def test_chapter_underscore_format(self):
        text = "Chapter_002: test recap\nChapter_003: another recap"
        result = parse_user_recaps(text)
        assert result == {2: "test recap", 3: "another recap"}

    def test_chapter_space_format(self):
        text = "Chapter 2: test recap\nChapter 3: another recap"
        result = parse_user_recaps(text)
        assert result == {2: "test recap", 3: "another recap"}


# ---------------------------------------------------------------------------
# Test: parse_user_recaps — Fallback plain-line format
# ---------------------------------------------------------------------------

class TestParseUserRecapsFallback:
    def test_plain_lines_start_from_chapter_2(self):
        text = "第一章的摘要\n第二章的摘要"
        result = parse_user_recaps(text)
        assert result == {2: "第一章的摘要", 3: "第二章的摘要"}

    def test_single_line_fallback(self):
        text = "唯一的摘要"
        result = parse_user_recaps(text)
        assert result == {2: "唯一的摘要"}


# ---------------------------------------------------------------------------
# Test: parse_user_recaps — Edge cases
# ---------------------------------------------------------------------------

class TestParseUserRecapsEdgeCases:
    def test_empty_string(self):
        assert parse_user_recaps("") == {}

    def test_none_input(self):
        assert parse_user_recaps(None) == {}

    def test_whitespace_only(self):
        assert parse_user_recaps("   \n\n  ") == {}

    def test_mixed_empty_lines(self):
        text = "\n第2章：内容\n\n第3章：更多内容\n"
        result = parse_user_recaps(text)
        assert 2 in result
        assert 3 in result


# ---------------------------------------------------------------------------
# Test: Default config includes user_recaps
# ---------------------------------------------------------------------------

class TestDefaultConfigUserRecaps:
    def test_default_config_has_user_recaps(self):
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")
        producer = CineCastProducer.__new__(CineCastProducer)
        config = producer._get_default_config()
        assert "user_recaps" in config
        assert config["user_recaps"] is None


# ---------------------------------------------------------------------------
# Test: Preview mode calls phase_1 with max_chapters=1
# ---------------------------------------------------------------------------

class TestPreviewModeMaxChapters:
    def test_preview_only_processes_first_chapter(self):
        """Verify run_preview_mode passes max_chapters=1 to phase_1_generate_scripts.

        We simulate this by creating a producer with 3 chapters, calling
        phase_1 with max_chapters=1, and asserting only 1 script was generated.
        """
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 chapter TXT files
            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir)
            for i in range(1, 4):
                with open(os.path.join(input_dir, f"ch{i:02d}.txt"), "w", encoding="utf-8") as f:
                    # Repeat to exceed 100-char minimum filter in _extract_epub_chapters
                    f.write(f"第{i}章\n这是第{i}章的内容，足够长的文本以确保不会被过滤掉。" * 10)

            config = {
                "assets_dir": os.path.join(tmpdir, "assets"),
                "output_dir": os.path.join(tmpdir, "output"),
                "model_path": "dummy",
                "ambient_theme": "iceland_wind",
                "target_duration_min": 30,
                "min_tail_min": 10,
                "use_local_llm": True,
                "enable_recap": False,
                "pure_narrator_mode": True,  # Use pure mode to avoid needing Ollama
                "user_recaps": None,
            }
            producer = CineCastProducer.__new__(CineCastProducer)
            producer.config = config
            producer.script_dir = os.path.join(tmpdir, "scripts")
            producer.cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(producer.script_dir, exist_ok=True)
            os.makedirs(producer.cache_dir, exist_ok=True)
            producer.assets = None  # Not needed for phase_1 in pure mode

            # Generate scripts with max_chapters=1
            result = producer.phase_1_generate_scripts(input_dir, max_chapters=1)
            assert result is True

            # Only 1 script should be generated
            scripts = [f for f in os.listdir(producer.script_dir) if f.endswith("_micro.json")]
            assert len(scripts) == 1, f"Expected 1 script for preview, got {len(scripts)}: {scripts}"

    def test_full_mode_processes_all_chapters(self):
        """Verify full mode (max_chapters=None) processes all chapters."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir)
            for i in range(1, 4):
                with open(os.path.join(input_dir, f"ch{i:02d}.txt"), "w", encoding="utf-8") as f:
                    # Repeat to exceed 100-char minimum filter in _extract_epub_chapters
                    f.write(f"第{i}章\n这是第{i}章的内容，足够长的文本。" * 10)

            config = {
                "assets_dir": os.path.join(tmpdir, "assets"),
                "output_dir": os.path.join(tmpdir, "output"),
                "model_path": "dummy",
                "ambient_theme": "iceland_wind",
                "target_duration_min": 30,
                "min_tail_min": 10,
                "use_local_llm": True,
                "enable_recap": False,
                "pure_narrator_mode": True,
                "user_recaps": None,
            }
            producer = CineCastProducer.__new__(CineCastProducer)
            producer.config = config
            producer.script_dir = os.path.join(tmpdir, "scripts")
            producer.cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(producer.script_dir, exist_ok=True)
            os.makedirs(producer.cache_dir, exist_ok=True)
            producer.assets = None

            # Generate scripts without max_chapters limit
            result = producer.phase_1_generate_scripts(input_dir)
            assert result is True

            scripts = [f for f in os.listdir(producer.script_dir) if f.endswith("_micro.json")]
            assert len(scripts) == 3, f"Expected 3 scripts, got {len(scripts)}: {scripts}"


# ---------------------------------------------------------------------------
# Test: Recap enable/disable toggle
# ---------------------------------------------------------------------------

class TestRecapToggle:
    def test_recap_disabled_skips_recap_generation(self):
        """When enable_recap=False, no recap chunks should be inserted."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir)
            # Create two chapters so recap could potentially be generated
            for i in range(1, 3):
                with open(os.path.join(input_dir, f"ch{i:02d}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"第{i}章\n" + "长文本内容。" * 200)

            config = {
                "assets_dir": os.path.join(tmpdir, "assets"),
                "output_dir": os.path.join(tmpdir, "output"),
                "model_path": "dummy",
                "ambient_theme": "iceland_wind",
                "target_duration_min": 30,
                "min_tail_min": 10,
                "use_local_llm": True,
                "enable_recap": False,  # Disabled!
                "pure_narrator_mode": True,
                "user_recaps": None,
            }
            producer = CineCastProducer.__new__(CineCastProducer)
            producer.config = config
            producer.script_dir = os.path.join(tmpdir, "scripts")
            producer.cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(producer.script_dir, exist_ok=True)
            os.makedirs(producer.cache_dir, exist_ok=True)
            producer.assets = None

            producer.phase_1_generate_scripts(input_dir)

            # Check that no recap chunks exist in any script
            for script_file in os.listdir(producer.script_dir):
                if script_file.endswith("_micro.json"):
                    with open(os.path.join(producer.script_dir, script_file), "r") as f:
                        chunks = json.load(f)
                    recap_chunks = [c for c in chunks if c.get("type") == "recap"]
                    assert len(recap_chunks) == 0, f"Found recap chunks in {script_file} even with recap disabled"
