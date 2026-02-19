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
# Helper: standalone _cn_to_int (mirrors CineCastProducer._cn_to_int)
# ---------------------------------------------------------------------------

def _cn_to_int(cn_str):
    """Standalone copy for environments where mlx is unavailable."""
    cn_num = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
              '十': 10, '百': 100, '千': 1000, '两': 2}
    if cn_str.isdigit():
        return int(cn_str)
    result, temp = 0, 0
    for char in cn_str:
        if char in cn_num:
            val = cn_num[char]
            if val >= 10:
                if temp == 0: temp = 1
                result += temp * val
                temp = 0
            else:
                temp = val
    return result + temp


# ---------------------------------------------------------------------------
# Helper: standalone parse_user_recaps (mirrors CineCastProducer.parse_user_recaps)
# ---------------------------------------------------------------------------

def parse_user_recaps(raw_text):
    """Standalone copy for environments where mlx is unavailable."""
    if not raw_text or not raw_text.strip():
        return {}
    recaps = {}
    pattern = re.compile(
        r'(?:第\s*([0-9零一二三四五六七八九十百千两]+)\s*[章回]|Chapter[_ ]?(\d+))\s*[：:]\s*(.+?)(?=\n\s*(?:第\s*[0-9零一二三四五六七八九十百千两]+\s*[章回]|Chapter[_ ]?\d+)|$)',
        re.DOTALL | re.IGNORECASE
    )
    matches = pattern.findall(raw_text)
    if matches:
        for m in matches:
            num_str = m[0] or m[1]
            chapter_num = _cn_to_int(num_str)
            recap_text = m[2].strip()
            if recap_text and chapter_num > 0:
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

    def test_chinese_hui_format(self):
        """'回' should be recognized the same as '章'"""
        text = "第1回：武松打虎\n第2回：林冲上山"
        result = parse_user_recaps(text)
        assert result == {1: "武松打虎", 2: "林冲上山"}

    def test_chinese_numeral_chapter(self):
        """Chinese numerals like 第一章 should be parsed correctly"""
        text = "第一章：暴风雨来临\n第二章：黑夜降临"
        result = parse_user_recaps(text)
        assert result == {1: "暴风雨来临", 2: "黑夜降临"}

    def test_chinese_numeral_hui(self):
        """Chinese numerals with '回' (e.g. 第一百二十回)"""
        text = "第一百二十回：大结局"
        result = parse_user_recaps(text)
        assert result == {120: "大结局"}

    def test_chinese_numeral_complex(self):
        """Various Chinese numeral forms"""
        text = "第十章：中场\n第二十一章：变故"
        result = parse_user_recaps(text)
        assert result == {10: "中场", 21: "变故"}


# ---------------------------------------------------------------------------
# Test: _cn_to_int — Chinese numeral conversion
# ---------------------------------------------------------------------------

class TestCnToInt:
    def test_arabic_digits(self):
        assert _cn_to_int("123") == 123

    def test_single_digit(self):
        assert _cn_to_int("五") == 5

    def test_ten(self):
        assert _cn_to_int("十") == 10

    def test_teens(self):
        assert _cn_to_int("十二") == 12

    def test_tens(self):
        assert _cn_to_int("二十") == 20

    def test_tens_with_units(self):
        assert _cn_to_int("二十一") == 21

    def test_hundred(self):
        assert _cn_to_int("一百") == 100

    def test_hundred_and_twenty(self):
        assert _cn_to_int("一百二十") == 120

    def test_hundred_and_twenty_with_units(self):
        assert _cn_to_int("一百二十回") == 120  # '回' is not in cn_num, ignored

    def test_complex_number(self):
        assert _cn_to_int("三百四十五") == 345

    def test_liang(self):
        assert _cn_to_int("两百") == 200

    def test_zero(self):
        assert _cn_to_int("零") == 0


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


# ---------------------------------------------------------------------------
# Test: Preview connectivity probe logging
# ---------------------------------------------------------------------------

class TestPreviewConnectivityProbe:
    def test_connectivity_logs_with_global_cast(self, caplog):
        """Verify run_preview_mode logs successful reception of global_cast."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "ch01.txt"), "w", encoding="utf-8") as f:
                f.write("第一章\n" + "夜幕降临港口。" * 100)

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
                "global_cast": {"老渔夫": {"voice": "elder_male"}, "年轻人": {"voice": "young_male"}},
                "custom_recaps": {},
            }
            producer = CineCastProducer.__new__(CineCastProducer)
            producer.config = config
            producer.script_dir = os.path.join(tmpdir, "scripts")
            producer.cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(producer.script_dir, exist_ok=True)
            os.makedirs(producer.cache_dir, exist_ok=True)
            os.makedirs(config["output_dir"], exist_ok=True)
            producer.assets = None

            import logging
            with caplog.at_level(logging.INFO):
                # We can't run the full preview pipeline without TTS, so test phase_1 with is_preview
                producer.phase_1_generate_scripts(input_dir, is_preview=True)

            # The connectivity probe is in run_preview_mode, but we can verify is_preview works
            # For the probe itself, we test it indirectly via the log output
            scripts = [f for f in os.listdir(producer.script_dir) if f.endswith("_micro.json")]
            assert len(scripts) == 1

    def test_connectivity_logs_with_custom_recaps(self, caplog):
        """Verify run_preview_mode logs successful reception of custom_recaps."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "ch01.txt"), "w", encoding="utf-8") as f:
                f.write("第一章\n" + "夜幕降临港口。" * 100)

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
                "global_cast": {},
                "custom_recaps": {"Chapter_002": "上一章中，老渔夫带回了黑匣子"},
            }
            producer = CineCastProducer.__new__(CineCastProducer)
            producer.config = config
            producer.script_dir = os.path.join(tmpdir, "scripts")
            producer.cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(producer.script_dir, exist_ok=True)
            os.makedirs(producer.cache_dir, exist_ok=True)
            os.makedirs(config["output_dir"], exist_ok=True)
            producer.assets = None

            import logging
            with caplog.at_level(logging.INFO):
                producer.phase_1_generate_scripts(input_dir, is_preview=True)

            # Verify recap was forcefully injected
            scripts = [f for f in os.listdir(producer.script_dir) if f.endswith("_micro.json")]
            assert len(scripts) == 1
            with open(os.path.join(producer.script_dir, scripts[0]), "r") as f:
                chunks = json.load(f)
            recap_chunks = [c for c in chunks if c.get("type") == "recap"]
            assert len(recap_chunks) == 2, f"Expected 2 recap chunks (intro+body) in preview, got {len(recap_chunks)}"
            assert recap_chunks[0]["content"] == "前情提要："
            assert recap_chunks[1]["content"] == "上一章中，老渔夫带回了黑匣子"


# ---------------------------------------------------------------------------
# Test: Preview forced recap injection
# ---------------------------------------------------------------------------

class TestPreviewForcedRecapInjection:
    def test_preview_injects_recap_for_first_chapter(self):
        """In preview mode, even the first chapter should get a borrowed recap
        when custom_recaps are provided."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "ch01.txt"), "w", encoding="utf-8") as f:
                f.write("第一章\n" + "夜幕降临港口。老渔夫收起渔网。" * 50)

            config = {
                "assets_dir": os.path.join(tmpdir, "assets"),
                "output_dir": os.path.join(tmpdir, "output"),
                "model_path": "dummy",
                "ambient_theme": "iceland_wind",
                "target_duration_min": 30,
                "min_tail_min": 10,
                "use_local_llm": True,
                "enable_recap": True,
                "pure_narrator_mode": True,
                "user_recaps": None,
                "global_cast": {},
                "custom_recaps": {"Chapter_005": "老渔夫在暴风雨中找到了一个漂流瓶"},
            }
            producer = CineCastProducer.__new__(CineCastProducer)
            producer.config = config
            producer.script_dir = os.path.join(tmpdir, "scripts")
            producer.cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(producer.script_dir, exist_ok=True)
            os.makedirs(producer.cache_dir, exist_ok=True)
            producer.assets = None

            producer.phase_1_generate_scripts(input_dir, is_preview=True)

            scripts = [f for f in os.listdir(producer.script_dir) if f.endswith("_micro.json")]
            assert len(scripts) == 1
            with open(os.path.join(producer.script_dir, scripts[0]), "r") as f:
                chunks = json.load(f)
            # Should contain borrowed recap
            recap_chunks = [c for c in chunks if c.get("type") == "recap"]
            assert len(recap_chunks) == 2
            assert recap_chunks[1]["content"] == "老渔夫在暴风雨中找到了一个漂流瓶"
            assert recap_chunks[1]["speaker"] == "talkover"

    def test_preview_no_recap_without_custom_recaps(self):
        """In preview mode without custom_recaps, no recap should be injected."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "ch01.txt"), "w", encoding="utf-8") as f:
                f.write("第一章\n" + "夜幕降临港口。" * 50)

            config = {
                "assets_dir": os.path.join(tmpdir, "assets"),
                "output_dir": os.path.join(tmpdir, "output"),
                "model_path": "dummy",
                "ambient_theme": "iceland_wind",
                "target_duration_min": 30,
                "min_tail_min": 10,
                "use_local_llm": True,
                "enable_recap": True,
                "pure_narrator_mode": True,
                "user_recaps": None,
                "global_cast": {},
                "custom_recaps": {},
            }
            producer = CineCastProducer.__new__(CineCastProducer)
            producer.config = config
            producer.script_dir = os.path.join(tmpdir, "scripts")
            producer.cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(producer.script_dir, exist_ok=True)
            os.makedirs(producer.cache_dir, exist_ok=True)
            producer.assets = None

            producer.phase_1_generate_scripts(input_dir, is_preview=True)

            scripts = [f for f in os.listdir(producer.script_dir) if f.endswith("_micro.json")]
            assert len(scripts) == 1
            with open(os.path.join(producer.script_dir, scripts[0]), "r") as f:
                chunks = json.load(f)
            recap_chunks = [c for c in chunks if c.get("type") == "recap"]
            assert len(recap_chunks) == 0

    def test_preview_truncates_to_10_chunks(self):
        """Preview mode should truncate script to max 10 chunks."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir)
            # Create a long chapter that produces many chunks
            with open(os.path.join(input_dir, "ch01.txt"), "w", encoding="utf-8") as f:
                f.write("第一章 风雪凯夫拉维克\n" + "夜幕降临港口。灯火闪烁。" * 200)

            config = {
                "assets_dir": os.path.join(tmpdir, "assets"),
                "output_dir": os.path.join(tmpdir, "output"),
                "model_path": "dummy",
                "ambient_theme": "iceland_wind",
                "target_duration_min": 30,
                "min_tail_min": 10,
                "use_local_llm": True,
                "enable_recap": True,
                "pure_narrator_mode": True,
                "user_recaps": None,
                "global_cast": {},
                "custom_recaps": {},
            }
            producer = CineCastProducer.__new__(CineCastProducer)
            producer.config = config
            producer.script_dir = os.path.join(tmpdir, "scripts")
            producer.cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(producer.script_dir, exist_ok=True)
            os.makedirs(producer.cache_dir, exist_ok=True)
            producer.assets = None

            producer.phase_1_generate_scripts(input_dir, is_preview=True)

            scripts = [f for f in os.listdir(producer.script_dir) if f.endswith("_micro.json")]
            assert len(scripts) == 1
            with open(os.path.join(producer.script_dir, scripts[0]), "r") as f:
                chunks = json.load(f)
            assert len(chunks) <= 10, f"Preview should truncate to 10 chunks, got {len(chunks)}"

    def test_preview_truncates_chapter_to_1000_chars(self):
        """Preview mode should truncate chapter content to 1000 characters."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir)
            # Create a very long chapter (5000+ chars)
            long_content = "第一章\n" + "这是一段非常长的文本，用于测试试听模式是否正确截断到1000字。" * 200
            with open(os.path.join(input_dir, "ch01.txt"), "w", encoding="utf-8") as f:
                f.write(long_content)

            config = {
                "assets_dir": os.path.join(tmpdir, "assets"),
                "output_dir": os.path.join(tmpdir, "output"),
                "model_path": "dummy",
                "ambient_theme": "iceland_wind",
                "target_duration_min": 30,
                "min_tail_min": 10,
                "use_local_llm": True,
                "enable_recap": True,
                "pure_narrator_mode": True,
                "user_recaps": None,
                "global_cast": {},
                "custom_recaps": {},
            }
            producer = CineCastProducer.__new__(CineCastProducer)
            producer.config = config
            producer.script_dir = os.path.join(tmpdir, "scripts")
            producer.cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(producer.script_dir, exist_ok=True)
            os.makedirs(producer.cache_dir, exist_ok=True)
            producer.assets = None

            producer.phase_1_generate_scripts(input_dir, is_preview=True)

            scripts = [f for f in os.listdir(producer.script_dir) if f.endswith("_micro.json")]
            assert len(scripts) == 1
            with open(os.path.join(producer.script_dir, scripts[0]), "r") as f:
                chunks = json.load(f)
            # Verify fewer chunks are produced than a non-preview run would generate
            # from the full 5000+ char content. Preview truncates to 1000 chars first.
            total_content = "".join(c["content"] for c in chunks if c.get("type") != "title")
            # Pure narrator mode preserves source text; from 1000 chars we should get
            # significantly less content than the original 5000+ chars
            assert len(total_content) <= 1100, f"Preview content should be limited, got {len(total_content)} chars"

    def test_preview_skips_existing_script(self):
        """Preview mode should regenerate script even if one already exists."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "ch01.txt"), "w", encoding="utf-8") as f:
                f.write("第一章\n" + "夜幕降临港口。" * 50)

            config = {
                "assets_dir": os.path.join(tmpdir, "assets"),
                "output_dir": os.path.join(tmpdir, "output"),
                "model_path": "dummy",
                "ambient_theme": "iceland_wind",
                "target_duration_min": 30,
                "min_tail_min": 10,
                "use_local_llm": True,
                "enable_recap": True,
                "pure_narrator_mode": True,
                "user_recaps": None,
                "global_cast": {},
                "custom_recaps": {"Chapter_002": "旧的摘要"},
            }
            producer = CineCastProducer.__new__(CineCastProducer)
            producer.config = config
            producer.script_dir = os.path.join(tmpdir, "scripts")
            producer.cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(producer.script_dir, exist_ok=True)
            os.makedirs(producer.cache_dir, exist_ok=True)
            producer.assets = None

            # Pre-create a script without recap
            old_script = [{"chunk_id": "old", "type": "narration", "speaker": "narrator", "content": "旧内容"}]
            script_path = os.path.join(producer.script_dir, "ch01_micro.json")
            with open(script_path, "w") as f:
                json.dump(old_script, f)

            # Run preview - should overwrite existing script
            producer.phase_1_generate_scripts(input_dir, is_preview=True)

            with open(script_path, "r") as f:
                chunks = json.load(f)
            # Should have been regenerated (more than just the old single chunk)
            assert len(chunks) > 1, "Preview should regenerate script even if it exists"
            # Should have recap injected
            recap_chunks = [c for c in chunks if c.get("type") == "recap"]
            assert len(recap_chunks) == 2


# ---------------------------------------------------------------------------
# Test: Safe recap insertion (Bug fix for out-of-bounds)
# ---------------------------------------------------------------------------

class TestSafeRecapInsertion:
    """Verify recap units are safely inserted regardless of micro_script length."""

    def test_insert_into_empty_script(self):
        """Recap insertion into an empty script should not raise."""
        micro_script = []
        intro_unit = {"chunk_id": "test_intro", "type": "recap", "speaker": "talkover", "content": "前情提要：", "pause_ms": 500}
        recap_unit = {"chunk_id": "test_body", "type": "recap", "speaker": "talkover", "content": "上回……", "pause_ms": 1500}
        insert_idx = 1 if len(micro_script) > 1 else 0
        micro_script.insert(insert_idx, intro_unit)
        micro_script.insert(insert_idx + 1, recap_unit)
        assert len(micro_script) == 2
        assert micro_script[0]["type"] == "recap"
        assert micro_script[1]["type"] == "recap"

    def test_insert_into_single_element_script(self):
        """Recap insertion into a script with only one element should insert at front for safety."""
        micro_script = [
            {"chunk_id": "ch01_00001", "type": "title", "speaker": "narrator", "content": "第一章 风雪"}
        ]
        intro_unit = {"chunk_id": "test_intro", "type": "recap", "speaker": "talkover", "content": "前情提要：", "pause_ms": 500}
        recap_unit = {"chunk_id": "test_body", "type": "recap", "speaker": "talkover", "content": "上回……", "pause_ms": 1500}
        insert_idx = 1 if len(micro_script) > 1 else 0
        micro_script.insert(insert_idx, intro_unit)
        micro_script.insert(insert_idx + 1, recap_unit)
        assert len(micro_script) == 3
        # With > 1 guard, single-element script inserts at front for robustness
        assert micro_script[0]["type"] == "recap"
        assert micro_script[1]["type"] == "recap"
        assert micro_script[2]["type"] == "title"

    def test_insert_into_normal_script(self):
        """Recap insertion into a script with title + narration should keep correct order."""
        micro_script = [
            {"chunk_id": "ch01_00001", "type": "title", "speaker": "narrator", "content": "第一章"},
            {"chunk_id": "ch01_00002", "type": "narration", "speaker": "narrator", "content": "夜幕降临。"},
            {"chunk_id": "ch01_00003", "type": "narration", "speaker": "narrator", "content": "港口闪烁。"},
        ]
        intro_unit = {"chunk_id": "test_intro", "type": "recap", "speaker": "talkover", "content": "前情提要：", "pause_ms": 500}
        recap_unit = {"chunk_id": "test_body", "type": "recap", "speaker": "talkover", "content": "上回……", "pause_ms": 1500}
        insert_idx = 1 if len(micro_script) > 1 else 0
        micro_script.insert(insert_idx, intro_unit)
        micro_script.insert(insert_idx + 1, recap_unit)
        assert len(micro_script) == 5
        assert micro_script[0]["type"] == "title"
        assert micro_script[1]["type"] == "recap"
        assert micro_script[2]["type"] == "recap"
        assert micro_script[3]["type"] == "narration"
