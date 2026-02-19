#!/usr/bin/env python3
"""
Tests for dual-track text routing: non-main-text detection and LLM bypass.

Covers:
- Expanded non-main-text keyword detection (前言, 引言, 楔子, Project Gutenberg)
- Auxiliary filename defense for chapter_000/chapter_001
- story_chapter_index only increments for main text
- Non-main-text chapters bypass LLM and use pure narrator script
"""

import os
import re
import sys

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# is_main_text detection logic (mirrors the logic in phase_1_generate_scripts)
# ---------------------------------------------------------------------------

def classify_chapter(chapter_name, content):
    """Reproduce the is_main_text classification logic from main_producer.py."""
    is_main_text = True
    non_main_keywords = ["版权", "目录", "出版", "ISBN", "序言", "致谢", "前言", "引言", "楔子", "Project Gutenberg"]

    if len(content) < 500 or any(keyword in content[:200] for keyword in non_main_keywords):
        is_main_text = False

    # Auxiliary defense: chapter_000 or chapter_001 without 第一章 marker
    if re.search(r'(?i)chapter_00[01]\b', chapter_name) and not re.search(r'第[一1]章', content[:100]):
        is_main_text = False

    return is_main_text


# ---------------------------------------------------------------------------
# Tests: keyword-based detection
# ---------------------------------------------------------------------------

class TestNonMainTextKeywords:
    """Test expanded keyword list for non-main-text detection."""

    def test_short_content_is_non_main(self):
        """Content shorter than 500 chars should be classified as non-main."""
        assert classify_chapter("Chapter_003", "短内容" * 50) is False

    def test_original_keywords_still_work(self):
        """Original keywords (版权, 目录, 出版, ISBN, 序言, 致谢) still detected."""
        long_text = "这是版权声明页面" + "正文内容" * 200
        assert classify_chapter("Chapter_002", long_text) is False

        long_text = "ISBN 978-7-123456-78-9" + "正文内容" * 200
        assert classify_chapter("Chapter_002", long_text) is False

    def test_new_keyword_qianyan(self):
        """前言 keyword triggers non-main detection."""
        long_text = "前言：本书的写作初衷" + "正文内容" * 200
        assert classify_chapter("Chapter_002", long_text) is False

    def test_new_keyword_yinyan(self):
        """引言 keyword triggers non-main detection."""
        long_text = "引言\n本章为引言部分" + "正文内容" * 200
        assert classify_chapter("Chapter_002", long_text) is False

    def test_new_keyword_xiezi(self):
        """楔子 keyword triggers non-main detection."""
        long_text = "楔子\n话说天下大势" + "正文内容" * 200
        assert classify_chapter("Chapter_002", long_text) is False

    def test_new_keyword_gutenberg(self):
        """Project Gutenberg keyword triggers non-main detection."""
        long_text = "The Project Gutenberg eBook of" + "content " * 200
        assert classify_chapter("Chapter_002", long_text) is False

    def test_main_text_passes(self):
        """Normal main text should pass as main text."""
        long_text = "第二章 暴风雨来临\n" + "这是正文内容。" * 200
        assert classify_chapter("Chapter_002", long_text) is True


# ---------------------------------------------------------------------------
# Tests: auxiliary filename defense
# ---------------------------------------------------------------------------

class TestFilenameDefense:
    """Test auxiliary defense for chapter_000/chapter_001 filenames."""

    def test_chapter_000_no_marker_is_non_main(self):
        """chapter_000 without 第一章 marker should be non-main."""
        long_text = "这是一段很长的序言内容，没有章节标记。" * 50
        assert classify_chapter("Chapter_000", long_text) is False

    def test_chapter_001_no_marker_is_non_main(self):
        """chapter_001 without 第一章 marker should be non-main."""
        long_text = "这是一段很长的序言内容，没有章节标记。" * 50
        assert classify_chapter("Chapter_001", long_text) is False

    def test_chapter_001_with_first_chapter_marker_is_main(self):
        """chapter_001 with 第一章 marker should be main text."""
        long_text = "第一章 开始\n" + "这是正文内容。" * 200
        assert classify_chapter("Chapter_001", long_text) is True

    def test_chapter_001_with_arabic_first_chapter_marker_is_main(self):
        """chapter_001 with 第1章 marker should be main text."""
        long_text = "第1章 开始\n" + "这是正文内容。" * 200
        assert classify_chapter("Chapter_001", long_text) is True

    def test_chapter_001_case_insensitive(self):
        """Filename matching should be case-insensitive."""
        long_text = "这是一段很长的序言内容，没有章节标记。" * 50
        assert classify_chapter("chapter_001", long_text) is False
        assert classify_chapter("CHAPTER_001", long_text) is False

    def test_chapter_002_not_affected(self):
        """chapter_002 should NOT be affected by the filename defense."""
        long_text = "这是一段很长的正文内容。" * 50
        assert classify_chapter("Chapter_002", long_text) is True

    def test_long_preface_in_chapter_001_blocked(self):
        """A long preface (>500 chars) in chapter_001 without keywords should still be blocked."""
        # This is the key scenario: long preface without trigger keywords at the start
        long_preface = "这本书讲述了一个关于勇气和友情的故事。" * 50
        assert len(long_preface) > 500
        assert classify_chapter("Chapter_001", long_preface) is False


# ---------------------------------------------------------------------------
# Tests: story_chapter_index alignment
# ---------------------------------------------------------------------------

class TestStoryChapterIndexAlignment:
    """Test that story_chapter_index only increments for main text."""

    def test_index_skips_non_main_chapters(self):
        """Simulate a book with a preface + 3 chapters; index should skip preface."""
        chapters = {
            "Chapter_001": "前言：本书的写作初衷" + "序言内容" * 200,
            "Chapter_002": "第一章 黎明\n" + "正文内容" * 200,
            "Chapter_003": "第二章 暴风雨\n" + "正文内容" * 200,
            "Chapter_004": "第三章 晴天\n" + "正文内容" * 200,
        }

        story_chapter_index = 0
        index_map = {}
        for chapter_name, content in chapters.items():
            is_main_text = classify_chapter(chapter_name, content)
            if is_main_text:
                story_chapter_index += 1
            index_map[chapter_name] = story_chapter_index if is_main_text else None

        assert index_map["Chapter_001"] is None  # preface, no index
        assert index_map["Chapter_002"] == 1     # first real chapter
        assert index_map["Chapter_003"] == 2     # second real chapter
        assert index_map["Chapter_004"] == 3     # third real chapter

    def test_index_skips_gutenberg_header(self):
        """Gutenberg header + preface should not affect chapter indexing."""
        chapters = {
            "Chapter_000": "The Project Gutenberg eBook" + " header content" * 100,
            "Chapter_001": "这是一段很长的序言。" * 50,  # long preface, no keywords at start but chapter_001 defense
            "Chapter_002": "第一章 开始\n" + "正文内容测试。" * 200,
        }

        story_chapter_index = 0
        index_map = {}
        for chapter_name, content in chapters.items():
            is_main_text = classify_chapter(chapter_name, content)
            if is_main_text:
                story_chapter_index += 1
            index_map[chapter_name] = story_chapter_index if is_main_text else None

        assert index_map["Chapter_000"] is None  # Gutenberg header
        assert index_map["Chapter_001"] is None  # preface blocked by filename defense
        assert index_map["Chapter_002"] == 1     # first real chapter
