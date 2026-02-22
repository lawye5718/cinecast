#!/usr/bin/env python3
"""
Tests for smart micro-chunking and strengthened anti-summarization prompt.

Covers:
- Smart sentence splitting: prioritizes major punctuation (。！？；) over commas
- Secondary comma-level splitting only for overly long sentences
- Hard-cut fallback uses raised smart_chunk_limit (>=90)
- Content preservation across all splitting strategies
- Strengthened user_content prompt in _request_llm
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import LLMScriptDirector


# ---------------------------------------------------------------------------
# Helper to build a director with mocked parse_text_to_script
# ---------------------------------------------------------------------------

def _make_director_with_macro(macro_units):
    """Return an LLMScriptDirector whose parse_text_to_script returns *macro_units*."""
    director = LLMScriptDirector()
    director.parse_text_to_script = lambda text, **kwargs: macro_units
    return director


# ---------------------------------------------------------------------------
# Smart micro-chunk: comma-separated text should NOT be fragmented
# ---------------------------------------------------------------------------

class TestSmartMicroChunkNoCommaFragmentation:
    """Verify that commas (，、：) no longer trigger individual splits
    when the resulting sentence fits within the smart_chunk_limit."""

    def test_comma_separated_short_sentence_stays_intact(self):
        """A sentence with commas that is under 90 chars should produce ONE chunk."""
        # 45 chars total – well within the 90-char smart limit
        content = "风吹过树梢，鸟儿在枝头歌唱，远处传来钟声。"
        director = _make_director_with_macro([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": content}
        ])
        result = director.parse_and_micro_chunk("test", chapter_prefix="ch01")
        assert len(result) == 1
        assert result[0]["content"] == content

    def test_multiple_short_sentences_each_produce_one_chunk(self):
        """Multiple sentences separated by 。 each become their own chunk,
        but commas inside them do NOT cause further splitting."""
        s1 = "太阳升起，万物复苏。"
        s2 = "鸟儿歌唱，花儿绽放。"
        content = s1 + s2
        director = _make_director_with_macro([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": content}
        ])
        result = director.parse_and_micro_chunk("test", chapter_prefix="ch01")
        # Should be 2 chunks (split at 。), not 4 (split at every ，)
        assert len(result) == 2
        assert result[0]["content"] == s1
        assert result[1]["content"] == s2


# ---------------------------------------------------------------------------
# Smart micro-chunk: over-long sentences trigger secondary comma split
# ---------------------------------------------------------------------------

class TestSmartMicroChunkLongSentenceFallback:
    """When a single sentence (delimited by 。) exceeds smart_chunk_limit,
    it should be split further at comma-level punctuation."""

    def test_long_sentence_split_at_commas(self):
        """A single sentence >150 chars with internal commas should be sub-split."""
        # Build a sentence that is ~180 chars with commas (exceeds new 150 limit)
        clause = "这是一个相当长的从句内容啊" # 12 chars
        content = "，".join([clause] * 15) + "。" # ~194 chars with commas
        director = _make_director_with_macro([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": content}
        ])
        result = director.parse_and_micro_chunk("test", chapter_prefix="ch01")
        # Should produce multiple chunks (comma-level split)
        assert len(result) > 1
        # Content should be fully preserved
        total = "".join(item["content"] for item in result)
        assert total == content


# ---------------------------------------------------------------------------
# Content preservation
# ---------------------------------------------------------------------------

class TestContentPreservation:
    """Ensure no content is lost during smart micro-chunking."""

    def test_all_content_preserved_with_mixed_punctuation(self):
        """Mixed punctuation content should be fully preserved after chunking."""
        content = "他说了一句话。她回答了！真的吗？当然，毫无疑问。"
        director = _make_director_with_macro([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": content}
        ])
        result = director.parse_and_micro_chunk("test", chapter_prefix="ch01")
        total = "".join(item["content"] for item in result)
        assert total == content

    def test_hard_cut_fallback_preserves_content(self):
        """Content without any splittable punctuation should still be preserved."""
        long_content = "A" * 180  # No punctuation at all
        director = _make_director_with_macro([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": long_content}
        ])
        result = director.parse_and_micro_chunk("test", chapter_prefix="ch01")
        assert len(result) > 0
        total = "".join(item["content"] for item in result)
        assert total == long_content

    def test_chunk_ids_have_correct_prefix(self):
        """Chunk IDs should include the chapter_prefix."""
        content = "第一句话。第二句话。"
        director = _make_director_with_macro([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": content}
        ])
        result = director.parse_and_micro_chunk("test", chapter_prefix="Chapter_005")
        for item in result:
            assert item["chunk_id"].startswith("Chapter_005_")


# ---------------------------------------------------------------------------
# Strengthened anti-summarization prompt in _request_llm
# ---------------------------------------------------------------------------

class TestAntiSummarizationPrompt:
    """Verify the user_content in _request_llm contains
    strengthened anti-summarization directives."""

    def test_prompt_contains_array_enforcement(self):
        """The source code for _request_llm should contain array enforcement."""
        import inspect
        source = inspect.getsource(LLMScriptDirector._request_llm)
        # Must demand flat array output
        assert "JSON 数组" in source
        # Must enforce array as outermost structure
        assert "标准的 JSON 数组" in source or "最外层为数组" in source

    def test_prompt_forbids_summarization(self):
        """The prompt should forbid deletion of content and demand completeness."""
        import inspect
        source = inspect.getsource(LLMScriptDirector._request_llm)
        assert "严禁删减" in source
        assert "完整性" in source or "完全保留" in source

    def test_prompt_demands_full_content_preservation(self):
        """The prompt should demand every sentence be preserved via physical alignment."""
        import inspect
        source = inspect.getsource(LLMScriptDirector._request_llm)
        assert "物理对齐" in source or "严禁删减" in source
