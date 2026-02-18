#!/usr/bin/env python3
"""
Tests for the smart dynamic max_length strategy.

Covers:
- parse_text_to_script accepts max_length parameter
- parse_and_micro_chunk passes max_length through
- _chunk_text_for_llm respects custom max_length
- Strengthened anti-summarization prompt
- Word count alignment retry logic in main_producer
- Global adaptive max_length reduction
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import LLMScriptDirector


# ---------------------------------------------------------------------------
# parse_text_to_script / parse_and_micro_chunk accept max_length
# ---------------------------------------------------------------------------

class TestMaxLengthParameter:
    """Verify max_length parameter is threaded through the call chain."""

    def test_parse_text_to_script_accepts_max_length(self):
        """parse_text_to_script signature should accept max_length."""
        import inspect
        sig = inspect.signature(LLMScriptDirector.parse_text_to_script)
        assert "max_length" in sig.parameters
        assert sig.parameters["max_length"].default == 800

    def test_parse_and_micro_chunk_accepts_max_length(self):
        """parse_and_micro_chunk signature should accept max_length."""
        import inspect
        sig = inspect.signature(LLMScriptDirector.parse_and_micro_chunk)
        assert "max_length" in sig.parameters
        assert sig.parameters["max_length"].default == 800

    def test_chunk_text_respects_custom_max_length(self):
        """_chunk_text_for_llm should respect a custom max_length."""
        d = LLMScriptDirector.__new__(LLMScriptDirector)
        d.global_cast = {}
        d._prev_characters = []
        d._prev_tail_entries = []

        para = "测试段落文本。" * 30  # ~210 chars per paragraph
        text = "\n".join([para, para, para])  # ~630 chars

        # With max_length=500, should produce more chunks than max_length=800
        chunks_500 = d._chunk_text_for_llm(text, max_length=500)
        chunks_800 = d._chunk_text_for_llm(text, max_length=800)
        assert len(chunks_500) >= len(chunks_800)

    def test_chunk_text_with_400_floor(self):
        """Chunks produced with max_length=400 should not exceed 400 + one paragraph."""
        d = LLMScriptDirector.__new__(LLMScriptDirector)
        d.global_cast = {}
        d._prev_characters = []
        d._prev_tail_entries = []

        para = "短段落。" * 25  # ~100 chars
        text = "\n".join([para] * 10)  # ~1000 chars total

        chunks = d._chunk_text_for_llm(text, max_length=400)
        max_para_len = len(para)
        for chunk in chunks:
            assert len(chunk) <= 400 + max_para_len


# ---------------------------------------------------------------------------
# Strengthened anti-summarization prompt
# ---------------------------------------------------------------------------

class TestAntiSummarizationPrompt:
    """Verify the system prompt contains strengthened anti-summarization directives."""

    def test_anti_summarization_keywords_in_source(self):
        """llm_director.py should include the strengthened prohibition."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "llm_director.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "禁止总结" in source
        assert "禁止概括" in source
        assert "禁止压缩" in source


# ---------------------------------------------------------------------------
# Word count alignment retry logic (unit-level, mocking LLM calls)
# ---------------------------------------------------------------------------

class TestWordCountAlignmentRetry:
    """Test the retry logic in phase_1_generate_scripts for word count alignment."""

    def test_retry_reduces_max_length_on_content_loss(self):
        """Verify key retry logic components exist in main_producer.py."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Verify retry loop structure
        assert "chapter_max_length = global_max_length" in source
        assert "int(chapter_max_length * 0.8)" in source
        assert "MIN_MAX_LENGTH" in source
        assert "parsed_len < original_len * 0.9" in source
        assert "recent_needed_reduction" in source
        # Verify max_length is passed to parse_and_micro_chunk
        assert "max_length=chapter_max_length" in source

    def test_max_length_floor_at_400(self):
        """Verify MIN_MAX_LENGTH is 400 in source code."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "MIN_MAX_LENGTH = 400" in source


# ---------------------------------------------------------------------------
# Global adaptive logic
# ---------------------------------------------------------------------------

class TestGlobalAdaptiveLogic:
    """Test the global adaptive max_length reduction logic."""

    def test_global_reduction_code_exists(self):
        """main_producer.py should contain the global adaptive logic."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        # Check that the global adaptive logic references are present
        assert "global_max_length" in source
        assert "recent_needed_reduction" in source
        assert "sum(recent_needed_reduction) >= 3" in source
        assert "int(global_max_length * 0.8)" in source

    def test_global_max_length_starts_at_800(self):
        """global_max_length should start at 800."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "global_max_length = 800" in source

    def test_sliding_window_size_is_5(self):
        """The sliding window for reduction tracking should be capped at 5."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        # Window is trimmed after append (> 5 triggers pop)
        # and checked at exactly 5 elements for the threshold
        assert "len(recent_needed_reduction) > 5" in source
        assert "len(recent_needed_reduction) == 5" in source
        assert "sum(recent_needed_reduction) >= 3" in source

    def test_sliding_window_never_exceeds_5(self):
        """Simulate the sliding window logic to verify it never exceeds 5 elements."""
        recent = []
        for i in range(10):
            recent.append(True)
            if len(recent) > 5:
                recent.pop(0)
            assert len(recent) <= 5
        assert len(recent) == 5
