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
        assert sig.parameters["max_length"].default == 50000

    def test_parse_and_micro_chunk_accepts_max_length(self):
        """parse_and_micro_chunk signature should accept max_length."""
        import inspect
        sig = inspect.signature(LLMScriptDirector.parse_and_micro_chunk)
        assert "max_length" in sig.parameters
        assert sig.parameters["max_length"].default == 50000

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
        """llm_director.py should include the physical alignment anti-merge/delete rules."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "llm_director.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "严禁合并" in source
        assert "严禁删减" in source


# ---------------------------------------------------------------------------
# Word count alignment retry logic (unit-level, mocking LLM calls)
# ---------------------------------------------------------------------------

class TestWordCountAlignmentRetry:
    """Test that the old retry/degradation logic has been removed (GLM-4.7-Flash upgrade)."""

    def test_no_retry_degradation_logic(self):
        """Verify old retry logic components have been removed from main_producer.py."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Old retry loop artifacts should no longer be present
        assert "chapter_max_length = global_max_length" not in source
        assert "MIN_MAX_LENGTH" not in source
        assert "recent_needed_reduction" not in source
        # GLM-4.7-Flash uses max_length=15000 for TPM-safe chunking
        assert "max_length=15000" in source

    def test_no_max_length_floor_at_400(self):
        """Verify MIN_MAX_LENGTH = 400 is no longer in source code."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "MIN_MAX_LENGTH = 400" not in source


# ---------------------------------------------------------------------------
# Global adaptive logic
# ---------------------------------------------------------------------------

class TestGlobalAdaptiveLogic:
    """Test that global adaptive max_length logic has been removed (GLM-4.7-Flash upgrade)."""

    def test_no_global_adaptive_logic(self):
        """main_producer.py should no longer contain the global adaptive logic."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        # Old global adaptive logic artifacts should be removed
        assert "global_max_length" not in source
        assert "recent_needed_reduction" not in source

    def test_glm_direct_call_with_15000(self):
        """GLM-4.7-Flash should use max_length=15000 for TPM-safe chunking."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "max_length=15000" in source
