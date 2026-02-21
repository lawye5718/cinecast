#!/usr/bin/env python3
"""
Tests for chunk size, balanced splitting, content-filter bypass, and anti-truncation prompt.

Covers:
- Max chunk default lowered from 10000 to 8000
- Balanced splitting: chapters over 8000 chars produce roughly equal chunks
- Inappropriate content exception bypasses retries and falls back to narrator
- System prompt includes content compliance disclaimer and anti-truncation instructions
"""

import inspect
import os
import sys
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import LLMScriptDirector


# ---------------------------------------------------------------------------
# Helper to build a director without real API keys
# ---------------------------------------------------------------------------

def _make_director():
    with mock.patch.object(LLMScriptDirector, "__init__", lambda self: None):
        d = LLMScriptDirector.__new__(LLMScriptDirector)
        d.model_name = "test"
        d.client = None
        d.global_cast = {}
        d.cast_profiles = {}
        d._local_session_cast = {}
        d.max_chars_per_chunk = 80
        return d


# ---------------------------------------------------------------------------
# _chunk_text_for_llm default & balanced splitting
# ---------------------------------------------------------------------------

class TestChunkTextDefaults:
    """Verify max_length default is now 8000."""

    def test_default_max_length_is_8000(self):
        sig = inspect.signature(LLMScriptDirector._chunk_text_for_llm)
        default = sig.parameters["max_length"].default
        assert default == 8000

    def test_parse_text_to_script_default_is_8000(self):
        sig = inspect.signature(LLMScriptDirector.parse_text_to_script)
        default = sig.parameters["max_length"].default
        assert default == 8000

    def test_parse_and_micro_chunk_default_is_8000(self):
        sig = inspect.signature(LLMScriptDirector.parse_and_micro_chunk)
        default = sig.parameters["max_length"].default
        assert default == 8000


class TestBalancedSplitting:
    """Verify chunks are split into roughly equal parts."""

    def test_short_text_not_split(self):
        d = _make_director()
        text = "短文本。\n第二段。\n"
        chunks = d._chunk_text_for_llm(text, max_length=8000)
        assert len(chunks) == 1

    def test_large_text_balanced_chunks(self):
        """An ~8600-char text should produce 2 chunks of ~4300 each, not 8000+600."""
        d = _make_director()
        # Build text with paragraphs totalling > 8000 chars
        paragraphs = [f"段落{i:03d}，" + "这是一段比较长的测试文本内容" * 10 for i in range(60)]
        text = "\n".join(paragraphs) + "\n"
        assert len(text) > 8000, f"Test setup error: text only {len(text)} chars"

        chunks = d._chunk_text_for_llm(text, max_length=8000)
        assert len(chunks) >= 2

        # Key assertion: the ratio between largest and smallest should be reasonable
        lengths = [len(c) for c in chunks]
        assert max(lengths) / max(min(lengths), 1) < 3.0, (
            f"Chunks too unbalanced: {lengths}"
        )

    def test_very_large_text_multiple_balanced(self):
        """A 20000-char text should produce ~3 chunks, each around 6600."""
        d = _make_director()
        paragraphs = [f"P{i:04d} " + "测试" * 30 for i in range(300)]
        text = "\n".join(paragraphs) + "\n"
        assert len(text) > 16000

        chunks = d._chunk_text_for_llm(text, max_length=8000)
        assert len(chunks) >= 2

        lengths = [len(c) for c in chunks]
        # No chunk should be more than 3x another
        assert max(lengths) / max(min(lengths), 1) < 3.0, (
            f"Chunks too unbalanced: {lengths}"
        )

    def test_empty_text_returns_empty(self):
        d = _make_director()
        assert d._chunk_text_for_llm("", max_length=8000) == []
        assert d._chunk_text_for_llm("   \n  \n  ", max_length=8000) == []


# ---------------------------------------------------------------------------
# Content-filter bypass (inappropriate content)
# ---------------------------------------------------------------------------

class TestInappropriateContentBypass:
    """Verify that 'inappropriate content' errors skip retries entirely."""

    def test_inappropriate_content_returns_narrator_fallback(self):
        d = _make_director()
        d.client = mock.MagicMock()
        d.client.chat.completions.create.side_effect = Exception(
            "Error: inappropriate content detected by safety filter"
        )

        result = d._request_llm("一些文本内容")
        # Should return a narrator fallback without retrying
        assert len(result) == 1
        assert result[0]["type"] == "narration"
        assert result[0]["speaker"] == "narrator"
        # create should only be called once (no retry)
        assert d.client.chat.completions.create.call_count == 1

    def test_data_inspection_failed_returns_narrator_fallback(self):
        d = _make_director()
        d.client = mock.MagicMock()
        d.client.chat.completions.create.side_effect = Exception(
            "Data inspection failed: content policy violation"
        )

        result = d._request_llm("另一些文本内容")
        assert len(result) == 1
        assert result[0]["type"] == "narration"
        assert result[0]["speaker"] == "narrator"
        assert d.client.chat.completions.create.call_count == 1

    def test_normal_error_still_retries(self):
        d = _make_director()
        d.client = mock.MagicMock()
        d.client.chat.completions.create.side_effect = Exception("Connection timeout")

        with mock.patch("time.sleep"):
            with pytest.raises(RuntimeError, match="超过最大重试次数"):
                d._request_llm("文本")

        # Should have retried 3 times
        assert d.client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# System prompt enhancements
# ---------------------------------------------------------------------------

class TestSystemPromptEnhancements:
    """Verify new prompt sections are present in _request_llm source."""

    def test_content_compliance_disclaimer_present(self):
        source = inspect.getsource(LLMScriptDirector._request_llm)
        assert "内容合规免责声明" in source

    def test_anti_truncation_instruction_present(self):
        source = inspect.getsource(LLMScriptDirector._request_llm)
        assert "防截断死亡红线" in source

    def test_original_core_rules_still_present(self):
        source = inspect.getsource(LLMScriptDirector._request_llm)
        assert "物理对齐" in source
        assert "根节点约束" in source
        assert "情绪约束" in source
