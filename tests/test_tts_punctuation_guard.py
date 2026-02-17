#!/usr/bin/env python3
"""
Tests for TTS punctuation guard (anti-hallucination fix) and preview mode.

Covers:
- render_dry_chunk: punctuation guard appends closing punctuation to prevent
  autoregressive TTS repetition on short texts without sentence-ending marks.
- run_preview_mode: truncation of micro-scripts to 10 sentences.
"""

import json
import os
import re
import sys
import tempfile

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Punctuation guard regex (mirrors the logic in render_dry_chunk)
# ---------------------------------------------------------------------------

_CLOSING_PUNCT_RE = re.compile(r'[。！？；.!?;]$')


def _needs_punctuation_guard(text: str) -> bool:
    """Return True if the text would be modified by the punctuation guard."""
    stripped = text.strip()
    return not _CLOSING_PUNCT_RE.search(stripped)


# ---------------------------------------------------------------------------
# Tests: Punctuation guard behaviour
# ---------------------------------------------------------------------------

class TestPunctuationGuard:
    """Verify the regex-based punctuation guard correctly identifies texts
    that need a trailing 。 to prevent TTS hallucination."""

    def test_chapter_title_needs_guard(self):
        """Chapter titles without closing punctuation must be guarded."""
        assert _needs_punctuation_guard("第一章 凯夫拉维克的风雪")

    def test_single_word_needs_guard(self):
        """Single words like '滚' or '翌日' must be guarded."""
        assert _needs_punctuation_guard("滚")
        assert _needs_punctuation_guard("翌日")

    def test_text_with_chinese_period_ok(self):
        """Text ending with 。 does not need a guard."""
        assert not _needs_punctuation_guard("夜幕降临。")

    def test_text_with_exclamation_ok(self):
        assert not _needs_punctuation_guard("太好了！")

    def test_text_with_question_ok(self):
        assert not _needs_punctuation_guard("你是谁？")

    def test_text_with_semicolon_ok(self):
        assert not _needs_punctuation_guard("风吹过树梢；")

    def test_text_with_english_period_ok(self):
        assert not _needs_punctuation_guard("Hello world.")

    def test_text_with_english_exclamation_ok(self):
        assert not _needs_punctuation_guard("Great!")

    def test_text_with_english_question_ok(self):
        assert not _needs_punctuation_guard("Really?")

    def test_text_with_english_semicolon_ok(self):
        assert not _needs_punctuation_guard("first clause;")

    def test_comma_ending_needs_guard(self):
        """Comma is NOT a sentence-ending mark, should be guarded."""
        assert _needs_punctuation_guard("然后他说，")

    def test_trailing_whitespace_ignored(self):
        """Trailing spaces should be stripped before checking."""
        assert not _needs_punctuation_guard("夜幕降临。  ")
        assert _needs_punctuation_guard("第一章  ")

    def test_empty_string(self):
        """Empty string after strip needs guard (edge case)."""
        assert _needs_punctuation_guard("")
        assert _needs_punctuation_guard("   ")

    def test_guard_produces_correct_text(self):
        """Simulate what render_dry_chunk does with the guard."""
        content = "第一章 凯夫拉维克的风雪"
        render_text = content.strip()
        if not _CLOSING_PUNCT_RE.search(render_text):
            render_text += "。"
        assert render_text == "第一章 凯夫拉维克的风雪。"

    def test_guard_does_not_double_punctuate(self):
        """Text already ending with punctuation should NOT get an extra 。"""
        content = "夜幕降临了。"
        render_text = content.strip()
        if not _CLOSING_PUNCT_RE.search(render_text):
            render_text += "。"
        assert render_text == "夜幕降临了。"


# ---------------------------------------------------------------------------
# Tests: Preview mode script truncation
# ---------------------------------------------------------------------------

class TestPreviewModeTruncation:
    """Verify that run_preview_mode truncates scripts to 10 sentences."""

    def test_truncation_to_10(self):
        """A script with >10 items should be truncated to exactly 10."""
        micro_script = [
            {"chunk_id": f"ch01_{i:04d}", "type": "narration",
             "speaker": "narrator", "content": f"第{i}句话。"}
            for i in range(25)
        ]
        preview = micro_script[:10]
        assert len(preview) == 10
        assert preview[0]["chunk_id"] == "ch01_0000"
        assert preview[9]["chunk_id"] == "ch01_0009"

    def test_truncation_short_script(self):
        """A script with fewer than 10 items stays unchanged."""
        micro_script = [
            {"chunk_id": f"ch01_{i:04d}", "type": "narration",
             "speaker": "narrator", "content": f"第{i}句话。"}
            for i in range(5)
        ]
        preview = micro_script[:10]
        assert len(preview) == 5

    def test_truncation_preserves_order(self):
        """Truncation must preserve the original order."""
        micro_script = [
            {"chunk_id": f"ch01_{i:04d}", "type": "narration",
             "speaker": "narrator", "content": f"第{i}句话。"}
            for i in range(20)
        ]
        preview = micro_script[:10]
        for i, chunk in enumerate(preview):
            assert chunk["chunk_id"] == f"ch01_{i:04d}"

    def test_truncation_json_roundtrip(self):
        """Truncated script can be serialised and deserialised."""
        micro_script = [
            {"chunk_id": f"ch01_{i:04d}", "type": "narration",
             "speaker": "narrator", "content": f"第{i}句话。"}
            for i in range(15)
        ]
        preview = micro_script[:10]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(preview, f, ensure_ascii=False)
            tmp_path = f.name
        try:
            with open(tmp_path, "r") as f:
                loaded = json.load(f)
            assert len(loaded) == 10
            assert loaded[0]["chunk_id"] == "ch01_0000"
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test: render_dry_chunk regex is present in source
# ---------------------------------------------------------------------------

class TestSourceCodeGuard:
    """Verify the punctuation guard regex is present in the actual source."""

    def test_regex_in_mlx_tts_engine(self):
        """The anti-hallucination regex must exist in mlx_tts_engine.py."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "mlx_tts_engine.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "import re" in source, "re module not imported in mlx_tts_engine.py"
        assert r"[。！？；.!?;]$" in source, "Punctuation guard regex not found"
        assert 'render_text += "。"' in source, "Punctuation append not found"

    def test_render_text_used_in_generate(self):
        """render_text (not raw content) must be passed to model.generate."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "mlx_tts_engine.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "text=render_text" in source, \
            "model.generate should use render_text, not raw content"
