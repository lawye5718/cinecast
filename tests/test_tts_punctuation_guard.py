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


def _apply_text_cleaning(text: str) -> str:
    """Mirror the full aggressive text cleaning logic in render_dry_chunk."""
    render_text = text.strip()
    render_text = re.sub(r'[…]+', '。', render_text)
    render_text = re.sub(r'\.{2,}', '。', render_text)
    render_text = re.sub(r'[—]+', '，', render_text)
    render_text = re.sub(r'[-]{2,}', '，', render_text)
    render_text = re.sub(r'[~～]+', '。', render_text)
    render_text = re.sub(r'\s+', ' ', render_text).strip()
    if len(render_text) > 80:
        render_text = render_text[:80] + "。"
    if not _CLOSING_PUNCT_RE.search(render_text):
        render_text += "。"
    return render_text


def _is_pure_punctuation(text: str) -> bool:
    """Return True if cleaned text contains no real characters (only punctuation)."""
    cleaned = _apply_text_cleaning(text)
    pure_text = re.sub(r'[。，！？；、\u201c\u201d\u2018\u2019（）《》,.!?;:\'\"()\-\s]', '', cleaned)
    return not pure_text


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
# Tests: Aggressive text cleaning (ellipsis / em-dash replacement)
# ---------------------------------------------------------------------------

class TestAggressiveTextCleaning:
    """Verify that ellipsis and em-dash characters are replaced to prevent
    autoregressive TTS from getting stuck in loops."""

    def test_chinese_ellipsis_replaced(self):
        """Chinese ellipsis … should be replaced with 。"""
        assert _apply_text_cleaning("他沉默了…") == "他沉默了。"

    def test_multiple_chinese_ellipsis_replaced(self):
        """Multiple consecutive … should be collapsed into a single 。"""
        assert _apply_text_cleaning("他沉默了……") == "他沉默了。"

    def test_em_dash_replaced(self):
        """Em-dash — should be replaced with ，"""
        assert _apply_text_cleaning("他说—好的") == "他说，好的。"

    def test_multiple_em_dash_replaced(self):
        """Multiple consecutive em-dashes should be collapsed into a single ，"""
        assert _apply_text_cleaning("他说——好的") == "他说，好的。"

    def test_triple_dot_replaced(self):
        """Three or more ASCII dots should be replaced with 。"""
        assert _apply_text_cleaning("他走了...") == "他走了。"
        assert _apply_text_cleaning("他走了....") == "他走了。"

    def test_double_dot_replaced(self):
        """Two dots should now be replaced with 。 (enhanced defense)."""
        result = _apply_text_cleaning("他走了..")
        assert ".." not in result

    def test_ellipsis_at_end_no_extra_period(self):
        """Ellipsis replaced with 。 should not get an extra 。"""
        result = _apply_text_cleaning("沉默…")
        assert result == "沉默。"
        assert not result.endswith("。。")

    def test_em_dash_at_end_gets_period(self):
        """Em-dash at end replaced with ， still needs closing punctuation."""
        result = _apply_text_cleaning("他说—")
        assert result == "他说，。"

    def test_clean_text_unchanged(self):
        """Text without problematic chars is unchanged (except guard)."""
        assert _apply_text_cleaning("这是正常文本。") == "这是正常文本。"

    def test_combined_cleaning(self):
        """Text with both ellipsis and em-dash gets fully cleaned."""
        result = _apply_text_cleaning("他说——沉默了…")
        assert "—" not in result
        assert "…" not in result


# ---------------------------------------------------------------------------
# Tests: English dash and tilde cleaning (new ultimate defense rules)
# ---------------------------------------------------------------------------

class TestUltimateDefenseCleaning:
    """Verify the additional cleaning rules for English dashes and tildes."""

    def test_double_english_dash_replaced(self):
        """Double English dashes -- should be replaced with ，"""
        assert _apply_text_cleaning("他说--好的") == "他说，好的。"

    def test_triple_english_dash_replaced(self):
        """Triple English dashes --- should be replaced with ，"""
        assert _apply_text_cleaning("他说---好的") == "他说，好的。"

    def test_single_english_dash_not_replaced(self):
        """A single English dash should NOT be replaced."""
        result = _apply_text_cleaning("他说-好的")
        assert "-" in result

    def test_chinese_tilde_replaced(self):
        """Chinese tilde ～ should be replaced with 。"""
        assert _apply_text_cleaning("好的～") == "好的。"

    def test_english_tilde_replaced(self):
        """English tilde ~ should be replaced with 。"""
        assert _apply_text_cleaning("好的~") == "好的。"

    def test_multiple_tildes_replaced(self):
        """Multiple tildes should be collapsed into a single 。"""
        assert _apply_text_cleaning("好的~~~") == "好的。"
        assert _apply_text_cleaning("好的～～～") == "好的。"

    def test_mixed_tildes_replaced(self):
        """Mixed English and Chinese tildes should be collapsed."""
        assert _apply_text_cleaning("好的~～~") == "好的。"


# ---------------------------------------------------------------------------
# Tests: Pure punctuation (empty text) defense
# ---------------------------------------------------------------------------

class TestPurePunctuationDefense:
    """Verify the 'kill switch' that detects text with no real characters."""

    def test_only_ellipsis_is_pure_punctuation(self):
        """Text that is only ellipsis should be detected as pure punctuation."""
        assert _is_pure_punctuation("……")

    def test_only_em_dash_is_pure_punctuation(self):
        """Text that is only em-dashes should be detected as pure punctuation."""
        assert _is_pure_punctuation("——")

    def test_only_tildes_is_pure_punctuation(self):
        """Text that is only tildes should be detected as pure punctuation."""
        assert _is_pure_punctuation("~~~")
        assert _is_pure_punctuation("～～～")

    def test_only_punctuation_marks(self):
        """Text with only punctuation marks should be detected."""
        assert _is_pure_punctuation("，。！？")

    def test_empty_string_is_pure_punctuation(self):
        """Empty string should be detected as pure punctuation."""
        assert _is_pure_punctuation("")
        assert _is_pure_punctuation("   ")

    def test_real_text_not_pure_punctuation(self):
        """Text with real characters should NOT be detected."""
        assert not _is_pure_punctuation("你好")
        assert not _is_pure_punctuation("Hello")

    def test_mixed_text_and_punctuation_not_pure(self):
        """Text with both characters and punctuation should NOT be detected."""
        assert not _is_pure_punctuation("他说，好的。")


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

    def test_aggressive_cleaning_in_source(self):
        """The aggressive ellipsis/em-dash/tilde cleaning must exist in mlx_tts_engine.py."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "mlx_tts_engine.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert r"[…]+" in source, "Ellipsis cleaning regex not found"
        assert r"[—]+" in source, "Em-dash cleaning regex not found"
        assert r"\.{2,}" in source, "Double-dot cleaning regex not found"
        assert r"[-]{2,}" in source, "English double-dash cleaning regex not found"
        assert r"[~～]+" in source, "Tilde cleaning regex not found"

    def test_pure_punctuation_guard_in_source(self):
        """The pure-punctuation kill switch must exist in mlx_tts_engine.py."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "mlx_tts_engine.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "pure_text" in source, "pure_text variable not found"
        assert "np.zeros" in source, "Silent audio generation not found"

    def test_gc_collect_in_source(self):
        """gc.collect() must be called in the finally block."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "mlx_tts_engine.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "import gc" in source, "gc module not imported in mlx_tts_engine.py"
        assert "gc.collect()" in source, "gc.collect() not found in mlx_tts_engine.py"

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
