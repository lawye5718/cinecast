#!/usr/bin/env python3
"""
Tests for script and rendering comprehensive monitoring.

Covers:
- Enhanced verify_integrity: 99% threshold detailed diff logging
- Enhanced verify_integrity: 90% threshold with detailed diff logging
- _log_content_diff: paragraph-level missing content analysis
- parse_and_micro_chunk: automatic narration fallback when ratio < 90%
- phase_2_render_dry_audio: enhanced exception and failure logging
"""

import logging
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import LLMScriptDirector


# ---------------------------------------------------------------------------
# Enhanced verify_integrity: 99% threshold
# ---------------------------------------------------------------------------

class TestVerifyIntegrity99Threshold:
    """Verify the 99% threshold triggers detailed diff logging."""

    @pytest.fixture
    def director(self):
        """Create a minimal LLMScriptDirector without network."""
        d = LLMScriptDirector.__new__(LLMScriptDirector)
        d.global_cast = {}
        d._prev_characters = []
        d._prev_tail_entries = []
        return d

    def test_99_threshold_logs_warning(self, director, caplog):
        """When ratio is between 90% and 99%, should log warning with details."""
        original = "a" * 100
        script = [{"content": "a" * 95}]  # 95% retention
        with caplog.at_level(logging.WARNING):
            result = director.verify_integrity(original, script)
        assert result is True
        assert "内容差异检测" in caplog.text
        assert "95" in caplog.text  # ratio should appear

    def test_99_threshold_exact_boundary_passes_silently(self, director, caplog):
        """Exactly 99% retention should not trigger the warning."""
        original = "a" * 100
        script = [{"content": "a" * 99}]  # Exactly 99%
        with caplog.at_level(logging.WARNING):
            result = director.verify_integrity(original, script)
        assert result is True
        assert "内容差异检测" not in caplog.text

    def test_100_percent_passes_silently(self, director, caplog):
        """100% retention should pass without warnings."""
        original = "a" * 100
        script = [{"content": "a" * 100}]
        with caplog.at_level(logging.WARNING):
            result = director.verify_integrity(original, script)
        assert result is True
        assert "内容差异检测" not in caplog.text

    def test_below_90_logs_detailed_error(self, director, caplog):
        """When ratio < 90%, should log detailed error with character counts."""
        original = "a" * 100
        script = [{"content": "a" * 50}]  # 50% retention
        with caplog.at_level(logging.ERROR):
            result = director.verify_integrity(original, script)
        assert result is False
        assert "详细差异" in caplog.text
        assert "缺失字数" in caplog.text


# ---------------------------------------------------------------------------
# _log_content_diff: paragraph-level analysis
# ---------------------------------------------------------------------------

class TestLogContentDiff:
    """Verify _log_content_diff detects missing paragraphs."""

    @pytest.fixture
    def director(self):
        d = LLMScriptDirector.__new__(LLMScriptDirector)
        d.global_cast = {}
        d._prev_characters = []
        d._prev_tail_entries = []
        return d

    def test_logs_missing_paragraphs(self, director, caplog):
        """Should identify and log paragraphs not found in script text."""
        original = "第一段完整的测试内容在这里。\n第二段也是完整的测试内容。\n第三段被完全忽略了不存在。"
        script_text = "第一段完整的测试内容在这里。第二段也是完整的测试内容。"
        with caplog.at_level(logging.WARNING):
            director._log_content_diff(original, script_text)
        assert "疑似缺失段落" in caplog.text
        assert "第三段被完全忽略了不存在" in caplog.text

    def test_no_log_when_all_present(self, director, caplog):
        """Should not log when all paragraphs are found in script."""
        original = "第一段内容。\n第二段内容。"
        script_text = "第一段内容。第二段内容。"
        with caplog.at_level(logging.WARNING):
            director._log_content_diff(original, script_text)
        assert "疑似缺失段落" not in caplog.text

    def test_empty_input_no_crash(self, director, caplog):
        """Should handle empty input gracefully."""
        with caplog.at_level(logging.WARNING):
            director._log_content_diff("", "")
        assert "疑似缺失段落" not in caplog.text

    def test_limits_logged_paragraphs(self, director, caplog):
        """Should limit logged missing paragraphs to 10."""
        paragraphs = [f"这是唯一的测试段落编号{i}内容" for i in range(20)]
        original = "\n".join(paragraphs)
        script_text = ""  # None of the paragraphs match
        with caplog.at_level(logging.WARNING):
            director._log_content_diff(original, script_text)
        assert "及其余" in caplog.text


# ---------------------------------------------------------------------------
# parse_and_micro_chunk: narration fallback
# ---------------------------------------------------------------------------

class TestNarrationFallback:
    """Verify automatic narration fallback when script integrity is too low."""

    @pytest.fixture
    def director(self):
        d = LLMScriptDirector.__new__(LLMScriptDirector)
        d.global_cast = {}
        d._prev_characters = []
        d._prev_tail_entries = []
        d._local_session_cast = {}
        d._cast_db_path = None
        d.max_chars_per_chunk = 60
        d.pure_narrator_chunk_limit = 100
        return d

    def test_fallback_to_narration_on_severe_loss(self, director, caplog):
        """Should fallback to narration when LLM script has <90% content."""
        original_text = "这是一段非常完整的原始文本内容。" * 20  # 280 chars

        # Simulate LLM returning very little content (< 90%)
        poor_script = [{"content": "极少", "type": "narration", "speaker": "narrator", "gender": "male"}]

        with mock.patch.object(director, 'parse_text_to_script', return_value=poor_script):
            with caplog.at_level(logging.WARNING):
                result = director.parse_and_micro_chunk(original_text, chapter_prefix="test_chapter")

        # Should have triggered narration fallback
        assert "自动切换旁白模式渲染原文" in caplog.text
        # Result should be narration-style (all chunks should be "narration" type)
        assert len(result) > 0
        for chunk in result:
            assert chunk["type"] == "narration"
            assert chunk["speaker"] == "narrator"

    def test_no_fallback_when_content_preserved(self, director):
        """Should NOT fallback when LLM script preserves content well."""
        original_text = "这是完整的原始文本。"

        good_script = [{"content": original_text, "type": "narration", "speaker": "narrator", "gender": "male"}]

        with mock.patch.object(director, 'parse_text_to_script', return_value=good_script):
            result = director.parse_and_micro_chunk(original_text, chapter_prefix="test_chapter")

        # Should have micro-chunked normally, not fallen back
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Rendering monitoring: source code verification
# ---------------------------------------------------------------------------

class TestRenderMonitoringSourceCode:
    """Verify render monitoring enhancements exist in source."""

    def test_render_failure_logging_in_source(self):
        """main_producer.py should log render failures with chunk details."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "渲染返回失败" in source
        assert "chunk_id" in source

    def test_render_exception_traceback_in_source(self):
        """main_producer.py should log traceback on render exceptions."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "异常堆栈" in source
        assert "traceback.format_exc()" in source

    def test_narration_fallback_in_source(self):
        """llm_director.py should contain narration fallback logic."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "llm_director.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "自动切换旁白模式渲染原文" in source
        assert "generate_pure_narrator_script" in source

    def test_99_threshold_in_source(self):
        """llm_director.py should contain the 0.99 threshold check."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "llm_director.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "ratio < 0.99" in source
        assert "_log_content_diff" in source
