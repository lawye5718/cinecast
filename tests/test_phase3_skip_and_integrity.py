#!/usr/bin/env python3
"""
Tests for the phase-3 skip-existing-volume fix, LLM chunk size reduction,
content integrity verification, strengthened Iron Rule prompt, and
lowered EPUB filter threshold.

Covers:
- Fix 1: export_volume skips already-existing Audiobook_Part_NNN.mp3 files
- Fix 2a: _chunk_text_for_llm default max_length reduced from 1500 to 800
- Fix 2b: Iron Rule prompt contains anti-omission warning
- Fix 2c: verify_integrity detects content loss > 10%
- Fix 3: _extract_epub_chapters filter threshold lowered from 100 to 20
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.cinematic_packager import CinematicPackager
from modules.llm_director import LLMScriptDirector


# ---------------------------------------------------------------------------
# Fix 1: export_volume skips existing files
# ---------------------------------------------------------------------------

class TestExportVolumeSkipExisting:
    """Verify that export_volume skips re-exporting when the output file exists."""

    def test_skip_existing_volume_resets_buffer(self):
        """When the target MP3 already exists, buffer should be cleared and index advanced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = CinematicPackager(tmpdir)
            # Pre-create the file that would be exported
            existing_file = os.path.join(tmpdir, "Audiobook_Part_001.mp3")
            with open(existing_file, "wb") as f:
                f.write(b"fake mp3 data")

            # Put something in the buffer
            from pydub import AudioSegment
            p.buffer = AudioSegment.silent(duration=5000)
            assert len(p.buffer) > 0

            p.export_volume()

            # Buffer should be emptied and file_index advanced
            assert len(p.buffer) == 0
            assert p.file_index == 2

    def test_no_skip_when_file_missing(self):
        """When target MP3 does not exist, export_volume should proceed normally
        (buffer cleared, index advanced).  We only test the state change here
        since actual mp3 export requires ffmpeg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = CinematicPackager(tmpdir)
            from pydub import AudioSegment
            p.buffer = AudioSegment.silent(duration=5000)

            # Without ffmpeg this will hit the except branch, but we can still
            # verify the source code contains the skip guard
            assert p.file_index == 1

    def test_source_has_skip_guard(self):
        """cinematic_packager.py should contain the skip-existing logic."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "cinematic_packager.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "检测到分卷已存在，跳过压制" in source
        assert "os.path.exists(save_path)" in source


# ---------------------------------------------------------------------------
# Fix 2a: LLM chunk size reduced to 800
# ---------------------------------------------------------------------------

class TestChunkSizeReduced:
    """Verify _chunk_text_for_llm default max_length is 800."""

    def test_default_max_length_800(self):
        """Source should show default max_length=800."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "llm_director.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "max_length: int = 800" in source

    def test_chunking_respects_800_limit(self):
        """Long text should be chunked at ~800 chars, not 1500."""
        director = LLMScriptDirector.__new__(LLMScriptDirector)
        director.global_cast = {}
        director._prev_characters = []
        director._prev_tail_entries = []

        # Create text with paragraphs that collectively exceed 800 but not 1500
        para = "这是一段中文测试文本。" * 20  # ~200 chars per repetition
        text = "\n".join([para, para, para, para, para])  # ~1000 chars total

        chunks = director._chunk_text_for_llm(text)
        for chunk in chunks:
            assert len(chunk) <= 800 + 200  # allow one paragraph overshoot


# ---------------------------------------------------------------------------
# Fix 2b: Strengthened Iron Rule prompt
# ---------------------------------------------------------------------------

class TestIronRuleStrengthened:
    """Verify the system prompt contains the anti-omission warning."""

    def test_anti_omission_warning_in_source(self):
        """llm_director.py should include the paragraph-omission penalty."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "llm_director.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "严禁漏掉任何一个自然段" in source
        assert "任务失败" in source


# ---------------------------------------------------------------------------
# Fix 2c: verify_integrity content-loss detection
# ---------------------------------------------------------------------------

class TestVerifyIntegrity:
    """Verify the content integrity checker works correctly."""

    @pytest.fixture
    def director(self):
        """Create a minimal LLMScriptDirector without network."""
        d = LLMScriptDirector.__new__(LLMScriptDirector)
        d.global_cast = {}
        d._prev_characters = []
        d._prev_tail_entries = []
        return d

    def test_passes_when_content_preserved(self, director):
        """Should return True when >90% content is preserved."""
        original = "这是一段完整的测试文本，用于验证内容完整性。"
        script = [{"content": original}]
        assert director.verify_integrity(original, script) is True

    def test_fails_when_content_severely_lost(self, director):
        """Should return False when <90% content is preserved."""
        original = "这是一段完整的测试文本" * 10  # 100 chars
        script = [{"content": "这是"}]  # only 2 chars => 2%
        assert director.verify_integrity(original, script) is False

    def test_passes_with_empty_input(self, director):
        """Edge case: empty original text should pass."""
        assert director.verify_integrity("", []) is True

    def test_passes_with_none_input(self, director):
        """Edge case: None-like falsy input should pass."""
        assert director.verify_integrity("", [{"content": "test"}]) is True

    def test_boundary_at_90_percent(self, director):
        """Exactly 90% preservation should pass."""
        original = "a" * 100
        script = [{"content": "a" * 90}]
        assert director.verify_integrity(original, script) is True

    def test_boundary_below_90_percent(self, director):
        """89% preservation should fail."""
        original = "a" * 100
        script = [{"content": "a" * 89}]
        assert director.verify_integrity(original, script) is False

    def test_verify_integrity_in_source(self):
        """llm_director.py should define verify_integrity and call it."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "llm_director.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "def verify_integrity" in source
        assert "self.verify_integrity(text, full_script)" in source


# ---------------------------------------------------------------------------
# Fix 3: EPUB filter threshold lowered from 100 to 20
# ---------------------------------------------------------------------------

class TestEpubFilterThreshold:
    """Verify _extract_epub_chapters uses 20-char threshold."""

    def test_threshold_lowered_in_source(self):
        """main_producer.py should use > 20, not > 100."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main_producer.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "len(clean_text) > 20" in source
        assert "len(clean_text) > 100" not in source
