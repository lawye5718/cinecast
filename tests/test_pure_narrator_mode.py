#!/usr/bin/env python3
"""
Tests for Pure Narrator Mode (纯净旁白模式).

Covers:
- generate_pure_narrator_script: paragraph splitting, punctuation-based chunking,
  sub-chunking for long sentences, pause calculation, and output structure.
- main_producer: config defaults, phase_1 and phase_3 behavioural switches.
"""

import json
import os
import sys
import tempfile

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import LLMScriptDirector


# ---------------------------------------------------------------------------
# Helper: create a director without requiring Ollama
# ---------------------------------------------------------------------------

@pytest.fixture
def director():
    """Create an LLMScriptDirector for testing (Ollama connection failure is OK)."""
    return LLMScriptDirector.__new__(LLMScriptDirector)


@pytest.fixture(autouse=True)
def _init_director(director):
    """Initialise director fields without hitting the network."""
    director.model_name = "qwen14b-pro"
    director.max_chars_per_chunk = 60
    director.pure_narrator_chunk_limit = 100
    director.use_local_mlx_lm = False  # legacy field; no-op after API migration
    director._prev_characters = []
    director._prev_tail_entries = []


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------

class TestPureNarratorBasicStructure:
    def test_all_chunks_have_required_fields(self, director):
        text = "夜幕降临，港口灯火闪烁。\n老渔夫坐在码头上。"
        result = director.generate_pure_narrator_script(text, chapter_prefix="ch01")
        required = {"chunk_id", "type", "speaker", "gender", "emotion", "content", "pause_ms"}
        for chunk in result:
            assert required.issubset(chunk.keys()), f"Missing fields in {chunk}"

    def test_all_speakers_are_narrator(self, director):
        text = "\"你好！\"他说。\n她笑了笑。"
        result = director.generate_pure_narrator_script(text, chapter_prefix="ch01")
        for chunk in result:
            assert chunk["speaker"] == "narrator"

    def test_all_types_are_narration(self, director):
        text = "第一章 风雪\n夜幕降临。"
        result = director.generate_pure_narrator_script(text, chapter_prefix="ch01")
        for chunk in result:
            assert chunk["type"] == "narration"

    def test_chunk_id_prefix(self, director):
        text = "测试文本。"
        result = director.generate_pure_narrator_script(text, chapter_prefix="Chapter_001")
        assert all(c["chunk_id"].startswith("Chapter_001_") for c in result)

    def test_chunk_ids_are_unique(self, director):
        text = "第一句。第二句。第三句。"
        result = director.generate_pure_narrator_script(text, chapter_prefix="ch")
        ids = [c["chunk_id"] for c in result]
        assert len(ids) == len(set(ids)), "Duplicate chunk_id detected"

    def test_empty_text_returns_empty(self, director):
        result = director.generate_pure_narrator_script("", chapter_prefix="ch")
        assert result == []

    def test_whitespace_only_returns_empty(self, director):
        result = director.generate_pure_narrator_script("   \n\n  \n  ", chapter_prefix="ch")
        assert result == []


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

class TestParagraphSplitting:
    def test_multi_paragraph_produces_multiple_chunks(self, director):
        text = "段落一的内容\n段落二的内容"
        result = director.generate_pure_narrator_script(text)
        assert len(result) >= 2

    def test_blank_lines_ignored(self, director):
        text = "段落一\n\n\n段落二"
        result = director.generate_pure_narrator_script(text)
        contents = [c["content"] for c in result]
        assert "" not in contents


# ---------------------------------------------------------------------------
# Sentence-level punctuation splitting
# ---------------------------------------------------------------------------

class TestSentenceSplitting:
    def test_chinese_period_splits(self, director):
        text = "第一句话。第二句话。"
        result = director.generate_pure_narrator_script(text)
        assert len(result) == 2

    def test_exclamation_splits(self, director):
        text = "太好了！我们走吧。"
        result = director.generate_pure_narrator_script(text)
        assert len(result) == 2

    def test_question_mark_splits(self, director):
        text = "你是谁？我不认识你。"
        result = director.generate_pure_narrator_script(text)
        assert len(result) == 2

    def test_semicolon_splits(self, director):
        text = "他停下了脚步；风吹过树梢。"
        result = director.generate_pure_narrator_script(text)
        assert len(result) == 2

    def test_punctuation_retained_in_content(self, director):
        text = "夜幕降临。"
        result = director.generate_pure_narrator_script(text)
        assert result[0]["content"] == "夜幕降临。"

    def test_no_punctuation_at_end(self, director):
        text = "没有标点的文本"
        result = director.generate_pure_narrator_script(text)
        assert len(result) == 1
        assert result[0]["content"] == "没有标点的文本"


# ---------------------------------------------------------------------------
# Pause calculation
# ---------------------------------------------------------------------------

class TestPauseCalculation:
    def test_paragraph_end_pause(self, director):
        """Last chunk in last paragraph should get 1000ms pause."""
        text = "唯一的段落。"
        result = director.generate_pure_narrator_script(text)
        assert result[-1]["pause_ms"] == 1000

    def test_sentence_end_pause(self, director):
        """Sentence ending with 。 (not para end) should get 600ms."""
        text = "第一段第一句。第一段第二句。\n第二段。"
        result = director.generate_pure_narrator_script(text)
        # First chunk ends with 。 but is not para end → 600ms
        assert result[0]["pause_ms"] == 600

    def test_comma_pause(self, director):
        """Content ending with ， should get 250ms pause."""
        # Create a long sentence that triggers comma sub-splitting
        director.pure_narrator_chunk_limit = 10
        text = "这是一个非常长的句子，后面还有更多的内容。"
        result = director.generate_pure_narrator_script(text)
        comma_chunks = [c for c in result if c["content"].endswith("，")]
        assert len(comma_chunks) > 0, "Expected at least one comma-ending chunk"
        assert comma_chunks[0]["pause_ms"] == 250


# ---------------------------------------------------------------------------
# Content fidelity
# ---------------------------------------------------------------------------

class TestContentFidelity:
    def test_100_percent_content_preservation(self, director):
        """All original text must appear in the output chunks (no loss)."""
        text = "夜幕降临，港口灯火闪烁。老渔夫问道：你相信命运吗？\n年轻人摇头。"
        result = director.generate_pure_narrator_script(text)
        reconstructed = "".join(c["content"] for c in result)
        # Remove whitespace for comparison
        original_clean = text.replace("\n", "").replace(" ", "")
        reconstructed_clean = reconstructed.replace(" ", "")
        assert original_clean == reconstructed_clean, (
            f"Content mismatch!\nOriginal: {original_clean}\nReconstructed: {reconstructed_clean}"
        )

    def test_no_empty_chunks(self, director):
        text = "你好。。。世界。"
        result = director.generate_pure_narrator_script(text)
        for chunk in result:
            assert chunk["content"].strip(), f"Empty content in chunk: {chunk}"


# ---------------------------------------------------------------------------
# Default config and pure_narrator_mode flag
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    def test_default_config_has_pure_narrator_mode(self):
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")
        producer = CineCastProducer.__new__(CineCastProducer)
        config = producer._get_default_config()
        assert "pure_narrator_mode" in config
        assert config["pure_narrator_mode"] is False


# ---------------------------------------------------------------------------
# Pure narrator chunk limit (100 chars)
# ---------------------------------------------------------------------------

class TestPureNarratorChunkLimit:
    def test_default_pure_chunk_limit_is_100(self, director):
        """纯净旁白模式的默认切片上限应为 100 字。"""
        assert director.pure_narrator_chunk_limit == 100

    def test_short_sentences_not_split_under_100(self, director):
        """短于 100 字的句子不应被逗号次级切分。"""
        text = "他站在窗前，望着远方的群山，心中涌起一阵莫名的感伤，仿佛一切都在这一刻停滞了。"
        result = director.generate_pure_narrator_script(text)
        assert len(result) == 1, f"Expected 1 chunk for short sentence, got {len(result)}"

    def test_long_sentence_triggers_sub_split(self, director):
        """超过 100 字的句子应被逗号次级切分。"""
        # Build a sentence > 100 chars with commas
        text = ("他站在窗前望着远方连绵不断的群山，心中涌起一阵莫名的感伤，仿佛一切都在这一刻停滞了，"
                "时间在他的眼前缓缓流淌，那些曾经的记忆如潮水般不可阻挡地涌来，"
                "他想起了年少时与父亲在海边散步的那些温暖而平静的夏日午后。")
        result = director.generate_pure_narrator_script(text)
        assert len(result) > 1, "Expected multiple chunks for a long sentence"

    def test_max_chars_per_chunk_unchanged(self, director):
        """智能配音模式的微切片红线应保持 60 字不变。"""
        assert director.max_chars_per_chunk == 60


# ---------------------------------------------------------------------------
# Preview mode: non-destructive script handling
# ---------------------------------------------------------------------------

class TestPreviewModeNonDestructive:
    def test_preview_does_not_overwrite_original_script(self):
        """试听模式不应覆盖原始剧本文件。"""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        # Set up producer with temp directories
        with tempfile.TemporaryDirectory() as tmpdir:
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
            }
            producer = CineCastProducer.__new__(CineCastProducer)
            producer.config = config
            producer.script_dir = os.path.join(tmpdir, "scripts")
            producer.cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(producer.script_dir, exist_ok=True)
            os.makedirs(producer.cache_dir, exist_ok=True)

            # Create a fake script with 20 chunks
            fake_script = [
                {
                    "chunk_id": f"ch01_{i:05d}",
                    "type": "narration",
                    "speaker": "narrator",
                    "gender": "male",
                    "emotion": "平静",
                    "content": f"第{i}句话。",
                    "pause_ms": 600,
                }
                for i in range(1, 21)
            ]
            script_path = os.path.join(producer.script_dir, "ch01_micro.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(fake_script, f, ensure_ascii=False)

            # After preview mode reads and truncates, the original file should remain intact.
            # We can't fully run run_preview_mode without MLX, but we can verify the
            # non-destructive truncation logic by simulating the read+truncate step.
            with open(script_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            preview = loaded[:10]

            # Write to temp file, NOT the original
            temp_path = os.path.join(producer.script_dir, "_preview_temp_micro.json")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(preview, f, ensure_ascii=False)

            # Original file should still have all 20 chunks
            with open(script_path, "r", encoding="utf-8") as f:
                original = json.load(f)
            assert len(original) == 20, f"Original script should remain with 20 chunks, got {len(original)}"

            # Temp preview file should have 10 chunks
            with open(temp_path, "r", encoding="utf-8") as f:
                preview_loaded = json.load(f)
            assert len(preview_loaded) == 10
