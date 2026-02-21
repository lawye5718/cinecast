#!/usr/bin/env python3
"""
Tests for MLX engine upgrades: tokenizer warning suppression, smart truncation,
dynamic silence, model path defaults, async I/O, and emotion prompts.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MLX_ENGINE_PATH = os.path.join(_PROJECT_ROOT, "modules", "mlx_tts_engine.py")


def _read_source():
    with open(_MLX_ENGINE_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Smart truncation helper (mirrors the engine logic)
# ---------------------------------------------------------------------------

_CLOSING_PUNCT_RE = re.compile(r'[。！？；.!?;]$')


def _smart_truncate(text: str, max_chars: int = 60) -> str:
    """Mirror the smart truncation logic in render_dry_chunk."""
    render_text = text.strip()
    render_text = re.sub(r'[…]+', '。', render_text)
    render_text = re.sub(r'\.{2,}', '。', render_text)
    render_text = re.sub(r'[—]+', '，', render_text)
    render_text = re.sub(r'[-]{2,}', '，', render_text)
    render_text = re.sub(r'[~～]+', '。', render_text)
    render_text = re.sub(r'\s+', ' ', render_text).strip()
    if len(render_text) > max_chars:
        safe_text = render_text[:max_chars]
        match = re.search(r'[。！？；.,!?;]', safe_text)
        if match:
            render_text = safe_text[:match.end()]
        else:
            render_text = safe_text + "。"
    if not _CLOSING_PUNCT_RE.search(render_text):
        render_text += "。"
    return render_text


# ---------------------------------------------------------------------------
# 1. Tokenizer regex warning suppression
# ---------------------------------------------------------------------------

class TestTokenizerWarningsSuppression:
    """Verify tokenizer warnings are suppressed at module level."""

    def test_warnings_import(self):
        source = _read_source()
        assert "import warnings" in source

    def test_filterwarnings_present(self):
        source = _read_source()
        assert 'warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")' in source

    def test_tokenizers_parallelism_env(self):
        source = _read_source()
        assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in source

    def test_fix_mistral_regex_env(self):
        source = _read_source()
        assert 'os.environ["FIX_MISTRAL_REGEX"] = "1"' in source


# ---------------------------------------------------------------------------
# 2. Smart truncation (punctuation-aware)
# ---------------------------------------------------------------------------

class TestSmartTruncation:
    """Verify smart punctuation-aware truncation replaces brute-force cut."""

    def test_source_has_smart_truncation(self):
        """Smart truncation regex must exist in source."""
        source = _read_source()
        assert "re.search" in source
        assert "safe_text" in source

    def test_truncate_at_nearest_punctuation(self):
        """Long text with punctuation should truncate at the last punct within limit."""
        # 30 chars + 。 + 29 chars = 60 chars, within limit; but let's test >60
        text = "这是一个很长很长的句子。后面还有很多很多的内容需要被截断掉因为它超过了六十个字符的限制范围所以需要智能截断"
        result = _smart_truncate(text, max_chars=60)
        # Should truncate at the 。 within the first 60 chars
        assert result.endswith("。")
        assert len(result) <= 61

    def test_truncate_no_punctuation_falls_back(self):
        """Long text without any punctuation should fall back to hard cut + 。"""
        text = "这" * 100  # 100 chars, no punctuation
        result = _smart_truncate(text, max_chars=60)
        assert result == "这" * 60 + "。"

    def test_short_text_not_truncated(self):
        """Text within limit should not be truncated."""
        text = "短文本"
        result = _smart_truncate(text)
        assert result == "短文本。"

    def test_truncate_preserves_meaningful_sentence(self):
        """Smart truncation should preserve a complete sentence."""
        text = "第一句话。第二句话也是很长很长很长很长很长很长很长很长很长很长很长很长很长很长的"
        result = _smart_truncate(text, max_chars=15)
        # Should truncate at the 。 after "第一句话"
        assert "第一句话。" == result


# ---------------------------------------------------------------------------
# 3. Dynamic silence generation
# ---------------------------------------------------------------------------

class TestDynamicSilence:
    """Verify dynamic silence duration based on punctuation type."""

    def test_source_has_dynamic_silence(self):
        source = _read_source()
        assert "duration" in source
        assert "0.6" in source
        assert "0.3" in source
        assert "0.15" in source

    def test_source_checks_original_text(self):
        source = _read_source()
        assert "original_text" in source

    def test_source_no_fixed_05_silence(self):
        """Fixed 0.5s silence should be replaced with dynamic durations."""
        source = _read_source()
        # The old "0.5" literal for silence should not appear in the
        # pure_text branch anymore
        render_func_start = source.index("def render_dry_chunk")
        render_func_body = source[render_func_start:]
        # Find the pure_text block
        pure_text_idx = render_func_body.index("if not pure_text:")
        # Check that within 20 lines after pure_text check, there's no "* 0.5"
        block = render_func_body[pure_text_idx:pure_text_idx + 800]
        assert "* 0.5)" not in block


# ---------------------------------------------------------------------------
# 4. Model path defaults (anti-degradation)
# ---------------------------------------------------------------------------

class TestModelPathDefaults:
    """Verify model paths have explicit defaults to prevent fallback to 0.6B."""

    def test_preset_default_path(self):
        source = _read_source()
        assert "./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit" in source

    def test_design_default_path(self):
        source = _read_source()
        assert "./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit" in source

    def test_clone_default_path(self):
        source = _read_source()
        assert "./models/Qwen3-TTS-12Hz-1.7B-Base-4bit" in source

    def test_model_paths_have_defaults(self):
        """All three modes should have default paths in _model_paths dict."""
        source = _read_source()
        # Each config.get call should have a default second argument
        init_start = source.index("def __init__")
        init_body = source[init_start:source.index("\n    def ", init_start + 1)]
        # Count the .get calls with defaults
        assert 'self.config.get("model_path_custom", "' in init_body
        assert 'self.config.get("model_path_design", "' in init_body
        assert 'self.config.get("model_path_base", "' in init_body


# ---------------------------------------------------------------------------
# 5. Async I/O thread pool
# ---------------------------------------------------------------------------

class TestAsyncIOThreadPool:
    """Verify ThreadPoolExecutor is used for async WAV writes."""

    def test_concurrent_futures_imported(self):
        source = _read_source()
        assert "import concurrent.futures" in source

    def test_io_executor_created(self):
        source = _read_source()
        assert "self.io_executor" in source
        assert "ThreadPoolExecutor(max_workers=1)" in source

    def test_async_write_wav_method(self):
        source = _read_source()
        assert "def _async_write_wav(self, path, data, sr)" in source

    def test_executor_submit_used(self):
        source = _read_source()
        assert "self.io_executor.submit(self._async_write_wav" in source

    def test_destroy_shuts_down_executor(self):
        source = _read_source()
        destroy_start = source.index("def destroy(self)")
        # Find the next method definition to extract the full destroy body
        next_def = source.find("\n    def ", destroy_start + 20)
        if next_def == -1:
            next_def = len(source)
        destroy_body = source[destroy_start:next_def]
        assert "io_executor" in destroy_body


# ---------------------------------------------------------------------------
# 6. Emotion prompts
# ---------------------------------------------------------------------------

class TestEmotionPrompts:
    """Verify EMOTION_PROMPTS dict and emotion-aware rendering."""

    def test_emotion_prompts_dict_exists(self):
        source = _read_source()
        assert "EMOTION_PROMPTS" in source

    def test_emotion_prompts_keys(self):
        source = _read_source()
        for emotion in ("愤怒", "悲伤", "激动", "恐惧", "平静"):
            assert f'"{emotion}"' in source

    def test_emotion_hijack_to_design_mode(self):
        """Non-calm emotion with instruct should switch to design mode."""
        source = _read_source()
        assert 'emotion != "平静"' in source
        assert '"instruct" in voice_cfg' in source
        assert 'mode = "design"' in source

    def test_emotion_modifier_combined_with_instruct(self):
        source = _read_source()
        assert "emotion_modifier" in source
        assert "base_instruct" in source


# ---------------------------------------------------------------------------
# 7. Source code structure guards
# ---------------------------------------------------------------------------

class TestSourceStructureGuards:
    """Additional guards that existing tests reference still hold."""

    def test_render_text_used_in_generate(self):
        source = _read_source()
        assert "text=render_text" in source

    def test_gc_collect_in_finally(self):
        source = _read_source()
        assert "gc.collect()" in source

    def test_pure_text_variable(self):
        source = _read_source()
        assert "pure_text" in source

    def test_np_zeros_for_silence(self):
        source = _read_source()
        assert "np.zeros" in source
