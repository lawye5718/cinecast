#!/usr/bin/env python3
"""
Tests for exponential backoff retry logic in LLMScriptDirector.

Covers:
- Exponential backoff on exceptions
- Max retries exhaustion raises RuntimeError
- Throttle sleep between chunks in parse_text_to_script
- Chunk default size
"""

import json
import os
import sys
import time
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import LLMScriptDirector


def _make_director():
    """Create a LLMScriptDirector with dummy credentials."""
    d = LLMScriptDirector.__new__(LLMScriptDirector)
    d.api_key = "fake-key"
    d.model_name = "qwen-flash"
    d.global_cast = {}
    d.cast_profiles = {}
    d._local_session_cast = {}
    d._prev_characters = []
    d._prev_tail_entries = []
    d.VOICE_ARCHETYPES = {}
    d.client = MagicMock()
    return d


def _ok_streaming_response(content_json):
    """Build a mock OpenAI streaming response returning *content_json* as the LLM output."""
    content_str = json.dumps(content_json, ensure_ascii=False)
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta = MagicMock()
    mock_chunk.choices[0].delta.content = content_str
    mock_chunk.choices[0].delta.reasoning_content = None
    return iter([mock_chunk])


class TestExponentialBackoff:
    """Tests for the retry / backoff logic inside _request_llm."""

    @patch("modules.llm_director.time.sleep", return_value=None)
    def test_exception_retries_with_exponential_backoff(self, mock_sleep):
        """Exceptions should trigger exponential backoff and eventually succeed."""
        ok = _ok_streaming_response([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "测试文本。"}
        ])
        director = _make_director()
        # First two calls: exception, third call: success
        director.client.chat.completions.create.side_effect = [
            Exception("rate limit"),
            Exception("rate limit"),
            ok,
        ]

        result = director._request_llm("测试文本。")

        assert len(result) == 1
        assert result[0]["content"] == "测试文本。"
        # Sleep called for two retries: 5*2^0=5, 5*2^1=10
        assert mock_sleep.call_count == 2
        sleep_args = [c[0][0] for c in mock_sleep.call_args_list]
        assert sleep_args[0] == 5
        assert sleep_args[1] == 10

    @patch("modules.llm_director.time.sleep", return_value=None)
    def test_max_retries_exhausted(self, mock_sleep):
        """After max_retries (3) consecutive failures, RuntimeError should be raised."""
        director = _make_director()
        director.client.chat.completions.create.side_effect = Exception("always fails")

        with pytest.raises(RuntimeError, match="超过最大重试次数"):
            director._request_llm("永远失败。")

        assert director.client.chat.completions.create.call_count == 3

    def test_successful_request_no_sleep(self):
        """A successful request should not trigger any sleep."""
        director = _make_director()
        director.client.chat.completions.create.return_value = _ok_streaming_response([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "成功。"}
        ])

        with patch("modules.llm_director.time.sleep") as mock_sleep:
            result = director._request_llm("成功。")

        assert len(result) == 1
        assert mock_sleep.call_count == 0

    def test_stream_parameter_used(self):
        """OpenAI SDK should be called with stream=True."""
        director = _make_director()
        director.client.chat.completions.create.return_value = _ok_streaming_response([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "流式测试。"}
        ])

        director._request_llm("流式测试。")

        _, kwargs = director.client.chat.completions.create.call_args
        assert kwargs["stream"] is True


class TestChunkThrottle:
    """Tests for the throttle behavior between chunks in parse_text_to_script."""

    @patch("modules.llm_director.time.sleep", return_value=None)
    def test_no_throttle_between_chunks(self, mock_sleep):
        """parse_text_to_script should NOT sleep between chunks (cloud API rate limiting handled by retry)."""
        director = _make_director()
        # Provide a cast_db_path attribute expected by _update_cast_db
        director.cast_db_path = None

        chunk_result = [
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "片段。"}
        ]

        # Make _chunk_text_for_llm return 3 chunks
        with patch.object(director, "_chunk_text_for_llm", return_value=["a", "b", "c"]), \
             patch.object(director, "_request_llm", return_value=chunk_result), \
             patch.object(director, "verify_integrity", return_value=True), \
             patch.object(director, "_update_cast_db"):
            director.parse_text_to_script("some text")

        # No throttle sleep(2) should be called between chunks
        sleep_2_calls = [c for c in mock_sleep.call_args_list if c[0] == (2,)]
        assert len(sleep_2_calls) == 0

    @patch("modules.llm_director.time.sleep", return_value=None)
    def test_no_throttle_for_single_chunk(self, mock_sleep):
        """No throttle sleep when there is only a single chunk."""
        director = _make_director()
        director.cast_db_path = None

        chunk_result = [
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "单片段。"}
        ]

        with patch.object(director, "_chunk_text_for_llm", return_value=["a"]), \
             patch.object(director, "_request_llm", return_value=chunk_result), \
             patch.object(director, "verify_integrity", return_value=True), \
             patch.object(director, "_update_cast_db"):
            director.parse_text_to_script("short text")

        sleep_2_calls = [c for c in mock_sleep.call_args_list if c[0] == (2,)]
        assert len(sleep_2_calls) == 0


class TestChunkDefaultSize:
    """Tests for the updated _chunk_text_for_llm default size."""

    def test_default_max_length_is_10000(self):
        """_chunk_text_for_llm should have default max_length of 10000."""
        import inspect
        director = _make_director()
        sig = inspect.signature(director._chunk_text_for_llm)
        assert sig.parameters["max_length"].default == 10000

    def test_short_text_not_chunked(self):
        """Text shorter than 10000 chars should be returned as a single chunk."""
        director = _make_director()
        text = "这是一段测试文本。\n第二段。\n第三段。"
        chunks = director._chunk_text_for_llm(text)
        assert len(chunks) == 1
