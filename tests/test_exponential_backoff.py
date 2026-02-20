#!/usr/bin/env python3
"""
Tests for exponential backoff retry and throttle logic in LLMScriptDirector.

Covers:
- Exponential backoff on 429 rate-limit responses
- Retry on 5xx server errors
- Fatal error on other 4xx responses (no retry)
- Retry on network-level exceptions (RequestException)
- Max retries exhaustion raises RuntimeError
- Throttle sleep between chunks in parse_text_to_script
"""

import json
import os
import sys
import time
from unittest.mock import patch, MagicMock

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_director import LLMScriptDirector


def _make_director():
    """Create a LLMScriptDirector with dummy credentials."""
    d = LLMScriptDirector.__new__(LLMScriptDirector)
    d.api_url = "https://fake.api/v1/chat/completions"
    d.api_key = "fake-key"
    d.model_name = "glm-4-flash"
    d.global_cast = {}
    d.cast_profiles = {}
    d._local_session_cast = {}
    d._prev_characters = []
    d._prev_tail_entries = []
    d.VOICE_ARCHETYPES = {}
    return d


def _ok_response(content_json):
    """Build a mock 200 response returning *content_json* as the LLM output."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(content_json, ensure_ascii=False)}}]
    }
    return resp


def _error_response(status_code, text="error"):
    """Build a mock HTTP error response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=resp
    )
    return resp


class TestExponentialBackoff:
    """Tests for the retry / backoff logic inside _request_llm."""

    @patch("modules.llm_director.time.sleep", return_value=None)
    @patch("modules.llm_director.requests.post")
    def test_429_retries_with_exponential_backoff(self, mock_post, mock_sleep):
        """429 should trigger exponential backoff and eventually succeed."""
        ok = _ok_response([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "测试文本。"}
        ])
        # First two calls: 429, third call: success
        mock_post.side_effect = [
            _error_response(429),
            _error_response(429),
            ok,
        ]

        director = _make_director()
        result = director._request_llm("测试文本。")

        assert len(result) == 1
        assert result[0]["content"] == "测试文本。"
        # Sleep should have been called 3 times: two 429 retries (3, 6) + one post-request cooldown (1.5)
        assert mock_sleep.call_count == 3
        mock_sleep.assert_any_call(3)   # base_wait_time * 2^0
        mock_sleep.assert_any_call(6)   # base_wait_time * 2^1
        mock_sleep.assert_any_call(1.5) # post-request cooldown (input < 8K)

    @patch("modules.llm_director.time.sleep", return_value=None)
    @patch("modules.llm_director.requests.post")
    def test_5xx_retries(self, mock_post, mock_sleep):
        """5xx errors should retry with fixed 5s wait."""
        ok = _ok_response([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "成功。"}
        ])
        mock_post.side_effect = [
            _error_response(502),
            ok,
        ]

        director = _make_director()
        result = director._request_llm("成功。")

        assert len(result) == 1
        # Sleep called twice: once for 5xx retry (5s) + once for post-request cooldown (1.5s)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(5)
        mock_sleep.assert_any_call(1.5)

    @patch("modules.llm_director.requests.post")
    def test_4xx_fatal_no_retry(self, mock_post):
        """Non-429/5xx HTTP errors should raise immediately without retry."""
        mock_post.return_value = _error_response(401, "Unauthorized")

        director = _make_director()
        with pytest.raises(RuntimeError, match="致命请求失败"):
            director._request_llm("测试。")

        # Only one call – no retry
        assert mock_post.call_count == 1

    @patch("modules.llm_director.time.sleep", return_value=None)
    @patch("modules.llm_director.requests.post")
    def test_network_exception_retries(self, mock_post, mock_sleep):
        """Network-level exceptions should retry with 5s wait."""
        ok = _ok_response([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "网络恢复。"}
        ])
        mock_post.side_effect = [
            requests.exceptions.ConnectionError("DNS failure"),
            ok,
        ]

        director = _make_director()
        result = director._request_llm("网络恢复。")

        assert len(result) == 1
        # Sleep called twice: once for network retry (5s) + once for post-request cooldown (1.5s)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(5)
        mock_sleep.assert_any_call(1.5)

    @patch("modules.llm_director.time.sleep", return_value=None)
    @patch("modules.llm_director.requests.post")
    def test_max_retries_exhausted(self, mock_post, mock_sleep):
        """After max_retries consecutive 429s, RuntimeError should be raised."""
        mock_post.side_effect = [_error_response(429)] * 5

        director = _make_director()
        with pytest.raises(RuntimeError, match="超过最大重试次数"):
            director._request_llm("永远失败。")

        assert mock_post.call_count == 5

    @patch("modules.llm_director.time.sleep", return_value=None)
    @patch("modules.llm_director.requests.post")
    def test_timeout_is_300(self, mock_post, mock_sleep):
        """requests.post should be called with timeout=300."""
        mock_post.return_value = _ok_response([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "超时测试。"}
        ])

        director = _make_director()
        director._request_llm("超时测试。")

        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 300


class TestChunkThrottle:
    """Tests for the throttle behavior between chunks in parse_text_to_script."""

    @patch("modules.llm_director.time.sleep", return_value=None)
    def test_no_throttle_between_chunks(self, mock_sleep):
        """parse_text_to_script should NOT sleep between chunks (cloud API rate limiting handled by 429 retry)."""
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


class TestPostRequestCooldown:
    """Tests for the post-request cooldown logic in _request_llm."""

    @patch("modules.llm_director.time.sleep", return_value=None)
    @patch("modules.llm_director.requests.post")
    def test_small_input_gets_short_cooldown(self, mock_post, mock_sleep):
        """Input < 8000 chars should trigger a 1.5s cooldown."""
        mock_post.return_value = _ok_response([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": "短文本。"}
        ])

        director = _make_director()
        director._request_llm("短文本。")

        mock_sleep.assert_called_once_with(1.5)

    @patch("modules.llm_director.time.sleep", return_value=None)
    @patch("modules.llm_director.requests.post")
    def test_large_input_gets_long_cooldown(self, mock_post, mock_sleep):
        """Input > 8000 chars should trigger a 30s cooldown."""
        large_text = "这" * 9000  # > 8000 chars
        mock_post.return_value = _ok_response([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": large_text}
        ])

        director = _make_director()
        director._request_llm(large_text)

        mock_sleep.assert_called_once_with(30)

    @patch("modules.llm_director.time.sleep", return_value=None)
    @patch("modules.llm_director.requests.post")
    def test_boundary_input_gets_short_cooldown(self, mock_post, mock_sleep):
        """Input exactly 8000 chars should get the short 1.5s cooldown."""
        boundary_text = "这" * 8000  # exactly 8000 chars
        mock_post.return_value = _ok_response([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": boundary_text}
        ])

        director = _make_director()
        director._request_llm(boundary_text)

        mock_sleep.assert_called_once_with(1.5)

    @patch("modules.llm_director.time.sleep", return_value=None)
    @patch("modules.llm_director.requests.post")
    def test_context_included_in_length_calculation(self, mock_post, mock_sleep):
        """Context length should be included in input_len for cooldown threshold."""
        text = "这" * 4000        # 4000 chars
        context = "前文" * 2500   # 5000 chars -> total = 9000 > 8000
        mock_post.return_value = _ok_response([
            {"type": "narration", "speaker": "narrator", "gender": "male",
             "emotion": "平静", "content": text}
        ])

        director = _make_director()
        director._request_llm(text, context=context)

        mock_sleep.assert_called_once_with(30)


class TestChunkDefaultSize:
    """Tests for the updated _chunk_text_for_llm default size."""

    def test_default_max_length_is_50000(self):
        """_chunk_text_for_llm should have default max_length of 50000."""
        import inspect
        director = _make_director()
        sig = inspect.signature(director._chunk_text_for_llm)
        assert sig.parameters["max_length"].default == 50000

    def test_short_text_not_chunked(self):
        """Text shorter than 50000 chars should be returned as a single chunk."""
        director = _make_director()
        text = "这是一段测试文本。\n第二段。\n第三段。"
        chunks = director._chunk_text_for_llm(text)
        assert len(chunks) == 1
