"""Lightweight OpenAI-compatible client backed by ``requests``.

This module provides a drop-in replacement for ``openai.OpenAI`` that talks
directly to a local Ollama (or any OpenAI-compatible) server using the
``requests`` library.  This eliminates the ``openai`` package dependency and
ensures the project always uses a local LLM, consistent with the approach
used in the main CineCast modules on the master branch.
"""

import requests


class _Usage:
    """Minimal token-usage object."""
    def __init__(self, data: dict):
        self.prompt_tokens = data.get("prompt_tokens", 0)
        self.completion_tokens = data.get("completion_tokens", 0)
        self.total_tokens = data.get("total_tokens", 0)


class _Message:
    def __init__(self, data: dict):
        self.content = data.get("content", "")
        self.role = data.get("role", "assistant")


class _Choice:
    def __init__(self, data: dict):
        self.message = _Message(data.get("message", {}))
        self.finish_reason = data.get("finish_reason", "stop")
        self.index = data.get("index", 0)


class _ChatCompletion:
    def __init__(self, data: dict):
        self.choices = [_Choice(c) for c in data.get("choices", [])]
        usage_data = data.get("usage")
        self.usage = _Usage(usage_data) if usage_data else None


class _Completions:
    def __init__(self, base_url: str, api_key: str):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key

    def create(self, *, model, messages, temperature=1.0, top_p=1.0,
               presence_penalty=0.0, max_tokens=4096, extra_body=None):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            resp = requests.post(self._url, json=payload, headers=headers,
                                 timeout=300)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"无法连接到本地大模型服务 ({self._url})。"
                " 请确认 Ollama 已启动并正在运行。"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"本地大模型服务响应超时 ({self._url})。"
                " 请检查模型是否正常运行。"
            )
        return _ChatCompletion(resp.json())


class _Chat:
    def __init__(self, base_url: str, api_key: str):
        self.completions = _Completions(base_url, api_key)


class OpenAI:
    """Drop-in replacement for ``openai.OpenAI`` using ``requests``."""
    def __init__(self, base_url="http://localhost:11434/v1", api_key="local"):
        self.chat = _Chat(base_url, api_key)
