"""Бэкенды LLM. Интерфейс один: список сообщений → строка с ответом модели."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from .schema import DocResult


class LLM(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class LLMUnavailable(RuntimeError):
    """Модель недоступна (не запущен Ollama и т.п.). Это не ошибка данных, ретраить бессмысленно."""


class OllamaLLM:
    """Локальная модель через Ollama HTTP API (https://github.com/ollama/ollama/blob/main/docs/api.md).
    В `format` передаём JSON-схему — Ollama ограничит генерацию ею (structured outputs)."""

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 120.0):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def complete(self, messages: list[dict]) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": DocResult.model_json_schema(),
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            raise LLMUnavailable(f"Ollama at {self.host} (model {self.model}) failed: {e}") from e


def make_llm(backend: str, model: str | None = None) -> LLM:
    if backend == "ollama":
        return OllamaLLM(model)
    from .mock_llm import HeuristicLLM, NaiveLLM  # ленивый импорт: боевой путь не зависит от моков

    if backend == "heuristic":
        return HeuristicLLM()
    if backend == "naive":
        return NaiveLLM()
    raise ValueError(f"unknown backend {backend!r} (ollama | heuristic | naive)")
