"""Бэкенды LLM. Интерфейс один: список сообщений → строка с ответом модели."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol

from .schema import DocResult


class LLM(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class LLMUnavailable(RuntimeError):
    """Модель недоступна (нет ключа, сеть, лимиты). Это не ошибка данных, ретраить бессмысленно."""


class GeminiLLM:
    """Google Gemini через REST (generateContent). Ключ — только из переменной окружения GOOGLE_API_KEY.
    Схема передаётся в responseJsonSchema: Gemini ограничивает генерацию ею, но мы всё равно валидируем сами."""

    def __init__(self, model: str | None = None, timeout: float = 90.0):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.key = os.environ.get("GOOGLE_API_KEY")
        self.timeout = timeout

    def complete(self, messages: list[dict]) -> str:
        if not self.key:
            raise LLMUnavailable("GOOGLE_API_KEY is not set")
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseJsonSchema": DocResult.model_json_schema()},
        }).encode()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.key}
        last: Exception | None = None
        for attempt in range(3):  # лимиты/перегрузка (429/5xx) — подождать и повторить; прочее — сразу ошибка
            try:
                with urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as e:
                last = e
                if e.code not in (429, 500, 503):
                    detail = e.read()[:300].decode("utf-8", "replace")
                    raise LLMUnavailable(f"Gemini {self.model}: HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
                last = e
            time.sleep(2 ** attempt)
        raise LLMUnavailable(f"Gemini {self.model} failed after retries: {last}") from last
