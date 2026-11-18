"""Пайплайн: промпт → LLM → разбор JSON → валидация схемы (с ретраями) → grounding → разрешение противоречий."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from .grounding import ground, reconcile
from .llm import LLM
from .prompts import build_messages, repair_message
from .schema import DocResult

MAX_CHARS = 20_000


@dataclass
class Outcome:
    result: DocResult
    ok: bool                                   # False → валидный ответ получить не удалось, вернули пустышку
    attempts: int = 0
    errors: list[str] = field(default_factory=list)  # причины отказа по каждой неудачной попытке


def parse_json(raw: str) -> dict:
    """Достаёт JSON-объект из ответа: допускает ```json-ограждение и болтовню вокруг."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in the model output")
    obj = json.loads(s[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value must be an object")
    return obj


def _format_errors(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(f"{'.'.join(map(str, x['loc'])) or '<root>'}: {x['msg']}" for x in e.errors()[:8])
    return f"{type(e).__name__}: {e}"


def extract(text: str, llm: LLM, *, max_attempts: int = 3, guard: bool = True) -> Outcome:
    """guard=False отключает grounding/reconcile — нужен только чтобы в eval показать, сколько он спасает."""
    warnings: list[str] = []
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        warnings.append(f"input_truncated: document cut to {MAX_CHARS} characters")

    messages = build_messages(text)
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        raw = llm.complete(messages)
        try:
            result = DocResult.model_validate(parse_json(raw))
        except (ValueError, ValidationError) as e:  # json.JSONDecodeError — подкласс ValueError
            msg = _format_errors(e)
            errors.append(msg)
            messages = messages + [{"role": "assistant", "content": raw}, repair_message(msg)]
            continue
        if guard:
            result = reconcile(ground(result, text), text)
        result.warnings[:0] = warnings
        return Outcome(result, True, attempt, errors)

    fallback = DocResult.empty(warnings + [
        f"validation_failed: no schema-valid answer after {max_attempts} attempts; all fields set to null. "
        f"Last error: {errors[-1][:300]}"
    ])
    return Outcome(fallback, False, max_attempts, errors)
