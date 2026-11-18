"""CLI: python -m docparse <файл|-> [--model NAME]

JSON по схеме — всегда в stdout (даже при провале, тогда это пустой результат с предупреждением).
Коды выхода: 0 — ответ прошёл валидацию; 3 — валидного ответа модели получить не удалось; 4 — модель недоступна (нет GOOGLE_API_KEY, сеть, лимиты).
"""
from __future__ import annotations

import argparse
import sys

from .llm import GeminiLLM, LLMUnavailable
from .pipeline import extract


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="docparse")
    ap.add_argument("path", help="текстовый файл документа или '-' для stdin")
    ap.add_argument("--model", help="модель Gemini (по умолчанию $GEMINI_MODEL или gemini-3.8-flash)")
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    text = sys.stdin.read() if args.path == "-" else open(args.path, encoding="utf-8").read()
    try:
        out = extract(text, GeminiLLM(args.model), max_attempts=args.max_attempts)
    except LLMUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 4
    print(out.result.model_dump_json(indent=2))
    if not out.ok:
        print(f"validation failed after {out.attempts} attempts:", file=sys.stderr)
        for i, e in enumerate(out.errors, 1):
            print(f"  attempt {i}: {e}", file=sys.stderr)
        return 3
    if out.attempts > 1:
        print(f"note: valid answer received on attempt {out.attempts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
