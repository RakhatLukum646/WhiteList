"""Мини-eval: python eval.py [--no-guard] [-v]

По каждому полю: верно / неверно / ГАЛЛЮЦИНАЦИЯ (в эталоне null, модель заполнила) / ПРОПУСК (в эталоне есть, модель дала null).
Галлюцинации и пропуски считаются раздельно: выдуманный БИН или сумма в бухгалтерии стоит дороже, чем пустое поле,
которое человек заполнит руками.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from datetime import date

from docparse.llm import GeminiLLM, LLMUnavailable, RecordingLLM, ReplayLLM, prompt_hash
from docparse.prompts import SYSTEM
from docparse.pipeline import extract
from docparse.schema import SCALAR_FIELDS, DocResult

ROOT = Path(__file__).parent
FIELDS = list(SCALAR_FIELDS) + ["line_items"]


def _s(v: str | None) -> str | None:
    return None if v is None else re.sub(r"\s+", " ", re.sub(r"[«»\"“”„'.,;:]", " ", v.casefold())).strip()


def _item_key(i: dict) -> tuple:
    return (_s(i["name"]), None if i["qty"] is None else round(i["qty"], 2), None if i["price"] is None else round(i["price"], 2))


def same(field: str, pred, gold) -> bool:
    if field in ("supplier_name", "doc_number"):
        return _s(pred) == _s(gold)
    if field in ("total_amount", "vat_amount"):
        return abs(pred - gold) < 0.005
    if field == "line_items":
        return Counter(map(_item_key, pred)) == Counter(map(_item_key, gold))
    return pred == gold


def classify(field: str, pred, gold) -> str:
    empty = (lambda v: v == []) if field == "line_items" else (lambda v: v is None)
    if empty(pred) and empty(gold):
        return "correct"
    if empty(gold):
        return "halluc"
    if empty(pred):
        return "omit"
    return "correct" if same(field, pred, gold) else "wrong"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--no-guard", action="store_true", help="отключить grounding/reconcile (для сравнения)")
    ap.add_argument("--docs", default=str(ROOT / "data" / "docs"))
    ap.add_argument("--gold", default=str(ROOT / "data" / "gold"))
    ap.add_argument("--replay", action="store_true", help="без ключа и сети: ответы настоящего Gemini из data/replay.json")
    ap.add_argument("--record", action="store_true", help="прогнать настоящий Gemini и записать ответы в data/replay.json")
    ap.add_argument("-v", "--verbose", action="store_true", help="печатать каждое расхождение")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    replay_file = ROOT / "data" / "replay.json"
    if args.replay:
        llm = ReplayLLM(str(replay_file))
        if llm.meta["prompt"] != prompt_hash(SYSTEM):
            print("warning: промпт изменился после записи ответов — перезапишите: python eval.py --record", file=sys.stderr)
    else:
        llm = GeminiLLM(args.model)
        if args.record:
            llm = RecordingLLM(llm)
    stats = {f: Counter() for f in FIELDS}
    item_tp = item_fp = item_fn = 0
    retried = failed = 0
    warn_need = warn_hit = 0
    diffs: list[str] = []

    docs = sorted(Path(args.docs).glob("*.txt"))
    for doc in docs:
        gold = DocResult.model_validate(json.loads((Path(args.gold) / f"{doc.stem}.json").read_text(encoding="utf-8")))
        try:
            out = extract(doc.read_text(encoding="utf-8"), llm, guard=not args.no_guard)
        except LLMUnavailable as e:
            print(f"error: {e}", file=sys.stderr)
            return 4
        retried += out.attempts > 1 and out.ok
        failed += not out.ok
        pred = out.result
        for f in FIELDS:
            p, g = getattr(pred, f), getattr(gold, f)
            if f == "line_items":
                p, g = [i.model_dump() for i in p], [i.model_dump() for i in g]
            kind = classify(f, p, g)
            stats[f][kind] += 1
            if kind != "correct":
                diffs.append(f"  {doc.stem} {f:14} {kind.upper():7} pred={p!r} gold={g!r}")
        pk, gk = Counter(map(_item_key, (i.model_dump() for i in pred.line_items))), Counter(map(_item_key, (i.model_dump() for i in gold.line_items)))
        item_tp += sum((pk & gk).values())
        item_fp += sum((pk - gk).values())
        item_fn += sum((gk - pk).values())
        codes = {w.split(":")[0] for w in pred.warnings}
        for need in (w.split(":")[0] for w in gold.warnings):
            warn_need += 1
            warn_hit += need in codes
            if need not in codes:
                diffs.append(f"  {doc.stem} warnings       MISSING expected warning code {need!r}")

    n = len(docs)
    model_name = getattr(llm, "model", None) or llm.inner.model
    print(f"model={model_name} guard={'off' if args.no_guard else 'on'} документов={n}\n")
    print(f"{'поле':15}{'верно':>7}{'точность':>10}{'галлюц.':>9}{'пропуск':>9}{'неверно':>9}")
    tot = Counter()
    for f in FIELDS:
        c = stats[f]
        tot.update(c)
        print(f"{f:15}{c['correct']:>7}{c['correct'] / n:>10.0%}{c['halluc']:>9}{c['omit']:>9}{c['wrong']:>9}")
    cells = n * len(FIELDS)
    print(f"{'ВСЕГО':15}{tot['correct']:>7}{tot['correct'] / cells:>10.0%}{tot['halluc']:>9}{tot['omit']:>9}{tot['wrong']:>9}")
    print(f"\nГАЛЛЮЦИНАЦИИ (в эталоне null, заполнено): {tot['halluc']}")
    print(f"ПРОПУСКИ     (в эталоне есть, дали null): {tot['omit']}")
    print(f"позиции line_items: верных {item_tp}, лишних(выдуманных) {item_fp}, пропущенных {item_fn}")
    print(f"ожидаемые предупреждения найдены: {warn_hit}/{warn_need}")
    print(f"ответ невалиден с 1-й попытки, но исправлен ретраем: {retried}; провал валидации (fallback): {failed}")
    if args.record:
        meta = {"model": llm.inner.model, "recorded": date.today().isoformat(), "prompt": prompt_hash(SYSTEM)}
        replay_file.write_text(json.dumps({"_meta": meta, "answers": llm.answers}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nответы записаны: {replay_file.name} ({meta['model']}, {meta['recorded']})")
    if args.verbose and diffs:
        print("\nРасхождения:\n" + "\n".join(diffs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
