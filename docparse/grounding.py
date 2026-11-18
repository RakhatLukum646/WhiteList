"""Защита от галлюцинаций: каждое значение от модели обязано находиться в тексте документа.

Правило одно: код может только УДАЛИТЬ (null) или ПОНИЗИТЬ уверенность, но никогда не добавляет значение сам.
Исключение — разрешение противоречий по явно описанному правилу (см. reconcile).
"""
from __future__ import annotations

import re

from . import textscan as ts
from .schema import DocResult

CONF_VERBATIM = 0.95   # значение найдено в тексте как есть
CONF_WORDS = 0.85      # сумма найдена только прописью
CONF_OCR = 0.6         # найдено только после исправления OCR-ошибок


def _norm(s: str) -> str:
    s = re.sub(r"[«»\"“”„'`.,;:()\[\]]", " ", s.casefold())
    return re.sub(r"\s+", " ", s).strip()


class _Evidence:
    """Где нашли значение: 'verbatim' | 'words' | 'ocr' | None."""

    def __init__(self, text: str):
        self.raw = text
        self.fixed = ts.normalize_ocr(text)
        self.raw_nums = ts.number_values(text)
        self.fixed_nums = ts.number_values(self.fixed)
        self.word_nums = {w.value for w in ts.word_amounts(text)}
        self.raw_dates = {f.iso for f in ts.dates(text)}
        self.fixed_dates = {f.iso for f in ts.dates(self.fixed)}
        self.norm_raw = _norm(text)

    def number(self, v: float) -> str | None:
        close = lambda s: any(abs(v - x) < 0.005 for x in s)  # noqa: E731
        if close(self.raw_nums):
            return "verbatim"
        if close(self.word_nums):
            return "words"
        if close(self.fixed_nums):
            return "ocr"
        return None

    def date(self, iso: str) -> str | None:
        if iso in self.raw_dates:
            return "verbatim"
        return "ocr" if iso in self.fixed_dates else None

    def digits12(self, value: str) -> str | None:
        if re.search(rf"(?<!\d){value}(?!\d)", self.raw):
            return "verbatim"
        return "ocr" if re.search(rf"(?<!\d){value}(?!\d)", self.fixed) else None

    def doc_number(self, value: str) -> str | None:
        """Номер должен стоять сразу после маркера (№, No, #, «номер»): иначе «1» находится в любом тексте."""
        marker = r"(?:№|N[°º]|\bNo\.?|#|номер|number)\s*:?\s*"
        pat = marker + re.escape(value) + r"(?![\w-])"
        if re.search(pat, self.raw, re.I):
            return "verbatim"
        return "ocr" if re.search(pat, self.fixed, re.I) else None

    def string(self, value: str) -> str | None:
        if value in self.raw:
            return "verbatim"
        if _norm(value) and _norm(value) in self.norm_raw:
            return "verbatim"
        return "ocr" if value in self.fixed else None


_CONF = {"verbatim": CONF_VERBATIM, "words": CONF_WORDS, "ocr": CONF_OCR}


def ground(result: DocResult, text: str) -> DocResult:
    """Возвращает новый DocResult, в котором остались только подтверждённые текстом значения."""
    ev = _Evidence(text)
    d = result.model_dump()
    warnings: list[str] = list(d["warnings"])
    conf: dict[str, float] = {}
    model_conf = d["field_confidence"]

    def drop(field: str, why: str) -> None:
        warnings.append(f"ungrounded_{field}: {why}; value dropped (set to null)")
        d[field] = None

    def keep(field: str, how: str) -> None:
        c = _CONF[how]
        conf[field] = min(c, model_conf.get(field, 1.0))
        if how == "ocr":
            warnings.append(f"ocr_repaired: {field} matched only after fixing OCR confusions (O/0, l/1, З/3)")

    if d["supplier_bin"] is not None:
        how = ev.digits12(d["supplier_bin"])
        drop("supplier_bin", "12 digits not found in the document") if how is None else keep("supplier_bin", how)

    if d["doc_date"] is not None:
        how = ev.date(d["doc_date"])
        drop("doc_date", "date not found in the document") if how is None else keep("doc_date", how)

    if d["supplier_name"] is not None:
        how = ev.string(d["supplier_name"])
        drop("supplier_name", "text not found in the document") if how is None else keep("supplier_name", how)

    if d["doc_number"] is not None:
        how = ev.doc_number(d["doc_number"])
        drop("doc_number", "no '№/No/#' marker followed by this value in the document") if how is None else keep("doc_number", how)

    if d["currency"] is not None:
        if d["currency"] in ts.currencies(text):
            keep("currency", "verbatim")
        else:
            drop("currency", "no currency marker for it in the document")

    for f in ("total_amount", "vat_amount"):
        if d[f] is not None:
            # 0 не подтверждаем: «НДС не облагается»/«00 тиын» — не утверждение «НДС = 0» (правило 4 в README)
            how = ev.number(d[f]) if d[f] > 0 else None
            drop(f, "number not found in the document (not computed by us on purpose)") if how is None else keep(f, how)

    items = []
    for it in d["line_items"]:
        if not ev.string(it["name"]):
            warnings.append(f"ungrounded_line_item: name {it['name']!r} not found in the document; item dropped")
            continue
        for f in ("qty", "price"):
            if it[f] is not None and ev.number(it[f]) is None:
                warnings.append(f"ungrounded_line_item_{f}: {it[f]} for {it['name']!r} not in the document; set to null")
                it[f] = None
        items.append(it)
    d["line_items"] = items

    if d["doc_type"] != "unknown":
        conf["doc_type"] = model_conf.get("doc_type", CONF_VERBATIM)
    d["field_confidence"] = conf
    d["warnings"] = warnings
    return DocResult.model_validate(d)


def reconcile(result: DocResult, text: str) -> DocResult:
    """Противоречия в документе → warnings + поле по ФИКСИРОВАННОМУ правилу (README, раздел «Правила»):
      R1. Несколько разных дат выдачи → берём первую по тексту.
      R2. Несколько разных «итого к оплате» → берём первую по тексту.
    Правило применяется кодом, а не «на совести» модели.
    Также ставит предупреждения о неочевидных несостыковках (но значения не меняет).
    """
    fixed = ts.normalize_ocr(text)
    d = result.model_dump()
    w: list[str] = d["warnings"]

    dts = ts.issue_dates(fixed)
    if len(dts) > 1:
        w.append(f"doc_date_conflict: document contains several issue dates {dts}; used the first ({dts[0]})")
        d["doc_date"] = dts[0]

    totals = ts.total_candidates(fixed)
    distinct = list(dict.fromkeys(totals))
    if len(distinct) > 1:
        w.append(f"total_conflict: document has several different totals {distinct}; used the first ({distinct[0]})")
        d["total_amount"] = distinct[0]

    cur = ts.currencies(text)
    if len(cur) > 1:
        w.append(f"multiple_currencies: markers for {cur} found in the document")

    total, vat = d["total_amount"], d["vat_amount"]
    if total is not None and vat is not None and vat > total:
        w.append("vat_exceeds_total: vat_amount is greater than total_amount")

    items = d["line_items"]
    if total is not None and items and all(i["qty"] is not None and i["price"] is not None for i in items):
        s = sum(i["qty"] * i["price"] for i in items)
        allowed = [total] + ([total - vat] if vat is not None else [])
        if not any(abs(s - a) < 0.01 for a in allowed):
            w.append(f"line_items_sum_mismatch: sum of qty*price = {s:g}, total_amount = {total:g}")

    d["warnings"] = w
    return DocResult.model_validate(d)
