"""Офлайн-заглушки вместо LLM (платного ключа нет, Ollama может быть не установлен).

ВАЖНО: это не языковые модели. HeuristicLLM — набор regex-правил, написанный под наш тестовый набор;
его цифры в eval НЕ показатель качества настоящей LLM. Пайплайн, валидация, grounding и eval вокруг него — настоящие.
NaiveLLM нарочно ведёт себя как плохая модель (выдумывает НДС, дату, БИН; иногда отдаёт невалидный JSON),
чтобы eval показал, что защита работает.
"""
from __future__ import annotations

import json
import re
import zlib
from datetime import date

from . import textscan as ts

_BUYER = re.compile(r"заказчик|покупатель|плательщик|buyer|bill\s+to|customer", re.I)
_LEGAL = r"(?:ТОО|ЖШС|АО|ИП|ЖК|ООО|LLC|LLP|Ltd)"
_QUOTED = re.compile(rf"\b({_LEGAL})\.?\s*[«\"“„]\s*([^»\"”“]+?)\s*[»\"”“]")
_SUPPLIER_LABEL = re.compile(r"^\W*(?:исполнитель|поставщик|продавец|supplier|seller|vendor|provider)\s*[:\-–]\s*(.+)$", re.I)
_COMPANY_QUOTED = re.compile(r"(?:от\s+компании|компания|company)\s+[«\"“]([^»\"”]+)[»\"”]", re.I)
_BIN = re.compile(r"(?:БИН|БСН|BIN)\b[^\d\n]{0,10}(\d{12})")
_NUMBER_ID = re.compile(r"(?:№|N[°º]\s*|\bNo\.?\s+|#)\s*([^\s,;]+)")
_TITLE = re.compile(r"^\W*(акт\b|сч[её]т|шот|invoice|коммерческое\s+предложение|коммерческая)", re.I)
_CUR_SUFFIX = r"(?:тг|тенге|₸|KZT|USD|EUR|\$|€)?"
_AMOUNT = r"\d[\d  .,]*"
_ROW_BULLET = re.compile(rf"^\s*[-•*–—]\s+(?P<name>.+?)\s+[—–-]\s+(?P<price>{_AMOUNT}?)\s*{_CUR_SUFFIX}\s*$")
_ROW_QTY = re.compile(rf"^(?P<name>.+?)\s+-\s+(?P<qty>\d+(?:[.,]\d+)?)\s*(?:ед|шт|усл|pcs|компл)\w*\.?\s+-\s+(?P<price>{_AMOUNT}?)\s*{_CUR_SUFFIX}\s*$", re.I)
_TABLE_HEAD = re.compile(r"наименован|description|товар", re.I)
_TABLE_COLS = re.compile(r"кол|qty|quantity|цена|price|сумма|amount", re.I)
_TABLE_END = re.compile(r"^\s*(итого|всего|total|subtotal|в\s+том\s+числе|ндс|vat|без)", re.I)


def _cols(line: str) -> list[str]:
    return [c.strip() for c in re.split(r"\s{2,}|\t", line.strip()) if c.strip()]


def _is_num(cell: str) -> bool:
    return bool(ts.NUM_FULL.fullmatch(cell))


def _doc_type(lines: list[str]) -> str:
    for line in [l for l in lines if l.strip()][:8]:
        m = _TITLE.match(line)
        if m:
            w = m.group(1).lower()
            return "act" if w.startswith("акт") else "commercial_offer" if w.startswith("коммерч") else "invoice"
    return "unknown"


def _supplier(lines: list[str]) -> str | None:
    def pretty(s: str) -> str:
        quoted = _QUOTED.findall(s)
        if quoted:
            # двуязычное «ЖШС … / ТОО …» — берём русскую форму
            legal, name = next((q for q in quoted if q[0] != "ЖШС"), quoted[0])
            return f'{legal} "{name}"'
        return re.split(r",?\s*(?:БИН|ИИН|БСН|BIN)\b", s)[0].strip(" .,;")

    for line in lines:
        m = _SUPPLIER_LABEL.match(line)
        if m:
            return pretty(m.group(1))
    for line in lines:
        if not _BUYER.search(line):
            if _QUOTED.search(line):
                return pretty(line)
            m = _COMPANY_QUOTED.search(line)
            if m:
                return m.group(1).strip()
    return None


def _bin(lines: list[str]) -> str | None:
    for i, line in enumerate(lines):
        m = _BIN.search(line)
        if not m:
            continue
        prev = next((lines[j] for j in range(i - 1, -1, -1) if lines[j].strip()), "")
        if _BUYER.search(line) or _BUYER.search(prev):
            continue  # БИН покупателя — не наш
        return m.group(1)
    return None


def _doc_number(text: str) -> str | None:
    for m in _NUMBER_ID.finditer(text):
        tok = m.group(1).strip(".,;:")
        if re.search(r"\d", tok):
            return tok
    return None


def _items(lines: list[str]) -> list[dict]:
    items, in_table = [], False
    for line in lines:
        if not in_table:
            if _TABLE_HEAD.search(line) and len(_TABLE_COLS.findall(line)) >= 2 and len(_cols(line)) >= 3:
                in_table = True
            continue
        if not line.strip():
            continue
        if _TABLE_END.match(line):
            break
        cells = _cols(line)
        k = len(cells)
        while k > 0 and _is_num(cells[k - 1]):
            k -= 1
        nums, name = [ts.parse_number(c) for c in cells[k:]], " ".join(cells[:k])
        if not name or not nums:
            continue
        qty, price = (nums[-3], nums[-2]) if len(nums) >= 3 else (None, nums[0])
        items.append({"name": name, "qty": qty, "price": price})
    if items:
        return items
    for line in lines:
        m = _ROW_QTY.match(line.strip())
        if m:
            items.append({"name": m["name"], "qty": ts.parse_number(m["qty"]), "price": ts.parse_number(m["price"])})
            continue
        m = _ROW_BULLET.match(line)
        if m and m["price"].strip():
            items.append({"name": m["name"], "qty": None, "price": ts.parse_number(m["price"].strip())})
    return items


def heuristic_extract(raw_text: str) -> dict:
    text = ts.normalize_ocr(raw_text)
    lines = text.splitlines()
    cur = ts.currencies(text)
    dts = ts.issue_dates(text)
    totals, vats = ts.total_candidates(text), ts.vat_candidates(text)
    return {
        "doc_type": _doc_type(lines),
        "supplier_name": _supplier(lines),
        "supplier_bin": _bin(lines),
        "doc_number": _doc_number(text),
        "doc_date": dts[0] if dts else None,
        "currency": cur[0] if cur else None,
        "total_amount": totals[0] if totals else None,
        "vat_amount": vats[0] if vats else None,
        "line_items": _items(lines),
        "field_confidence": {},
        "warnings": [],
    }


def _document_of(messages: list[dict]) -> str:
    m = re.search(r"<document>\n(.*)\n</document>", messages[1]["content"], re.S)
    return m.group(1) if m else messages[1]["content"]


class HeuristicLLM:
    def complete(self, messages: list[dict]) -> str:
        return json.dumps(heuristic_extract(_document_of(messages)), ensure_ascii=False)


class NaiveLLM:
    """«Уверенно выдумывающая» модель: берёт ответ эвристики и дополняет его тем, чего в документе нет."""

    def complete(self, messages: list[dict]) -> str:
        text = _document_of(messages)
        d = heuristic_extract(text)
        # типичные грехи: НДС «по ставке», сегодняшняя дата, валюта по умолчанию, БИН «из головы»
        if d["vat_amount"] is None and d["total_amount"] is not None:
            d["vat_amount"] = round(d["total_amount"] * 16 / 116, 2)
        if d["doc_date"] is None:
            d["doc_date"] = date.today().isoformat()
        if d["currency"] is None:
            d["currency"] = "KZT"
        if d["supplier_bin"] is None:
            d["supplier_bin"] = "123456789012"
        if d["doc_number"] is None:
            d["doc_number"] = "1"
        # в трети случаев первый ответ ломает формат; после замечания об ошибке — исправляется
        if len(messages) == 2 and zlib.crc32(text.encode()) % 3 == 0:
            if zlib.crc32(text.encode()) % 2 == 0:
                d["total_amount"] = f"{d['total_amount']} тенге"          # число строкой
                return "Вот JSON:\n```json\n" + json.dumps(d, ensure_ascii=False) + "\n```"
            del d["warnings"]                                              # забытый ключ
        return json.dumps(d, ensure_ascii=False)
