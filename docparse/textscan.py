"""Детерминированные сканеры текста: OCR-нормализация, даты, числа, суммы прописью, валюты.

Используются: (1) проверка, что значение от модели реально есть в документе
(grounding), (2) поиск конфликтов (две даты, два «итого»). Здесь нет вызовов LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

LETTERS = "A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі"

# --- OCR: буквы, похожие на цифры, внутри числовых токенов ---------------------------------
_C = "0-9OОoоlIІЗ"  # символы, которые OCR путает с цифрами (O/О→0, l/I/І→1, З→3)
_OCR_TABLE = str.maketrans({"O": "0", "О": "0", "o": "0", "о": "0", "l": "1", "I": "1", "І": "1", "З": "3"})
_OCR_TOKEN = re.compile(rf"(?<![{LETTERS}])[{_C}]+(?:[ .,][{_C}]+)*(?![{LETTERS}])")


def normalize_ocr(text: str) -> str:
    """Чинит только токены, где есть настоящая цифра: «2О26»→«2026», «Ol»/«ООО» не трогает."""

    def fix(m: re.Match) -> str:
        tok = m.group(0)
        first = re.split(r"[ .,]", tok)[0]
        if not re.search(r"\d", tok):
            return tok
        if len(first) < 2 and not first.isdigit():  # одинокая «о»/«I» — это слово, а не цифра
            return tok
        return tok.translate(_OCR_TABLE)

    return _OCR_TOKEN.sub(fix, text)


# --- Даты ------------------------------------------------------------------------------------
_MONTH_PREFIXES = [
    ("январ", 1), ("қаңтар", 1), ("феврал", 2), ("ақпан", 2), ("март", 3), ("наурыз", 3),
    ("апрел", 4), ("сәуір", 4), ("мая", 5), ("май", 5), ("мамыр", 5), ("июн", 6), ("маусым", 6),
    ("июл", 7), ("шілде", 7), ("август", 8), ("тамыз", 8), ("сентябр", 9), ("қыркүйек", 9),
    ("октябр", 10), ("қазан", 10), ("ноябр", 11), ("қараша", 11), ("декабр", 12), ("желтоқсан", 12),
]
_DATE_RES = [
    re.compile(r"(?<!\d)(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(?P<d>\d{1,2})[./-](?P<m>\d{1,2})[./-](?P<y>\d{4})(?!\d)"),
    re.compile(r"(?<!\d)(?P<d>\d{1,2})\s+(?P<mon>[^\W\d_]+)\s+(?P<y>\d{4})"),
]
# слова перед датой, делающие её НЕ датой документа, а сроком/окончанием
_VALIDITY_CUE = re.compile(r"действител|срок|valid|until|expir|не\s+позднее|оплатить\s+до|дейін|\bдо\s*$", re.I)


@dataclass(frozen=True)
class Found:
    iso: str
    start: int
    end: int


def _month_from_word(word: str) -> int | None:
    w = word.lower()
    for prefix, month in _MONTH_PREFIXES:
        if w.startswith(prefix):
            return month
    return None


def dates(text: str) -> list[Found]:
    out: list[Found] = []
    for rx in _DATE_RES:
        for m in rx.finditer(text):
            g = m.groupdict()
            month = _month_from_word(g["mon"]) if "mon" in g else int(g["m"])
            if month is None:
                continue
            try:
                iso = date(int(g["y"]), month, int(g["d"])).isoformat()
            except ValueError:
                continue
            if not any(f.start < m.end() and m.start() < f.end for f in out):
                out.append(Found(iso, m.start(), m.end()))
    return sorted(out, key=lambda f: f.start)


def issue_dates(text: str) -> list[str]:
    """Даты, которые могут быть датой документа (без «действительно до …»), в порядке появления."""
    res: list[str] = []
    for f in dates(text):
        line_start = text.rfind("\n", 0, f.start) + 1
        ctx = text[max(line_start, f.start - 30):f.start]
        if _VALIDITY_CUE.search(ctx):
            continue
        if f.iso not in res:
            res.append(f.iso)
    return res


# --- Числа ------------------------------------------------------------------------------------
_NUM = re.compile(r"(?<![\d.,])(\d{1,3}(?:[  ,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d+)?)(?!\d)")


def parse_number(tok: str) -> float:
    s = tok.replace(" ", "").replace(" ", "")
    if "." in s and "," in s:
        s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
    return float(s)


@dataclass(frozen=True)
class Num:
    value: float
    start: int
    end: int
    percent: bool  # сразу за числом стоит «%» — это ставка, не сумма


def numbers(text: str) -> list[Num]:
    date_spans = [(f.start, f.end) for f in dates(text)]
    out = []
    for m in _NUM.finditer(text):
        if any(s < m.end() and m.start() < e for s, e in date_spans):
            continue
        pct = text[m.end():m.end() + 2].lstrip().startswith("%")
        out.append(Num(parse_number(m.group(1)), m.start(), m.end(), pct))
    return out


def number_values(text: str) -> set[float]:
    """Значения, которые могут быть суммой/количеством. Ставки («16%») — не суммы, в набор не входят."""
    return {n.value for n in numbers(text) if not n.percent}


# --- Суммы прописью (рус. + каз.) -------------------------------------------------------------
_ADD = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12,
    "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
    "восемнадцать": 18, "девятнадцать": 19, "двадцать": 20, "тридцать": 30, "сорок": 40,
    "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90, "сто": 100,
    "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500, "шестьсот": 600, "семьсот": 700,
    "восемьсот": 800, "девятьсот": 900,
    "нөл": 0, "бір": 1, "екі": 2, "үш": 3, "төрт": 4, "бес": 5, "алты": 6, "жеті": 7, "сегіз": 8,
    "тоғыз": 9, "он": 10, "жиырма": 20, "отыз": 30, "қырық": 40, "елу": 50, "алпыс": 60, "жетпіс": 70,
    "сексен": 80, "тоқсан": 90,
}
_SCALE = {"тысяча": 1_000, "тысячи": 1_000, "тысяч": 1_000, "мың": 1_000,
          "миллион": 1_000_000, "миллиона": 1_000_000, "миллионов": 1_000_000}
_HUNDRED_MULT = {"жүз"}  # каз.: «үш жүз» = 3·100 (в русском сотни — отдельные слова)
_CURRENCY_WORDS = {"тенге", "теңге", "тг", "тенг", "доллар", "долларов", "доллара", "евро", "usd", "kzt", "eur"}


def words_to_number(words: list[str]) -> float:
    total = cur = 0
    for w in words:
        if w in _ADD:
            cur += _ADD[w]
        elif w in _HUNDRED_MULT:
            cur = (cur or 1) * 100
        elif w in _SCALE:
            total += (cur or 1) * _SCALE[w]
            cur = 0
    return float(total + cur)


def word_amounts(text: str) -> list[Num]:
    """Суммы прописью. Берём только серии числительных, за которыми идёт название валюты."""
    toks = [(m.group(0).lower(), m.start(), m.end()) for m in re.finditer(r"[^\W\d_]+", text)]
    vocab = set(_ADD) | set(_SCALE) | _HUNDRED_MULT
    out: list[Num] = []
    i = 0
    while i < len(toks):
        if toks[i][0] not in vocab:
            i += 1
            continue
        j = i
        while j < len(toks) and toks[j][0] in vocab:
            j += 1
        if j < len(toks) and toks[j][0] in _CURRENCY_WORDS:
            val = words_to_number([t[0] for t in toks[i:j]])
            if val > 0:
                out.append(Num(val, toks[i][1], toks[j - 1][2], False))
        i = j
    return out


# --- Валюты -----------------------------------------------------------------------------------
_CURRENCY_RES = {
    "KZT": re.compile(r"тенге|теңге|\bтг\b|₸|KZT", re.I),
    "USD": re.compile(r"USD|\$|доллар", re.I),
    "EUR": re.compile(r"EUR|€|евро", re.I),
}


def currencies(text: str) -> list[str]:
    """Коды валют в порядке первого упоминания."""
    hits = []
    for code, rx in _CURRENCY_RES.items():
        m = rx.search(text)
        if m:
            hits.append((m.start(), code))
    return [c for _, c in sorted(hits)]


# --- Итого / НДС ------------------------------------------------------------------------------
_TOTAL_PRIMARY = re.compile(
    r"всего\s+к\s+оплате|итого\s+к\s+оплате|к\s+оплате|всего\s+с\s+ндс|итого\s+с\s+ндс|жалпы\s+сомасы|"
    r"барлығы|total\s+due|amount\s+due|grand\s+total|\btotal\b", re.I)
_TOTAL_SECONDARY = re.compile(r"итого|всего", re.I)


def amounts_in_line(line: str) -> list[float]:
    """Суммы в строке: сначала прописью, потом цифрами (без процентов и нулей вроде «00 тиын»)."""
    words = word_amounts(line)
    if words:
        return [w.value for w in words]
    return [n.value for n in numbers(line) if not n.percent and n.value > 0]


def total_candidates(text: str) -> list[float]:
    """Значения «итого к оплате» в порядке появления. Сначала строки с явной меткой «к оплате/всего с НДС/Total»,
    если таких нет — просто «Итого»/«Всего»."""
    for rx in (_TOTAL_PRIMARY, _TOTAL_SECONDARY):
        vals = []
        for line in text.splitlines():
            if rx.search(line):
                a = amounts_in_line(line)
                if a:
                    vals.append(a[-1])
        if vals:
            return vals
    return []


# --- Роли и метки: значение подтверждается только в «своём» контексте ---------------------------
BUYER = re.compile(r"заказчик|покупатель|плательщик|buyer|bill\s+to|customer", re.I)
_VAT_LABEL = re.compile(r"(?<![сc]\s)(?:ндс|ққс|vat)", re.I)  # «с НДС» — это итог, а не сам НДС


def supplier_bin_candidates(text: str) -> set[str]:
    """12-значные токены, которые могут быть БИН поставщика: не на строке заказчика/покупателя
    и не на строке сразу после неё (там БИН стоит под названием заказчика)."""
    lines = text.splitlines()
    out: set[str] = set()
    for i, line in enumerate(lines):
        prev = next((lines[j] for j in range(i - 1, -1, -1) if lines[j].strip()), "")
        if BUYER.search(line) or BUYER.search(prev):
            continue
        out.update(re.findall(r"(?<!\d)\d{12}(?!\d)", line))
    return out


def vat_numbers(text: str) -> set[float]:
    """Числа на строках с меткой НДС/VAT, кроме ставок (16%) и нулей."""
    out: set[float] = set()
    for line in text.splitlines():
        if _VAT_LABEL.search(line):
            out.update(n.value for n in numbers(line) if not n.percent and n.value > 0)
    return out
