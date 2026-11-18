import pytest

from docparse import textscan as ts


@pytest.mark.parametrize("text,expected", [
    ("Үш жүз қырық мың теңге 00 тиын", 340000),
    ("Сто двадцать тысяч тенге", 120000),
    ("(Один миллион двести пятьдесят тысяч) тенге", 1250000),
    ("Екі жүз елу мың теңге", 250000),
])
def test_amount_in_words(text, expected):
    assert [w.value for w in ts.word_amounts(text)] == [expected]


def test_words_without_currency_are_not_amounts():
    assert ts.word_amounts("он человек сто раз") == []


@pytest.mark.parametrize("text,iso", [
    ("от 12.11.2026", "2026-11-12"),
    ("Date: 2026-10-01", "2026-10-01"),
    ("09 қазан 2026 ж.", "2026-10-09"),
    ("5 ноября 2026 года", "2026-11-05"),
])
def test_date_formats(text, iso):
    assert [f.iso for f in ts.dates(text)] == [iso]


def test_impossible_date_is_ignored():
    assert ts.dates("31.02.2026") == []


def test_validity_date_is_not_an_issue_date():
    text = "Предложение действительно до 30.11.2026.\nСоставлено 05.11.2026."
    assert ts.issue_dates(text) == ["2026-11-05"]


def test_ocr_fix_only_touches_numeric_tokens():
    assert ts.normalize_ocr("БИН 1Ol24OO03571") == "БИН 101240003571"
    assert ts.normalize_ocr("от О3.11.2О26") == "от 03.11.2026"
    assert ts.normalize_ocr("ООО Ромашка, Olga, о нас") == "ООО Ромашка, Olga, о нас"
    assert ts.normalize_ocr("М4ОО") == "М4ОО"  # марка цемента — не число


@pytest.mark.parametrize("tok,val", [("186 180", 186180), ("1,548.50", 1548.5), ("1 234 567,89", 1234567.89), ("12,5", 12.5)])
def test_number_parsing(tok, val):
    assert ts.parse_number(tok) == val
