import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from docparse.schema import DocResult

GOLD = sorted((Path(__file__).parent.parent / "data" / "gold").glob("*.json"))


def valid() -> dict:
    return json.loads(GOLD[0].read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", GOLD, ids=lambda p: p.stem)
def test_every_gold_file_matches_schema(path):
    DocResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("field,bad", [
    ("supplier_bin", "12345"),                # не 12 цифр
    ("supplier_bin", "17034002155X"),         # не цифры
    ("doc_date", "12.11.2026"),               # не ISO
    ("doc_date", "2026-13-45"),               # формально ISO, но не дата
    ("currency", "RUB"),                      # вне перечисления
    ("doc_type", "receipt"),                  # вне перечисления
    ("total_amount", "186 180 тенге"),        # строка вместо числа
    ("total_amount", -5),                     # отрицательная сумма
    ("supplier_name", "  "),                  # пустая строка вместо null
])
def test_invalid_values_are_rejected(field, bad):
    d = valid()
    d[field] = bad
    with pytest.raises(ValidationError):
        DocResult.model_validate(d)


def test_missing_key_is_rejected():
    d = valid()
    del d["vat_amount"]
    with pytest.raises(ValidationError):
        DocResult.model_validate(d)


def test_extra_key_is_rejected():
    d = valid()
    d["total_with_vat"] = 1
    with pytest.raises(ValidationError):
        DocResult.model_validate(d)


def test_unknown_confidence_key_and_range_rejected():
    d = valid()
    d["field_confidence"] = {"nonexistent": 0.5}
    with pytest.raises(ValidationError):
        DocResult.model_validate(d)
    d["field_confidence"] = {"supplier_bin": 1.5}
    with pytest.raises(ValidationError):
        DocResult.model_validate(d)
