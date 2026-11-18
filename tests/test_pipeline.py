"""Пайплайн на скриптованной «модели»: проверяем поведение системы, а не качество модели."""
import json
from pathlib import Path

from docparse.pipeline import extract

DOC1 = (Path(__file__).parent.parent / "data" / "docs" / "01.txt").read_text(encoding="utf-8")
DOC7 = (Path(__file__).parent.parent / "data" / "docs" / "07.txt").read_text(encoding="utf-8")


def answer(**over) -> dict:
    d = dict(doc_type="invoice", supplier_name='ТОО "Алматы Трейд Компани"', supplier_bin="170340021554",
             doc_number="1245", doc_date="2026-11-12", currency="KZT", total_amount=186180, vat_amount=25680,
             line_items=[], field_confidence={}, warnings=[])
    d.update(over)
    return d


class Scripted:
    """Отдаёт заранее заданные ответы по очереди и запоминает, что ему присылали."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages):
        self.calls.append(messages)
        r = self.replies.pop(0)
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)


def test_valid_answer_passes_first_time():
    out = extract(DOC1, Scripted(answer()))
    assert out.ok and out.attempts == 1 and out.errors == []
    assert out.result.total_amount == 186180


def test_invalid_answer_is_retried_with_the_error_message():
    llm = Scripted(answer(total_amount="186 180 тенге"), answer())
    out = extract(DOC1, llm)
    assert out.ok and out.attempts == 2
    assert "total_amount" in out.errors[0]
    repair = llm.calls[1][-1]["content"]  # модели показали, что именно не так
    assert "total_amount" in repair and llm.calls[1][-2]["role"] == "assistant"


def test_garbage_then_markdown_fenced_json_is_accepted():
    out = extract(DOC1, Scripted("извините, не могу", "```json\n" + json.dumps(answer()) + "\n```"))
    assert out.ok and out.attempts == 2


def test_never_valid_returns_safe_empty_result_not_exception():
    out = extract(DOC1, Scripted("not json", {"doc_type": "invoice"}, answer(supplier_bin="12")))
    assert not out.ok and out.attempts == 3 and len(out.errors) == 3
    r = out.result
    assert r.doc_type == "unknown" and r.total_amount is None and r.line_items == []
    assert any(w.startswith("validation_failed") for w in r.warnings)


def test_vat_not_present_in_document_is_dropped_not_computed():
    text = "Счёт № 5 от 01.10.2026\nВсего к оплате: 116 000 тенге"
    out = extract(text, Scripted(answer(total_amount=116000, vat_amount=16000, doc_number="5", doc_date="2026-10-01",
                                        supplier_name=None, supplier_bin=None)))
    assert out.result.vat_amount is None
    assert any(w.startswith("ungrounded_vat_amount") for w in out.result.warnings)


def test_todays_date_and_invented_bin_are_dropped():
    text = "ТОО \"Ромашка 170340021554\"\nСчёт № 5\nВсего к оплате: 1 000 тенге"
    out = extract(text, Scripted(answer(supplier_name='ТОО "Ромашка 170340021554"', supplier_bin="123456789012",
                                        doc_date="2026-11-18", total_amount=1000, vat_amount=None, doc_number="5")))
    r = out.result
    assert r.doc_date is None and r.supplier_bin is None
    assert r.supplier_name == 'ТОО "Ромашка 170340021554"'  # настоящее значение из текста не тронуто


def test_bin_embedded_in_a_longer_digit_string_is_not_accepted():
    text = "Счёт № 5\nРасчётный счёт 99170340021554001"
    out = extract(text, Scripted(answer(doc_number="5", doc_date=None, supplier_name=None, total_amount=None,
                                        vat_amount=None, currency=None)))
    assert out.result.supplier_bin is None


def test_generic_doc_number_is_not_accepted_without_marker():
    out = extract("Просим счёт. 1 шт.", Scripted(answer(doc_number="1", supplier_bin=None, doc_date=None,
                  supplier_name=None, total_amount=None, vat_amount=None, currency=None)))
    assert out.result.doc_number is None


def test_conflicting_dates_and_totals_produce_warnings_and_follow_the_rule():
    # модель «выбрала» второе значение; код возвращает по правилу «первое по тексту»
    out = extract(DOC7, Scripted(answer(supplier_name='ТОО "Тараз Моторс"', supplier_bin="150840012345",
                                        doc_number="88", doc_date="2026-11-05", total_amount=125000, vat_amount=None)))
    r = out.result
    assert r.doc_date == "2026-11-03" and r.total_amount == 120000
    codes = {w.split(":")[0] for w in r.warnings}
    assert {"doc_date_conflict", "total_conflict"} <= codes


def test_confidence_is_lower_for_ocr_repaired_values():
    text = "ТОО \"Семей Снаб\"\nБИН 1Ol24OO03571\nСчёт № 117"
    out = extract(text, Scripted(answer(supplier_name='ТОО "Семей Снаб"', supplier_bin="101240003571", doc_number="117",
                                        doc_date=None, total_amount=None, vat_amount=None, currency=None)))
    c = out.result.field_confidence
    assert c["supplier_bin"] < c["supplier_name"]
    assert any(w.startswith("ocr_repaired: supplier_bin") for w in out.result.warnings)


def test_unavailable_model_is_not_retried():
    import pytest
    from docparse.llm import LLMUnavailable

    class Down:
        n = 0

        def complete(self, messages):
            self.n += 1
            raise LLMUnavailable("down")

    d = Down()
    with pytest.raises(LLMUnavailable):
        extract(DOC1, d)
    assert d.n == 1


def test_zero_vat_is_not_accepted_even_if_zero_appears_in_text():
    text = "Шот № 5\nЖалпы сомасы: Үш жүз қырық мың теңге 00 тиын\nНДС не облагается"
    out = extract(text, Scripted(answer(doc_number="5", total_amount=340000, vat_amount=0.0, supplier_name=None,
                                        supplier_bin=None, doc_date=None)))
    assert out.result.vat_amount is None


def test_model_may_report_confidence_for_line_items():
    out = extract(DOC1, Scripted(answer(field_confidence={"line_items": 1.0})))
    assert out.ok and out.attempts == 1
