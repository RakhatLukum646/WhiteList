"""Фиксированная схема ответа. Всё, что не проходит эту схему, в систему не попадает."""
from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Числа принимаем только как JSON-числа: строка "186 180 тенге" — ошибка, а не повод "приведения типов".
Num = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]

SCALAR_FIELDS = (
    "doc_type", "supplier_name", "supplier_bin", "doc_number",
    "doc_date", "currency", "total_amount", "vat_amount",
)


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    qty: Optional[Num]
    price: Optional[Num]


class DocResult(BaseModel):
    # extra="forbid": лишнее поле = ошибка. Поля без значений по умолчанию: пропущенный ключ = ошибка,
    # модель обязана явно написать null, а не молча забыть поле.
    model_config = ConfigDict(extra="forbid")

    doc_type: Literal["invoice", "commercial_offer", "act", "unknown"]
    supplier_name: Optional[str]
    supplier_bin: Optional[str]
    doc_number: Optional[str]
    doc_date: Optional[str]
    currency: Optional[Literal["KZT", "USD", "EUR"]]
    total_amount: Optional[Num]
    vat_amount: Optional[Num]
    line_items: list[LineItem]
    field_confidence: dict[str, Annotated[float, Field(strict=True, ge=0, le=1)]]
    warnings: list[str]

    @field_validator("supplier_bin")
    @classmethod
    def _bin_12_digits(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not re.fullmatch(r"\d{12}", v):
            raise ValueError("supplier_bin must be exactly 12 digits")
        return v

    @field_validator("doc_date")
    @classmethod
    def _date_iso(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("doc_date must look like YYYY-MM-DD")
        date.fromisoformat(v)  # отсекает 2026-13-45
        return v

    @field_validator("supplier_name", "doc_number")
    @classmethod
    def _not_blank(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("empty string is not allowed, use null")
        return v

    @field_validator("field_confidence")
    @classmethod
    def _known_keys(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - set(SCALAR_FIELDS)
        if unknown:
            raise ValueError(f"unknown field_confidence keys: {sorted(unknown)}")
        return v

    @classmethod
    def empty(cls, warnings: list[str]) -> "DocResult":
        """Безопасный ответ, когда доверять нечему: всё null, doc_type=unknown."""
        return cls(
            doc_type="unknown", supplier_name=None, supplier_bin=None, doc_number=None,
            doc_date=None, currency=None, total_amount=None, vat_amount=None,
            line_items=[], field_confidence={}, warnings=warnings,
        )
