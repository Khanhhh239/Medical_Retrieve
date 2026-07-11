# -*- coding: utf-8 -*-
"""
P1 — Parser số đo lường (phục vụ nhận diện KẾT_QUẢ_XÉT_NGHIỆM ở S3c).

Chỉ PHÂN TÍCH, không viết lại output (text xuất ra luôn = raw[start:end], §3.1).

Điểm mấu chốt — dấu phẩy NHẬP NHẰNG (§3.4 medical.md):
    "14,43"  -> thập phân (kiểu VN)  = 14.43
    "21,000" -> phân cách nghìn      = 21000
Quy tắc: phẩy theo sau đúng 3 chữ số (và lặp được) -> nghìn; 1-2 chữ số -> thập phân.
Test set thật dùng CHẤM thập phân (114) áp đảo phẩy (12) -> cả hai đều phải chịu được.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# thứ tự nhánh QUAN TRỌNG: nghìn trước, rồi thập phân, rồi nguyên
_NUM = re.compile(
    r"[+-]?(?:"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"   # 21,000 | 1,234.56  (phân cách nghìn)
    r"|\d+[.,]\d+"                     # 0.01 | 6.3 | 14,43  (thập phân)
    r"|\d+"                            # 30 | 12987          (nguyên)
    r")"
)
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


@dataclass
class Number:
    raw: str
    start: int             # offset TƯƠNG ĐỐI trong text truyền vào
    end: int
    kind: str              # "thousands" | "decimal" | "int"
    value: float


def _classify(tok: str) -> "tuple[str, float]":
    if _THOUSANDS.match(tok):
        return "thousands", float(tok.replace(",", ""))
    if "." in tok or "," in tok:
        return "decimal", float(tok.replace(",", "."))
    return "int", float(tok)


def find_numbers(text: str) -> List[Number]:
    out: List[Number] = []
    for m in _NUM.finditer(text):
        tok = m.group(0)
        kind, val = _classify(tok)
        out.append(Number(raw=tok, start=m.start(), end=m.end(), kind=kind, value=val))
    return out


# arrow/en-dash nối 2 số -> 1 khoảng "1.1-->0.8". KHÔNG dùng gạch ngang '-' trần
# (tránh coi '3-4 ngày' là range). Yêu cầu mũi tên / en-dash / em-dash rõ ràng.
_RANGE = re.compile(
    r"(?P<a>[+-]?\d+(?:[.,]\d+)?)\s*(?:-->|->|—|–)\s*(?P<b>[+-]?\d+(?:[.,]\d+)?)"
)


@dataclass
class RangeTok:
    raw: str
    start: int
    end: int
    a: float
    b: float


def find_ranges(text: str) -> List[RangeTok]:
    out: List[RangeTok] = []
    for m in _RANGE.finditer(text):
        a = float(m.group("a").replace(",", "."))
        b = float(m.group("b").replace(",", "."))
        out.append(RangeTok(m.group(0), m.start(), m.end(), a, b))
    return out


def looks_like_measurement(text: str) -> bool:
    """Heuristic: dòng ngắn CÓ số -> ứng viên kết quả xét nghiệm."""
    t = text.strip()
    if not t or len(t) > 80:
        return False
    return bool(_NUM.search(t))
