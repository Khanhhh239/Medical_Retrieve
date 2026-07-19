# -*- coding: utf-8 -*-
"""
v3 · Khối D — Data cho ASSERTION HỌC (thay ConText regex).

Mỗi khái niệm assertable -> 1 ví dụ phân-loại-đa-nhãn: chèn marker [E]..[/E] quanh
khái niệm trong CỬA SỔ ngữ cảnh (bắt được cả header 'Tiền sử:' cho isHistorical +
cue phủ định trước đó), nhãn = vector 3 bit {isNegated, isHistorical, isFamily}.

Nhãn lấy từ silver LLM — phần LLM GIỎI nhất là ngữ nghĩa (phủ định/tiền sử), khác với
biên (nhiễu). Thuần logic, không GPU -> test được.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

ASSERT_LABELS = ["isNegated", "isHistorical", "isFamily"]
ASSERTABLE = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}
E_OPEN, E_CLOSE = "[E]", "[/E]"


def mark(raw: str, start: int, end: int, before: int = 300, after: int = 80) -> str:
    """Ngữ cảnh quanh khái niệm với marker [E] .. [/E]. Cửa sổ ký tự (bắt header + cue)."""
    ls = max(0, start - before)
    le = min(len(raw), end + after)
    return (raw[ls:start] + " " + E_OPEN + " " + raw[start:end] + " " + E_CLOSE + " "
            + raw[end:le])


def labels_of(assertions: Sequence[str]) -> List[float]:
    aset = set(assertions)
    return [1.0 if a in aset else 0.0 for a in ASSERT_LABELS]


def build_examples(labeled_docs) -> List[Tuple[str, List[float]]]:
    """labeled_docs: [(doc_id, text, [Concept])] -> [(marked_text, [3 bit])] cho type assertable."""
    out: List[Tuple[str, List[float]]] = []
    for _did, text, concepts in labeled_docs:
        for c in concepts:
            if c.type in ASSERTABLE:
                out.append((mark(text, c.start, c.end), labels_of(c.assertions)))
    return out
