# -*- coding: utf-8 -*-
"""Nhãn BIO cho NER 5 type (P4)."""
from __future__ import annotations

from typing import List, Tuple

TYPES = ["TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "CHẨN_ĐOÁN", "THUỐC"]

LABELS: List[str] = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)


def bio_to_spans(label_ids: List[int], offsets: List[Tuple[int, int]]):
    """Chuỗi nhãn BIO + offset token -> list (char_start, char_end, type)."""
    spans = []
    cur_type, cur_s, cur_e = None, None, None

    def flush():
        nonlocal cur_type, cur_s, cur_e
        if cur_type is not None:
            spans.append((cur_s, cur_e, cur_type))
        cur_type, cur_s, cur_e = None, None, None

    for lid, (a, b) in zip(label_ids, offsets):
        if a == b:                      # token đặc biệt / pad
            continue
        lab = ID2LABEL.get(int(lid), "O")
        if lab == "O":
            flush()
            continue
        prefix, typ = lab.split("-", 1)
        if prefix == "B" or typ != cur_type:
            flush()
            cur_type, cur_s, cur_e = typ, a, b
        else:                            # I- cùng type -> nối
            cur_e = b
    flush()
    return spans
