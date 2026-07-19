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
    return [(s, e, t) for s, e, t, _ in
            bio_to_spans_conf(label_ids, offsets, [1.0] * len(label_ids))]


def bio_to_spans_conf(label_ids: List[int], offsets: List[Tuple[int, int]],
                      token_conf: List[float]):
    """Như bio_to_spans nhưng kèm ĐỘ TIN CẬY span = trung bình prob token trong span.
    Trả list (char_start, char_end, type, conf). Dùng để lọc span thừa (conf thấp)."""
    spans = []
    cur_type, cur_s, cur_e, cur_cf = None, None, None, []

    def flush():
        nonlocal cur_type, cur_s, cur_e, cur_cf
        if cur_type is not None:
            conf = sum(cur_cf) / len(cur_cf) if cur_cf else 0.0
            spans.append((cur_s, cur_e, cur_type, conf))
        cur_type, cur_s, cur_e, cur_cf = None, None, None, []

    for lid, (a, b), cf in zip(label_ids, offsets, token_conf):
        if a == b:                      # token đặc biệt / pad
            continue
        lab = ID2LABEL.get(int(lid), "O")
        if lab == "O":
            flush()
            continue
        prefix, typ = lab.split("-", 1)
        if prefix == "B" or typ != cur_type:
            flush()
            cur_type, cur_s, cur_e, cur_cf = typ, a, b, [cf]
        else:                            # I- cùng type -> nối
            cur_e = b
            cur_cf.append(cf)
    flush()
    return spans
