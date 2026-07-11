# -*- coding: utf-8 -*-
"""
P2/S7 — Assembler + Writer (§S7 medical.md).

- Grounding check §3.1: loại mọi concept mà raw[start:end] != text.
- Sort theo position. KHÔNG dedupe (cụm lặp giữ nguyên — position khác nhau).
- BỎ HẲN key `candidates` cho type không map (triệu chứng / tên XN / kết quả XN).
"""
from __future__ import annotations

import json
from typing import List, Sequence

from ..io.offsets import is_grounded
from ..metric.scorer import Concept, CANDIDATE_TYPES


def to_records(concepts: Sequence[Concept], raw: str) -> List[dict]:
    recs: List[dict] = []
    for c in sorted(concepts, key=lambda x: (x.start, x.end)):
        if not is_grounded(raw, c.text, c.position):
            continue                      # chặn cứng hallucination / lệch offset
        d = {
            "text": c.text,
            "position": [c.start, c.end],
            "type": c.type,
            "assertions": list(c.assertions),
        }
        if c.type in CANDIDATE_TYPES:     # chỉ CHẨN_ĐOÁN / THUỐC mới có key này
            d["candidates"] = list(c.candidates)
        recs.append(d)
    return recs


def save_json(concepts: Sequence[Concept], raw: str, path: str) -> List[dict]:
    recs = to_records(concepts, raw)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    return recs
