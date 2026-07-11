# -*- coding: utf-8 -*-
"""Đọc/ghi tập có nhãn dạng JSONL: {doc_id, text, concepts:[...]}."""
from __future__ import annotations

import json
from typing import List, Tuple

from ..metric.scorer import Concept

LabeledDoc = Tuple[str, str, List[Concept]]


def concept_to_dict(c: Concept) -> dict:
    return {
        "text": c.text,
        "type": c.type,
        "position": [c.start, c.end],
        "assertions": list(c.assertions),
        "candidates": list(c.candidates),
    }


def save_labeled(path: str, items: List[LabeledDoc]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, text, concepts in items:
            rec = {"doc_id": doc_id, "text": text,
                   "concepts": [concept_to_dict(c) for c in concepts]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_labeled(path: str) -> List[LabeledDoc]:
    out: List[LabeledDoc] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append((d["doc_id"], d["text"],
                        [Concept.from_dict(c) for c in d["concepts"]]))
    return out
