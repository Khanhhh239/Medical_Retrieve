# -*- coding: utf-8 -*-
"""P5 — Nối linking vào concept: điền candidates cho THUỐC (RxNorm) & CHẨN_ĐOÁN (ICD)."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import List, Optional

from ..metric.scorer import Concept
from .kb import load_rxnorm, load_icd10
from .linker import Linker


@lru_cache(maxsize=1)
def _rx() -> Linker:
    return Linker(load_rxnorm())


@lru_cache(maxsize=1)
def _icd() -> Linker:
    return Linker(load_icd10())


def get_linkers():
    """(rx, icd) — dùng cho KB-trim khi khử nhiễu span."""
    return _rx(), _icd()


def link_concepts(concepts: List[Concept], top_k: int = 1) -> List[Concept]:
    out: List[Concept] = []
    for c in concepts:
        if c.type == "THUỐC":
            out.append(replace(c, candidates=tuple(_rx().link(c.text, "drug", top_k))))
        elif c.type == "CHẨN_ĐOÁN":
            out.append(replace(c, candidates=tuple(_icd().link(c.text, "disease", top_k))))
        else:
            out.append(c)
    return out
