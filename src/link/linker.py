# -*- coding: utf-8 -*-
"""
P5 — Linker: mention text -> mã chuẩn (retrieve exact -> fuzzy).

Precision-first (Jaccard phạt đoán thừa): mặc định trả TOP-1. Thuốc: tách tên khỏi
liều trước khi match (brand/ingredient). Bệnh: match toàn chuỗi.
"""
from __future__ import annotations

import re
from typing import List, Optional

from rapidfuzz import process, fuzz

from ..io.offsets import normalize_str
from .kb import KB


class Linker:
    def __init__(self, kb: KB, fuzzy_threshold: int = 90):
        self.kb = kb
        self.threshold = fuzzy_threshold
        self._exact: dict = {}
        self._choices: List[str] = []
        self._choice_code: List[str] = []
        for code, terms in kb.code_to_terms.items():
            for t in terms:
                nt = normalize_str(t)
                self._exact.setdefault(nt, [])
                if code not in self._exact[nt]:
                    self._exact[nt].append(code)
                self._choices.append(nt)
                self._choice_code.append(code)

    def _drug_name(self, text_norm: str) -> str:
        # cắt ở KHOẢNG TRẮNG + số (liều), KHÔNG cắt số dính trong tên ('vitamin b12')
        m = re.search(r"\s\d", text_norm)
        name = text_norm[:m.start()] if m else text_norm
        return name.strip(" -.,;:")

    def link(self, text: str, kind: str = "disease", top_k: int = 1) -> List[str]:
        q = normalize_str(text)
        if not q:
            return []
        if q in self._exact:                          # exact TOÀN mention (api-cache RxNav)
            return self._exact[q][:top_k]
        if kind == "drug":
            name = self._drug_name(q)                 # cắt liều -> tên hoạt chất
            if name and name in self._exact:          # exact tên (seed/RRF ingredient)
                return self._exact[name][:top_k]
            q = name or q
        hit = process.extractOne(q, self._choices, scorer=fuzz.token_sort_ratio)
        if hit and hit[1] >= self.threshold:          # fuzzy toàn chuỗi
            return [self._choice_code[hit[2]]]
        return []
