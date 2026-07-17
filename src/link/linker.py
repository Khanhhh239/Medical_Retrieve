# -*- coding: utf-8 -*-
"""
P5 — Linker: mention text -> mã chuẩn (retrieve exact -> fuzzy).

Precision-first (Jaccard phạt đoán thừa): mặc định trả TOP-1. Thuốc:
- CÓ liều (mg...) -> dựng query cấp SẢN PHẨM 'name dose oral tablet' để lấy mã SCD
  (vd 'amlodipine 10mg' -> 308135 = đúng mã ví dụ đề) TRƯỚC khi lùi về hoạt chất.
- map SYNONYM INN/VN -> tên RxNorm (US): paracetamol->acetaminophen, ...
Bệnh: match toàn chuỗi.
"""
from __future__ import annotations

import re
from typing import List, Optional

from rapidfuzz import process, fuzz

from ..io.offsets import normalize_str
from ..config import get as _cfg
from .kb import KB

_DEFAULT_FUZZY = _cfg("linker", "fuzzy_threshold", 90)

# liều: số + đơn vị (chấp nhận dính 'amlodipine10mg' hiếm, và '10 mg')
_DOSE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg|mcg|ug|g|ml|iu)\b", re.I)

# INN / cách viết VN-quốc tế -> tên hoạt chất RxNorm (US). Chỉ map cái CHẮC CHẮN.
_SYNONYM = {
    "paracetamol": "acetaminophen",
    "salbutamol": "albuterol",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "frusemide": "furosemide",
    "lignocaine": "lidocaine",
    "lidocaine hydrochloride": "lidocaine",
    "rifampicin": "rifampin",
    "glyceryl trinitrate": "nitroglycerin",
    "trinitrin": "nitroglycerin",
    "co-trimoxazole": "sulfamethoxazole / trimethoprim",
    "cotrimoxazole": "sulfamethoxazole / trimethoprim",
    "amoxicilline": "amoxicillin",
    "ciprofloxacine": "ciprofloxacin",
    "vitamin b1": "thiamine",
    "vitamin b6": "pyridoxine",
    "vitamin b12": "cyanocobalamin",
    "vitamin c": "ascorbic acid",
}


class Linker:
    def __init__(self, kb: KB, fuzzy_threshold: int = _DEFAULT_FUZZY):
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

    def _canon(self, name: str) -> str:
        """Đưa tên về dạng RxNorm (US) qua bảng synonym."""
        return _SYNONYM.get(name.strip(" -.,;:"), name.strip(" -.,;:"))

    def _drug_name(self, text_norm: str) -> str:
        # cắt ở KHOẢNG TRẮNG + số (liều), KHÔNG cắt số dính trong tên ('vitamin b12')
        m = re.search(r"\s\d", text_norm)
        name = text_norm[:m.start()] if m else text_norm
        return name.strip(" -.,;:")

    def _product_query(self, text_norm: str) -> Optional[str]:
        """CÓ liều -> 'name <num> <unit> oral tablet' (khớp mã SCD sản phẩm)."""
        m = _DOSE.search(text_norm)
        if not m:
            return None
        name = self._canon(text_norm[:m.start()])
        if not name:
            return None
        num = m.group(1).replace(",", ".")
        unit = m.group(2).lower()
        return normalize_str(f"{name} {num} {unit} oral tablet")

    def link(self, text: str, kind: str = "disease", top_k: int = 1) -> List[str]:
        q = normalize_str(text)
        if not q:
            return []
        if q in self._exact:                          # exact TOÀN mention
            return self._exact[q][:top_k]
        if kind == "drug":
            pq = self._product_query(q)               # ưu tiên mã cấp SẢN PHẨM (SCD)
            if pq and pq in self._exact:
                return self._exact[pq][:top_k]
            name = self._canon(self._drug_name(q))    # -> tên hoạt chất (đã map synonym)
            if name and name in self._exact:
                return self._exact[name][:top_k]
            q = name or q
        hit = process.extractOne(q, self._choices, scorer=fuzz.token_sort_ratio)
        if hit and hit[1] >= self.threshold:          # fuzzy toàn chuỗi
            return [self._choice_code[hit[2]]]
        return []
