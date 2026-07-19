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

try:
    from datagen.abbrev import expand as _abbrev_expand
except Exception:                                    # datagen không nằm trên path -> no-op
    def _abbrev_expand(x):
        return x

_DEFAULT_FUZZY = _cfg("linker", "fuzzy_threshold", 90)
_DX_FUZZY = _cfg("linker", "disease_fuzzy_threshold", 82)   # bệnh nới hơn (VN nhiều biến thể)

# liều: số + đơn vị (chấp nhận dính 'amlodipine10mg' hiếm, và '10 mg')
_DOSE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg|mcg|ug|g|ml|iu)\b", re.I)

# từ THỪA trong chẩn đoán phá match ('tăng huyết áp nguyên phát' -> I10)
_DX_MOD = re.compile(
    r"\b(không đặc hiệu|không xác định|nguyên phát|thứ phát|nghi ngờ|theo dõi|"
    r"chưa rõ nguyên nhân|mức độ \w+|giai đoạn \w+)\b", re.I)

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
    def __init__(self, kb: KB, fuzzy_threshold: int = _DEFAULT_FUZZY,
                 aggressive: bool = False):
        # aggressive=False (MẶC ĐỊNH): disease = exact -> fuzzy chặt (bản 34.60).
        # aggressive=True: thêm strip-modifier + substring + fuzzy nới — ĐÃ ĐO trên
        # leaderboard là XẤU HƠN (over-fill mã sai, gold nhiều candidate rỗng) -> chỉ để A/B.
        self.kb = kb
        self.threshold = fuzzy_threshold
        self.aggressive = aggressive
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

    def _substring_code(self, q: str) -> Optional[str]:
        """Tìm mã có TERM là chuỗi con của q (mention cụ thể hơn) — chọn term DÀI nhất;
        hoặc term chứa q — chọn term NGẮN nhất (gần nghĩa nhất). Chỉ term >=5 ký tự."""
        best_in = (0, None)      # (len term, code) term ⊂ q
        best_out = (10 ** 9, None)   # (len term, code) q ⊂ term
        for nt, code in zip(self._choices, self._choice_code):
            if len(nt) < 5:
                continue
            if nt in q and len(nt) > best_in[0]:
                best_in = (len(nt), code)
            elif len(q) >= 5 and q in nt and len(nt) < best_out[0]:
                best_out = (len(nt), code)
        return best_in[1] or best_out[1]

    def _disease_link(self, q: str, top_k: int) -> List[str]:
        # 1) exact sau khi bỏ TỪ THỪA + phần sau dấu phẩy
        for v in (q, re.sub(r",.*$", "", q).strip(), _DX_MOD.sub("", re.sub(r",.*$", "", q)).strip(" ,")):
            if v and v in self._exact:
                return self._exact[v][:top_k]
        core = _DX_MOD.sub("", re.sub(r",.*$", "", q)).strip(" ,") or q
        # 2) substring (term ⊂ mention / mention ⊂ term)
        sc = self._substring_code(core)
        if sc:
            return [sc]
        # 3) fuzzy NỚI cho bệnh
        hit = process.extractOne(core, self._choices, scorer=fuzz.token_sort_ratio)
        if hit and hit[1] >= _DX_FUZZY:
            return [self._choice_code[hit[2]]]
        return []

    def link(self, text: str, kind: str = "disease", top_k: int = 1) -> List[str]:
        q = normalize_str(text)
        if not q:
            return []
        if q in self._exact:                          # exact TOÀN mention
            return self._exact[q][:top_k]
        qx = normalize_str(_abbrev_expand(q))         # v3: viết tắt -> đầy đủ (exact, an toàn)
        if qx != q and qx in self._exact:
            return self._exact[qx][:top_k]
        if kind == "drug":
            pq = self._product_query(q)               # ưu tiên mã cấp SẢN PHẨM (SCD)
            if pq and pq in self._exact:
                return self._exact[pq][:top_k]
            name = self._canon(self._drug_name(q))    # -> tên hoạt chất (đã map synonym)
            if name and name in self._exact:
                return self._exact[name][:top_k]
            hit = process.extractOne(name or q, self._choices, scorer=fuzz.token_sort_ratio)
            return [self._choice_code[hit[2]]] if hit and hit[1] >= self.threshold else []
        if self.aggressive:
            return self._disease_link(q, top_k)       # nhánh nới (A/B only — leaderboard xấu hơn)
        hit = process.extractOne(q, self._choices, scorer=fuzz.token_sort_ratio)   # v1: fuzzy chặt
        return [self._choice_code[hit[2]]] if hit and hit[1] >= self.threshold else []
