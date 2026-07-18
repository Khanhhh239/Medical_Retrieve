# -*- coding: utf-8 -*-
"""
v2 — DISTANT SUPERVISION: dùng KB/từ điển làm NHÃN (xương sống của bản thảo v2).

Thay vì tin LLM (nạp rác), ta quét KB thuốc/bệnh + lexicon triệu chứng/XN lên 100 file
THẬT → nhãn có BIÊN SẠCH (đúng chuỗi từ điển, không nuốt ngoặc/cả câu) + candidate điền
sẵn. Đây là bài toán distant-supervision NER (KB + corpus không nhãn).

Match: n-gram từ (longest-first, không chồng lấn), khớp trên chuỗi CHUẨN HOÁ (không dấu,
gộp khoảng trắng) nhưng span trả về là offset RAW → giữ bất biến raw[s:e]==text.
LLM silver (đã khử nhiễu) bù recall cho triệu chứng/kết-quả mà KB không phủ.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from src.metric.scorer import Concept
from src.io.offsets import normalize_str

_WS = re.compile(r"\s+")
_TOK = re.compile(r"\S+")
_EDGE_R = " \t,.;:)]}\"'”’"
_EDGE_L = " \t([{\"'“‘"


def _norm(s: str) -> str:
    return _WS.sub(" ", normalize_str(s)).strip()


def _trim_edge(raw: str, s: int, e: int):
    """Cắt dấu câu rìa (token \\S+ dính ',' '.') để n-gram khớp key sạch, offset vẫn liền."""
    while e > s and raw[e - 1] in _EDGE_R:
        e -= 1
    while s < e and raw[s] in _EDGE_L:
        s += 1
    return s, e


def build_term_index(rx, icd, symptoms=(), labs=(), max_words: int = 8) -> Dict[str, Tuple[str, Optional[str]]]:
    """norm_term -> (type, code). Thuốc lấy tên hoạt chất/brand (bỏ chuỗi có LIỀU, >3 từ)."""
    idx: Dict[str, Tuple[str, Optional[str]]] = {}

    def add(term: str, typ: str, code: Optional[str]):
        k = _norm(term)
        if len(k) >= 3 and len(k.split()) <= max_words and k not in idx:
            idx[k] = (typ, code)

    for code, terms in icd.kb.code_to_terms.items():
        for t in terms:
            add(t, "CHẨN_ĐOÁN", code)
    for code, terms in rx.kb.code_to_terms.items():
        for t in terms:
            if not re.search(r"\d", t) and len(t.split()) <= 3:   # ingredient/brand, bỏ SCD có liều
                add(t, "THUỐC", code)
    for s in symptoms:
        add(s, "TRIỆU_CHỨNG", None)
    for l in labs:
        add(l, "TÊN_XÉT_NGHIỆM", None)
    return idx


def distant_label(raw: str, idx: Dict[str, Tuple[str, Optional[str]]],
                  max_words: int = 8) -> List[Concept]:
    """Gán nhãn 1 văn bản bằng KB. Longest-match, không chồng lấn, grounded."""
    toks = [(m.start(), m.end()) for m in _TOK.finditer(raw)]
    out: List[Concept] = []
    i, n = 0, len(toks)
    while i < n:
        hit = None
        for j in range(min(n, i + max_words), i, -1):        # dài nhất trước
            s, e = _trim_edge(raw, toks[i][0], toks[j - 1][1])
            if e <= s:
                continue
            info = idx.get(_norm(raw[s:e]))
            if info:
                hit = (s, e, info[0], info[1])
                break
        if hit:
            s, e, typ, code = hit
            out.append(Concept(raw[s:e], typ, (s, e), (), (code,) if code else ()))
            while i < n and toks[i][0] < e:                  # nhảy qua token đã khớp
                i += 1
        else:
            i += 1
    return out
