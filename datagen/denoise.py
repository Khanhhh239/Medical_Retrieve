# -*- coding: utf-8 -*-
"""
v2 — KHỬ NHIỄU nhãn/span (dùng chung cho training-prep và inference).

Triết lý v2: LLM/model hay nạp rác (filler 'bệnh nội khoa', nuốt ngoặc '(uống hôm nay)',
mảnh '(' 'm' 'U', span cả câu). Module này làm sạch span mà VẪN giữ bất biến grounding
raw[s:e]==text:
  - cắt ngoặc đuôi + dấu câu thừa 2 đầu (điều chỉnh offset),
  - bỏ span rác (<=2 ký tự, toàn dấu, cụm filler),
  - KB-trim (tuỳ chọn): thuốc/chẩn đoán cắt về sub-span link được KB,
  - khử trùng lặp + giải chồng lấn.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.metric.scorer import Concept
from src.io.offsets import normalize_str

_PAREN_TAIL = re.compile(r"\s*[\(（\[][^\)）\]]*[\)）\]]\s*$")
_EDGE_L = " \t.,;:-–—)）]"
_EDGE_R = " \t.,;:-–—(（["

# cụm CHUNG CHUNG không phải khái niệm codable (so khớp sau normalize)
_FILLER = {normalize_str(x) for x in [
    "bệnh", "benh", "bệnh nhân", "bệnh lý", "bệnh lí", "bệnh nội khoa",
    "bệnh hiện tại", "bệnh lý bất thường", "bệnh lý mãn tính", "bệnh lý mạn tính",
    "tình trạng", "chẩn đoán", "ẩn đoán", "các", "u", "kết quả", "xét nghiệm",
    "triệu chứng", "điều trị", "hiện tại", "tiền sử", "khám", "theo dõi", "nhập viện",
]}


def _strip_span(raw: str, s: int, e: int) -> Tuple[int, int]:
    """Cắt whitespace 2 đầu + ngoặc đuôi + dấu câu rìa. Trả (s,e) mới (sub-range của cũ)."""
    while s < e and raw[s].isspace():
        s += 1
    while e > s and raw[e - 1].isspace():
        e -= 1
    # cắt lặp các nhóm ngoặc ở đuôi: 'X (a) (b)' -> 'X'
    m = _PAREN_TAIL.search(raw[s:e])
    while m and s + m.start() > s:
        e = s + m.start()
        m = _PAREN_TAIL.search(raw[s:e])
    while e > s and raw[e - 1] in _EDGE_L:
        e -= 1
    while s < e and raw[s] in _EDGE_R:
        s += 1
    return s, e


def is_garbage(text: str) -> bool:
    t = text.strip()
    if len(t) <= 2:                                  # mảnh '(' 'm' 'U' 'bi'
        return True
    if re.fullmatch(r"[\W\d_]+", t):                 # toàn dấu/số
        return True
    if normalize_str(t) in _FILLER:                  # filler chung chung
        return True
    return False


def _kb_trim(raw: str, s: int, e: int, ctype: str, linkers) -> Tuple[int, int]:
    """Thuốc/chẩn đoán: nếu span đầy đủ KHÔNG link được nhưng 1 tiền tố link được -> cắt
    về tiền tố đó (offset vẫn liền mạch). linkers = (rx, icd) hoặc None."""
    if linkers is None:
        return s, e
    rx, icd = linkers
    lk = rx if ctype == "THUỐC" else (icd if ctype == "CHẨN_ĐOÁN" else None)
    if lk is None:
        return s, e
    kind = "drug" if ctype == "THUỐC" else "disease"
    if lk.link(raw[s:e], kind):                      # đã link được -> giữ nguyên
        return s, e
    # thử cắt bớt TỪ CUỐI (giữ tiền tố) tìm sub-span link được
    toks = list(re.finditer(r"\S+", raw[s:e]))
    for k in range(len(toks) - 1, 0, -1):
        e2 = s + toks[k - 1].end()
        if lk.link(raw[s:e2], kind):
            return _strip_span(raw, s, e2)
    return s, e


def clean_concepts(raw: str, concepts: List[Concept],
                   linkers=None, kb_trim: bool = False) -> List[Concept]:
    """Làm sạch danh sách concept. Giữ grounding. kb_trim=True cần linkers=(rx,icd)."""
    out: List[Concept] = []
    seen = set()
    for c in concepts:
        s, e = _strip_span(raw, c.start, c.end)
        if kb_trim:
            s, e = _kb_trim(raw, s, e, c.type, linkers)
            s, e = _strip_span(raw, s, e)
        if e <= s:
            continue
        txt = raw[s:e]
        if is_garbage(txt):
            continue
        key = (s, e, c.type)
        if key in seen:
            continue
        seen.add(key)
        out.append(Concept(txt, c.type, (s, e), c.assertions, c.candidates))
    out.sort(key=lambda x: (x.start, x.end))
    return out
