# -*- coding: utf-8 -*-
"""
P1/S2 — Section segmenter (config-driven, §3.2/§3.3 medical.md).

Gán mỗi ký tự của tài liệu vào 1 section canonical. Section là feature mạnh nhất
cho type resolution (S4) và isHistorical (S5). Không exact-match (17 header × 90
sub-section, nhiều lỗi chính tả) -> substring (pattern dài nhất thắng) rồi fuzzy.

KHÔNG chỉnh sửa raw, KHÔNG đổi offset — chỉ gắn nhãn khoảng [char_start,char_end).
"""
from __future__ import annotations

import os
import re
import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml
from rapidfuzz import fuzz

from ..io.offsets import normalize_str
from ..config import get as _cfg

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs", "sections.yaml")

FUZZ_THRESHOLD = _cfg("sections", "fuzz_threshold", 88)   # token_sort_ratio tối thiểu (fuzzy)
BARE_CONF = _cfg("sections", "bare_conf", 0.95)           # bare-header phải khớp >= conf này
OTHER = "OTHER"


def _load_config(path: str = _CONFIG_PATH) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # đảm bảo pattern đã bỏ dấu/thường (idempotent)
    return {canon: [normalize_str(p) for p in pats] for canon, pats in cfg.items()}


_CONFIG = _load_config()


SUBSTR_COVERAGE = _cfg("sections", "substr_coverage", 0.60)   # pattern phủ >= tỉ lệ header


def match_canonical(header_text: str, config: Optional[Dict[str, List[str]]] = None
                    ) -> Tuple[Optional[str], float]:
    """
    header_text (thô) -> (canonical, confidence). Thứ tự ưu tiên:
      1. EXACT (h == pattern) — chắc chắn nhất.
      2. Pattern là substring của header VÀ phủ >= 60% header (header có thêm chữ
         thừa) — pattern dài nhất thắng. Ngưỡng phủ chặn 'benh su' cướp
         'benh su hien tai'.
      3. Fuzzy token_sort_ratio TOÀN chuỗi (phạt token thiếu, khác token_set nên
         'benh su' KHÔNG được điểm 100 với 'benh su hien tai').
    Xử lý đúng lồng nhau ('tien su benh hien tai' vs 'tien su benh') và typo.
    """
    config = config or _CONFIG
    h = normalize_str(header_text).strip()
    if not h:
        return None, 0.0

    # tier 1 — exact
    for canon, pats in config.items():
        if h in pats:
            return canon, 1.0

    # tier 2 — pattern ⊂ header, phủ đủ, dài nhất thắng
    best_canon, best_len = None, 0
    for canon, pats in config.items():
        for p in pats:
            if p and p in h and len(p) > best_len:
                best_canon, best_len = canon, len(p)
    if best_canon and best_len >= SUBSTR_COVERAGE * len(h):
        return best_canon, 0.99

    # tier 3 — fuzzy toàn chuỗi
    fz_canon, fz_score = None, 0.0
    for canon, pats in config.items():
        for p in pats:
            s = fuzz.token_sort_ratio(h, p)
            if s > fz_score:
                fz_canon, fz_score = canon, s
    if fz_score >= FUZZ_THRESHOLD:
        return fz_canon, fz_score / 100.0

    # fallback — substring phủ chưa đủ nhưng vẫn là tín hiệu duy nhất
    if best_canon:
        return best_canon, 0.60
    return None, 0.0


@dataclass
class SectionSpan:
    canonical: str
    char_start: int
    char_end: int          # nửa mở
    header_text: str = ""
    doc_section: str = ""  # section ĐÁNH SỐ cha (vd HISTORY_PAST) — cho isHistorical


@dataclass
class Segmentation:
    doc_id: str
    spans: List[SectionSpan] = field(default_factory=list)
    _starts: List[int] = field(default_factory=list)

    def finalize(self):
        self._starts = [s.char_start for s in self.spans]
        return self

    def section_at(self, pos: int) -> str:
        sp = self.span_at(pos)
        return sp.canonical if sp else OTHER

    def span_at(self, pos: int) -> Optional[SectionSpan]:
        if not self.spans:
            return None
        i = bisect.bisect_right(self._starts, pos) - 1
        if i < 0:
            return None
        return self.spans[i]


_HEADER_NUM = re.compile(r"^\s*\d+\s*[.)]\s*(.+?)\s*$")
_SUBLABEL = re.compile(r"^\s*([^:\n]{3,60}?)\s*:\s*(.*)$")
_BULLET = re.compile(r"^\s*[-*•]\s*")


def header_of(content: str, config: Optional[Dict[str, List[str]]] = None):
    """
    Trả (kind, header_text, canonical) nếu dòng là TIÊU ĐỀ MỤC; None nếu là dòng dữ liệu.
    kind ∈ {num, colon, bare}. NHẬN CẢ tiêu đề viết dưới dạng bullet ('* Thuốc trước...')
    — đây là chỗ bug file 36: sub-header là bullet nên trước đây bị bỏ sót.
    """
    config = config or _CONFIG
    m = _HEADER_NUM.match(content)
    if m:                                          # '1. Tiền sử bệnh'
        canon, _ = match_canonical(m.group(1), config)
        return "num", m.group(1), (canon or OTHER)
    mb = _BULLET.match(content)
    body = content[mb.end():] if mb else content   # bỏ dấu bullet rồi mới xét
    m2 = _SUBLABEL.match(body)
    if m2:                                          # 'Label: ...' (khớp canonical mới là header)
        canon, _ = match_canonical(m2.group(1), config)
        return ("colon", m2.group(1), canon) if canon is not None else None
    canon, conf = match_canonical(body.strip(), config)   # bare header (khớp MẠNH)
    if canon is not None and conf >= BARE_CONF:
        return "bare", body.strip(), canon
    return None


def _lines_with_offsets(raw: str) -> List[Tuple[int, str]]:
    """(char_start_của_dòng, nội_dung_dòng_không_gồm_newline)."""
    out, off = [], 0
    for line in raw.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        out.append((off, content))
        off += len(line)
    return out


def segment(raw: str, doc_id: str = "", config: Optional[Dict[str, List[str]]] = None
            ) -> Segmentation:
    """Chia tài liệu thành các section canonical liên tục, phủ toàn bộ ký tự."""
    config = config or _CONFIG
    seg = Segmentation(doc_id=doc_id)
    cur_canon, cur_start, cur_header, cur_doc = OTHER, 0, "", OTHER

    def close(upto: int):
        if upto > cur_start:
            seg.spans.append(SectionSpan(cur_canon, cur_start, upto, cur_header, cur_doc))

    for line_start, content in _lines_with_offsets(raw):
        if not content.strip():
            continue
        h = header_of(content, config)          # nhận cả bullet-header (fix file 36)
        if h is None:
            continue
        kind, header_txt, new_canon = h
        if new_canon is None:
            continue
        close(line_start)
        cur_canon, cur_start, cur_header = new_canon, line_start, header_txt
        if kind == "num":                       # section đánh số -> doc_section cha
            cur_doc = new_canon

    close(len(raw))
    if not seg.spans:
        seg.spans.append(SectionSpan(OTHER, 0, len(raw), ""))
    return seg.finalize()
