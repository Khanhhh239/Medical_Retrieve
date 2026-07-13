# -*- coding: utf-8 -*-
"""
P2/S5 — ConText assertion rule-based (§S5 medical.md).

Vì sao rule chứ không model: isHistorical bám section (đo được 361 cue/94 file),
isFamily quá thưa để học (10/7), GT nhiều khả năng cũng sinh bằng ConText. Phủ định
là điểm yếu cố hữu của LLM zero-shot -> giao cho rule.

QUAN TRỌNG: cue khớp trên text CÓ DẤU + RANH GIỚI TỪ \b. Bỏ dấu sẽ làm
'bố/bó/bò' -> 'bo' và 'không' chứa 'ong' => false positive hàng loạt (đã dính).

Chỉ áp cho TRIỆU_CHỨNG / CHẨN_ĐOÁN / THUỐC. Trả subset của
{isNegated, isFamily, isHistorical}, [] nếu không có.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Pattern

import yaml

from ..config import get as _cfg

_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs", "assertion_cues.yaml")

NEG_WINDOW = _cfg("context", "neg_window", 35)
HISTORICAL_SECTIONS = frozenset({"HISTORY_PAST"})
ASSERTABLE = frozenset({"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"})


def _compile(cues: List[str]) -> List[Pattern]:
    # \b ... \b trên text có dấu (re mặc định Unicode: chữ VN là \w)
    return [re.compile(r"\b" + re.escape(c.lower()) + r"\b") for c in cues]


class _Cues:
    def __init__(self, cfg: Dict[str, List[str]]):
        self.negated = _compile(cfg["isNegated"])
        self.negated_raw = [c.lower() for c in cfg["isNegated"]]
        self.neg_exc = [c.lower() for c in cfg.get("negation_exceptions", [])]
        self.family = _compile(cfg["isFamily"])
        self.historical = _compile(cfg["isHistorical"])
        self.hist_headers = [c.lower() for c in cfg.get("historical_section_headers", [])]


def _load(path: str = _CFG_PATH) -> _Cues:
    with open(path, "r", encoding="utf-8") as f:
        return _Cues(yaml.safe_load(f))


_CUES = _load()


def _any(patterns: List[Pattern], text_low: str) -> bool:
    return any(p.search(text_low) for p in patterns)


def leading_negation_len(content: str, cues: Optional[_Cues] = None) -> int:
    """
    Nếu `content` mở đầu bằng cue phủ định ('Không đánh trống ngực'), trả số ký tự
    (cue + khoảng trắng) cần cắt để lấy khái niệm sạch. 0 nếu không / nếu là ngoại lệ.
    """
    cues = cues or _CUES
    low = content.lower()
    if any(low.startswith(ex) for ex in cues.neg_exc):
        return 0
    best = 0
    for cue in cues.negated_raw:
        if low.startswith(cue) and low[len(cue):len(cue) + 1] in (" ", ""):
            best = max(best, len(cue))
    if best == 0:
        return 0
    j = best
    while j < len(content) and content[j].isspace():
        j += 1
    return j


def detect_assertions(concept_type: str,
                      line_text: str,
                      concept_start_in_line: int,
                      *,
                      section: str = "OTHER",
                      section_header: str = "",
                      doc_section: str = "OTHER",
                      negated_by_prefix: bool = False,
                      cues: Optional[_Cues] = None) -> List[str]:
    """concept_start_in_line: offset khái niệm TÍNH TỪ đầu line_text.
    doc_section: section ĐÁNH SỐ cha (vd HISTORY_PAST) -> lan isHistorical xuống
    sub-section (bệnh mãn tính nằm trong 'Tiền sử bệnh' vẫn là historical)."""
    cues = cues or _CUES
    if concept_type not in ASSERTABLE:
        return []

    out: List[str] = []
    line_low = line_text.lower()
    header_low = section_header.lower()

    # isHistorical: section / doc_section cha / header / cue trong line
    hist = (section in HISTORICAL_SECTIONS
            or doc_section in HISTORICAL_SECTIONS
            or any(h in header_low for h in cues.hist_headers)
            or _any(cues.historical, line_low))
    if hist:
        out.append("isHistorical")

    # isNegated: prefix đã cắt, hoặc cue đứng TRƯỚC khái niệm trong cửa sổ
    neg = negated_by_prefix
    if not neg:
        window = line_low[max(0, concept_start_in_line - NEG_WINDOW):concept_start_in_line]
        if _any(cues.negated, window) and not any(ex in window for ex in cues.neg_exc):
            neg = True
    if neg:
        out.append("isNegated")

    # isFamily: cue trong line hoặc header
    if _any(cues.family, line_low) or _any(cues.family, header_low):
        out.append("isFamily")

    return out
