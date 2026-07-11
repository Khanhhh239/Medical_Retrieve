# -*- coding: utf-8 -*-
"""
Bổ sung assertion cho các span (dùng cho nhánh NER: model chỉ ra span+type, còn
assertion vẫn do ConText rule quyết — §S5). Baseline đã tự gắn assertion nên không
cần; hàm này để NER-path thành pipeline ĐẦY ĐỦ.
"""
from __future__ import annotations

import bisect
from dataclasses import replace
from typing import List, Tuple

from ..metric.scorer import Concept
from ..segment.sections import segment
from ..assert_.context import detect_assertions


def _line_table(raw: str) -> Tuple[List[int], List[Tuple[int, str]]]:
    starts, lines = [], []
    off = 0
    for line in raw.splitlines(keepends=True):
        starts.append(off)
        lines.append((off, line.rstrip("\r\n")))
        off += len(line)
    return starts, lines


def add_assertions(raw: str, concepts: List[Concept]) -> List[Concept]:
    seg = segment(raw)
    starts, lines = _line_table(raw)
    out: List[Concept] = []
    for c in concepts:
        i = bisect.bisect_right(starts, c.start) - 1
        line_start, line_text = lines[i] if 0 <= i < len(lines) else (0, "")
        sp = seg.span_at(c.start)
        a = detect_assertions(
            c.type, line_text, c.start - line_start,
            section=(sp.canonical if sp else "OTHER"),
            section_header=(sp.header_text if sp else ""),
            doc_section=(sp.doc_section if sp else "OTHER"),
        )
        out.append(replace(c, assertions=tuple(a)))
    return out
