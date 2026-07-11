# -*- coding: utf-8 -*-
r"""
P1/S1 — Máy offset (nền tảng của BẤT BIẾN §3.1 medical.md).

Ý tưởng then chốt để bất biến `raw[start:end] == text` KHÔNG THỂ SAI:
    normalized view có ĐỘ DÀI BẰNG raw (biến đổi 1:1 mỗi ký tự) => ánh xạ offset
    là IDENTITY. Index i trong `norm` <-> index i trong `raw`.

Nhờ vậy: tìm kiếm không dấu / không phân biệt hoa-thường trên `norm`, nhưng span
tìm được áp thẳng vào `raw` cho ra đúng substring gốc — không lệch một ký tự.

Nhiễu khoảng trắng (dấu cách đôi, xuống dòng) xử lý lúc KHỚP (\s+ linh hoạt),
KHÔNG bằng cách nén text — để giữ identity mapping tuyệt đối.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

# đ/Đ không phân rã NFD -> map tay để giữ 1:1
_DEACCENT_EXTRA = {"đ": "d", "Đ": "D"}


def deaccent_char(c: str) -> str:
    """Trả về đúng 1 ký tự base không dấu cho 1 ký tự vào (giữ 1:1)."""
    if c in _DEACCENT_EXTRA:
        return _DEACCENT_EXTRA[c]
    nfd = unicodedata.normalize("NFD", c)
    base = "".join(ch for ch in nfd if not unicodedata.combining(ch))
    if len(base) == 1:
        return base
    if len(base) == 0:
        return c            # ký tự combining thuần -> giữ nguyên
    return base[0]          # hiếm: ligature -> lấy base đầu, vẫn 1:1


def norm_char(c: str, lowercase: bool = True, deaccent: bool = True) -> str:
    """Chuẩn hoá 1 ký tự -> đúng 1 ký tự (bảo toàn độ dài)."""
    x = c
    if deaccent:
        x = deaccent_char(x)
    if lowercase:
        xl = x.lower()
        x = xl if len(xl) == 1 else x     # guard: lower() không được đổi độ dài
    if len(x) != 1:
        return c                           # guard cuối: luôn 1:1
    return x


def normalize_str(s: str, lowercase: bool = True, deaccent: bool = True) -> str:
    return "".join(norm_char(c, lowercase, deaccent) for c in s)


@dataclass
class Match:
    start: int
    end: int          # nửa mở [start, end)
    raw: str          # == document.raw[start:end] (bất biến)


class CharView:
    """
    View chuẩn hoá cùng độ dài với raw. Dùng để tìm kiếm; span trả về là offset
    trên raw (identity). BẤT BIẾN: len(norm) == len(raw).
    """

    def __init__(self, raw: str, lowercase: bool = True, deaccent: bool = True):
        self.raw = raw
        self.norm = normalize_str(raw, lowercase, deaccent)
        # bất biến sống còn — nếu sai thì mọi offset về sau sai
        assert len(self.norm) == len(self.raw), (
            f"CharView phá vỡ 1:1: len(norm)={len(self.norm)} != len(raw)={len(self.raw)}"
        )

    def _pattern(self, needle: str, ws_flexible: bool, whole_word: bool) -> re.Pattern:
        n = normalize_str(needle)
        if ws_flexible:
            toks = [re.escape(t) for t in n.split()]
            body = r"\s+".join(toks) if toks else re.escape(n)
        else:
            body = re.escape(n)
        if whole_word:
            # biên "từ" gồm chữ Latin, chữ VN có dấu, chữ số
            wc = r"0-9A-Za-zÀ-ỹ"
            body = rf"(?<![{wc}]){body}(?![{wc}])"
        return re.compile(body)

    def find_all(self, needle: str, *, ws_flexible: bool = True,
                 whole_word: bool = False) -> List[Match]:
        """Tìm mọi lần xuất hiện của `needle` (không dấu, không hoa-thường)."""
        if not needle.strip():
            return []
        pat = self._pattern(needle, ws_flexible, whole_word)
        out: List[Match] = []
        for m in pat.finditer(self.norm):
            s, e = m.start(), m.end()
            # cắt khoảng trắng đầu/cuối (entity không bắt đầu/kết thúc bằng space)
            while s < e and self.raw[s].isspace():
                s += 1
            while e > s and self.raw[e - 1].isspace():
                e -= 1
            if s < e:
                out.append(Match(s, e, self.raw[s:e]))
        return out

    def first(self, needle: str, start: int = 0, **kw) -> Optional[Match]:
        for m in self.find_all(needle, **kw):
            if m.start >= start:
                return m
        return None


def is_grounded(raw: str, text: str, position: Tuple[int, int]) -> bool:
    """Kiểm tra BẤT BIẾN §3.1: raw[start:end] == text."""
    s, e = position
    if not (0 <= s <= e <= len(raw)):
        return False
    return raw[s:e] == text
