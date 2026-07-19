# -*- coding: utf-8 -*-
"""
v3 · Khối B — Từ điển VIẾT TẮT lâm sàng VN (giải Semantic Gap).

Bác sĩ viết tắt: THA, ĐTĐ, NMCT... KB chuẩn không có -> distant/linking mù -> bỏ sót.
Map viết-tắt -> cụm ĐẦY ĐỦ; cụm đầy đủ tra KB ra type+mã như thường.

NGUYÊN TẮC (chống bẫy fuzzy):
  - KEY ở dạng ĐÃ CHUẨN HOÁ (normalize_str: chữ thường + bỏ dấu, đ->d) để khớp thẳng.
  - Khớp EXACT theo TOKEN (\\S+), KHÔNG fuzzy -> precision-safe.
  - CHỈ viết tắt >=3 ký tự, KHÔNG mơ hồ. Bỏ 'K'/'VP'/2-ký-tự (đa nghĩa) -> để model quyết.
"""
from __future__ import annotations

# KEY: viết tắt đã normalize (thường + bỏ dấu). VALUE: cụm đầy đủ (có dấu — sẽ normalize khi tra KB).
ABBREV = {
    "tha": "tăng huyết áp",
    "dtd": "đái tháo đường",
    "dtd type 2": "đái tháo đường type 2",
    "dtd typ 2": "đái tháo đường type 2",
    "nmct": "nhồi máu cơ tim",
    "tbmmn": "tai biến mạch máu não",
    "bptnmt": "bệnh phổi tắc nghẽn mạn tính",
    "copd": "bệnh phổi tắc nghẽn mạn tính",
    "rllpm": "rối loạn lipid máu",
    "xhth": "xuất huyết tiêu hóa",
    "hpq": "hen phế quản",
    "bmv": "bệnh mạch vành",
    "dmv": "bệnh động mạch vành",
    "stm": "suy thận mạn",
    "stc": "suy thận cấp",
    "vgb": "viêm gan b",
    "vgc": "viêm gan c",
    "thk": "thoái hóa khớp",
    "vkdt": "viêm khớp dạng thấp",
    "sdd": "suy dinh dưỡng",
    "nkh": "nhiễm khuẩn huyết",
    "ntdtn": "nhiễm trùng đường tiết niệu",
    "taltmc": "tăng áp lực tĩnh mạch cửa",
    "vrt": "viêm ruột thừa",
    "vpq": "viêm phế quản",
    "sxh": "sốt xuất huyết",
    "gerd": "trào ngược dạ dày thực quản",
}


def expand(term_norm: str) -> str:
    """term (ĐÃ normalize) là viết tắt -> cụm đầy đủ (có dấu); else giữ nguyên."""
    return ABBREV.get(term_norm.strip(), term_norm)
