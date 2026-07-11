# -*- coding: utf-8 -*-
"""Test máy offset (P1/S1) — bất biến 1:1 và grounding."""
import pytest

from src.io.offsets import (
    CharView, deaccent_char, norm_char, normalize_str, is_grounded,
)


@pytest.mark.parametrize("c,exp", [
    ("á", "a"), ("ế", "e"), ("ọ", "o"), ("ữ", "u"), ("ị", "i"),
    ("đ", "d"), ("Đ", "D"), ("A", "A"), (" ", " "), (".", "."), ("5", "5"),
])
def test_deaccent_char_is_1to1(c, exp):
    out = deaccent_char(c)
    assert out == exp
    assert len(out) == 1


@pytest.mark.parametrize("s", [
    "Bệnh nhân bị Tăng huyết áp",
    "amlodipine 10 mg po daily",
    "WBC:14,43; NEUT% (Tỷ lệ):76,4",
    "đánh trống ngực — khó thở",
    "ĐẦY ĐỦ  KHÔNG   DẤU cách",
    "",
])
def test_charview_length_invariant(s):
    v = CharView(s)
    assert len(v.norm) == len(v.raw)   # BẤT BIẾN sống còn


def test_normalize_str_lowercase_deaccent():
    assert normalize_str("Táo Bón") == "tao bon"
    assert normalize_str("Đau Thượng Vị") == "dau thuong vi"


def test_find_all_roundtrip_grounding():
    raw = "Bệnh nhân bị Táo  Bón nặng, kèm ĐAU thượng vị."
    v = CharView(raw)
    # tìm không dấu + không hoa-thường + linh hoạt khoảng trắng (double space)
    ms = v.find_all("tao bon")
    assert len(ms) == 1
    m = ms[0]
    # bất biến: span áp vào raw cho ra đúng chuỗi gốc (giữ hoa + double space)
    assert raw[m.start:m.end] == m.raw == "Táo  Bón"
    # đau thượng vị (viết HOA một phần)
    m2 = v.find_all("dau thuong vi")[0]
    assert raw[m2.start:m2.end] == "ĐAU thượng vị"


def test_find_all_whole_word():
    raw = "metoprololsuccinate và metoprolol riêng"
    v = CharView(raw)
    # whole_word: chỉ khớp token đứng riêng, không khớp bên trong 'metoprololsuccinate'
    ms = v.find_all("metoprolol", whole_word=True)
    assert len(ms) == 1
    assert raw[ms[0].start:ms[0].end] == "metoprolol"


def test_find_all_multiple_occurrences():
    raw = "táo bón rồi lại táo bón"
    v = CharView(raw)
    ms = v.find_all("tao bon")
    assert len(ms) == 2
    for m in ms:
        assert raw[m.start:m.end] == m.raw


def test_is_grounded():
    raw = "abcdef"
    assert is_grounded(raw, "cde", (2, 5))
    assert not is_grounded(raw, "cdX", (2, 5))
    assert not is_grounded(raw, "cde", (2, 6))   # lệch biên
    assert not is_grounded(raw, "x", (10, 11))   # ngoài phạm vi
