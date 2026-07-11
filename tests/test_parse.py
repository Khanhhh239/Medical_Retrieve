# -*- coding: utf-8 -*-
"""Test parser số (P1) — trọng tâm: phẩy thập phân vs phẩy nghìn."""
import pytest

from src.parse.numbers import find_numbers, find_ranges, looks_like_measurement


@pytest.mark.parametrize("tok,kind,val", [
    ("0.01", "decimal", 0.01),
    ("6.3", "decimal", 6.3),
    ("6,3", "decimal", 6.3),        # phẩy thập phân VN (1-2 chữ số sau)
    ("14,43", "decimal", 14.43),    # ví dụ đề
    ("76,4", "decimal", 76.4),
    ("21,000", "thousands", 21000.0),   # phẩy phân cách nghìn (đúng 3 chữ số)
    ("1,234.5", "thousands", 1234.5),
    ("30", "int", 30.0),
    ("12987", "int", 12987.0),
])
def test_classify_single_number(tok, kind, val):
    nums = find_numbers(tok)
    assert len(nums) == 1
    assert nums[0].kind == kind
    assert nums[0].value == pytest.approx(val)


def test_number_span_is_relative_and_exact():
    text = "troponin 0.01"
    n = find_numbers(text)[0]
    assert text[n.start:n.end] == "0.01"


def test_find_numbers_in_lab_line():
    text = "WBC:14,43; NEUT:76,4; LYPH:12,8"
    vals = [n.value for n in find_numbers(text)]
    assert vals == pytest.approx([14.43, 76.4, 12.8])


def test_bnp_thousands_not_decimal():
    # 'bnp 21,000' KHÔNG được hiểu là 21.0
    n = find_numbers("bnp 21,000")[0]
    assert n.kind == "thousands"
    assert n.value == pytest.approx(21000.0)


def test_find_ranges():
    r = find_ranges("lactate 1.1-->0.8")
    assert len(r) == 1
    assert r[0].a == pytest.approx(1.1)
    assert r[0].b == pytest.approx(0.8)


def test_range_ignores_bare_hyphen():
    # '3-4 ngày' KHÔNG phải range (fix #3) — chỉ mũi tên/en-dash mới là range
    assert find_ranges("3-4 ngày") == []
    assert len(find_ranges("1.1–0.8")) == 1        # en-dash vẫn nhận


def test_looks_like_measurement():
    assert looks_like_measurement("- kali 6.3")
    assert looks_like_measurement("EF30")
    assert not looks_like_measurement("đánh trống ngực")
    assert not looks_like_measurement("")
