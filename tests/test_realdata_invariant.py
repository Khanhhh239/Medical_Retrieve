# -*- coding: utf-8 -*-
"""
Test BẤT BIẾN trên 100 file THẬT (bỏ qua nếu chưa giải nén data/test/input).
Đây là 'data-processing test' theo §7 medical.md.
"""
import os
import glob

import pytest

from src.io.loader import load_dataset
from src.io.offsets import CharView, is_grounded
from src.segment.sections import segment

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "test", "input")

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(_DIR, "*.txt")),
    reason="Chưa có data/test/input — giải nén input.zip để chạy test này.",
)


def _docs():
    return load_dataset(_DIR)


def test_charview_1to1_on_all_files():
    for d in _docs():
        v = CharView(d.raw)
        assert len(v.norm) == len(d.raw), f"1:1 vỡ ở {d.doc_id}"


def test_sections_cover_contiguously_all_files():
    for d in _docs():
        seg = segment(d.raw, d.doc_id)
        assert seg.spans[0].char_start == 0
        assert seg.spans[-1].char_end == len(d.raw)
        for a, b in zip(seg.spans, seg.spans[1:]):
            assert a.char_end == b.char_start


def test_find_all_grounded_on_all_files():
    probes = ["khó thở", "metoprolol", "kali", "sốt", "tăng huyết áp"]
    total = 0
    for d in _docs():
        v = CharView(d.raw)
        for needle in probes:
            for m in v.find_all(needle):
                total += 1
                assert d.raw[m.start:m.end] == m.raw
                assert is_grounded(d.raw, m.raw, (m.start, m.end))
    assert total > 0        # phải bắt được ít nhất vài mention có thật
