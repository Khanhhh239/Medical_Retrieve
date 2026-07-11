# -*- coding: utf-8 -*-
"""Test section segmenter (P1/S2)."""
import pytest

from src.segment.sections import segment, match_canonical, OTHER


@pytest.mark.parametrize("header,exp", [
    ("Lý do nhập viện", "SYMPTOM"),
    ("Các triệu chứng hiện tại", "SYMPTOM"),
    ("Các bệnh lý mãn tính", "DIAGNOSIS"),
    ("Chẩn đoán", "DIAGNOSIS"),
    ("Thuốc trước khi nhập viện", "DRUG"),
    ("Kết quả xét nghiệm", "LAB"),
    ("Kết quả chẩn đoán hình ảnh", "LAB"),          # imaging -> LAB, không phải DIAGNOSIS
    ("Tiền sử phẫu thuật / thủ thuật", "PROC"),
])
def test_match_canonical_clean(header, exp):
    canon, conf = match_canonical(header)
    assert canon == exp


def test_match_canonical_nested_longest_wins():
    # 'tien su benh hien tai' phải thắng 'tien su benh'
    assert match_canonical("Tiền sử bệnh hiện tại")[0] == "HISTORY_PRESENT"
    assert match_canonical("Tiền sử bệnh")[0] == "HISTORY_PAST"


def test_match_canonical_fuzzy_typo():
    # 'Bệnh sử  hin tại' (lỗi 'hin'=hiện, dấu cách đôi) vẫn ra HISTORY_PRESENT
    canon, conf = match_canonical("Bệnh sử  hin tại")
    assert canon == "HISTORY_PRESENT"
    assert conf < 1.0        # đi qua nhánh fuzzy


def test_match_canonical_unknown():
    canon, conf = match_canonical("Xin chào bạn hôm nay")
    assert canon is None


def test_segment_assigns_and_covers():
    raw = (
        "1. Tiền sử bệnh\n"
        "Thuốc trước khi nhập viện:\n"
        "- metoprolol 25mg po bid\n"
        "2. Bệnh sử hiện tại\n"
        "Lý do nhập viện: sốt, ho\n"
        "Các triệu chứng hiện tại\n"
        "- sốt\n"
        "3. Đánh giá tại bệnh viện\n"
        "Kết quả xét nghiệm: kali 6.3\n"
    )
    seg = segment(raw, "t")
    # phủ toàn bộ, liên tục
    assert seg.spans[0].char_start == 0
    assert seg.spans[-1].char_end == len(raw)
    for a, b in zip(seg.spans, seg.spans[1:]):
        assert a.char_end == b.char_start

    # section_at trỏ đúng
    idx_metop = raw.index("metoprolol")
    assert seg.section_at(idx_metop) == "DRUG"
    idx_sot = raw.index("- sốt")
    assert seg.section_at(idx_sot) == "SYMPTOM"
    idx_kali = raw.index("kali")
    assert seg.section_at(idx_kali) == "LAB"


def test_segment_star_bullets_dont_break():
    # file 36 dùng bullet '*' — không được nuốt nhầm thành header
    raw = "Kết quả xét nghiệm\n*   creatinin tăng\n*   ast cao\n"
    seg = segment(raw, "t")
    assert seg.section_at(raw.index("creatinin")) == "LAB"
