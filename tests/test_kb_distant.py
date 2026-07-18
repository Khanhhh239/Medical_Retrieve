# -*- coding: utf-8 -*-
"""Test distant-supervision (KB làm nhãn biên sạch, grounded)."""
import types

from datagen.kb_distant import build_term_index, distant_label
from src.io.offsets import is_grounded


def _fake(d):
    return types.SimpleNamespace(kb=types.SimpleNamespace(code_to_terms=d))


def test_distant_label_clean_and_grounded():
    rx = _fake({"1191": ["aspirin"], "6918": ["metoprolol"]})
    icd = _fake({"I10": ["tăng huyết áp"], "E11": ["đái tháo đường type 2"]})
    idx = build_term_index(rx, icd, symptoms=["khó thở"], labs=["WBC"])
    raw = "BN tăng huyết áp, khó thở, dùng aspirin và metoprolol. WBC cao."
    cs = distant_label(raw, idx)
    by = {c.text: c for c in cs}
    # biên sạch + đúng type + candidate điền
    assert "tăng huyết áp" in by and by["tăng huyết áp"].type == "CHẨN_ĐOÁN"
    assert by["tăng huyết áp"].candidates == ("I10",)
    assert "aspirin" in by and by["aspirin"].candidates == ("1191",)
    assert "khó thở" in by and by["khó thở"].type == "TRIỆU_CHỨNG"
    assert "WBC" in by and by["WBC"].type == "TÊN_XÉT_NGHIỆM"
    # mọi span grounded
    assert all(is_grounded(raw, c.text, c.position) for c in cs)


def test_distant_longest_match_no_overlap():
    icd = _fake({"E11": ["đái tháo đường type 2"], "E10": ["đái tháo đường"]})
    rx = _fake({})
    idx = build_term_index(rx, icd)
    raw = "chẩn đoán đái tháo đường type 2 nhiều năm"
    cs = distant_label(raw, idx)
    # phải khớp cụm DÀI (E11), không cắt thành 'đái tháo đường' (E10)
    assert len(cs) == 1 and cs[0].candidates == ("E11",)
    assert cs[0].text == "đái tháo đường type 2"


def test_distant_skips_drug_with_dose_string():
    # thuốc dạng SCD có LIỀU không đưa vào index (tránh 'amlodipine 10 mg oral tablet')
    rx = _fake({"308135": ["amlodipine 10 MG Oral Tablet"], "17767": ["amlodipine"]})
    idx = build_term_index(rx, _fake({}))
    raw = "dùng amlodipine mỗi sáng"
    cs = distant_label(raw, idx)
    assert len(cs) == 1 and cs[0].text == "amlodipine" and cs[0].candidates == ("17767",)
