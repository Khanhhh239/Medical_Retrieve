# -*- coding: utf-8 -*-
"""Test ConText assertion (P2/S5)."""
from src.assert_.context import detect_assertions, leading_negation_len


def test_leading_negation_strip():
    assert leading_negation_len("Không đánh trống ngực") == len("Không ")
    assert leading_negation_len("Phủ nhận sốt") == len("Phủ nhận ")
    assert leading_negation_len("đánh trống ngực") == 0


def test_negation_exception_not_stripped():
    # 'không đặc hiệu' = unspecified, KHÔNG phải phủ định
    assert leading_negation_len("không đặc hiệu") == 0


def test_eligible_gate():
    assert detect_assertions("TÊN_XÉT_NGHIỆM", "kali 6.3", 0) == []
    assert detect_assertions("KẾT_QUẢ_XÉT_NGHIỆM", "6.3", 0) == []


def test_historical_by_section():
    a = detect_assertions("TRIỆU_CHỨNG", "đau ngực", 0, section="HISTORY_PAST")
    assert "isHistorical" in a


def test_historical_by_header():
    a = detect_assertions("THUỐC", "metoprolol 25mg", 0, section="DRUG",
                          section_header="Thuốc trước khi nhập viện")
    assert "isHistorical" in a


def test_negation_before_concept():
    line = "Không đánh trống ngực"
    a = detect_assertions("TRIỆU_CHỨNG", line, len("Không "), section="SYMPTOM")
    assert "isNegated" in a


def test_negation_by_prefix_flag():
    a = detect_assertions("TRIỆU_CHỨNG", "khó thở", 0, negated_by_prefix=True)
    assert "isNegated" in a


def test_family_cue():
    line = "bố bệnh nhân bị đau bụng tương tự"
    a = detect_assertions("CHẨN_ĐOÁN", line, line.index("đau"), section="OTHER")
    assert "isFamily" in a


def test_no_false_negation_qualifier_after_concept():
    line = "bệnh thận mạn, không đặc hiệu"
    a = detect_assertions("CHẨN_ĐOÁN", line, 0, section="DIAGNOSIS")
    assert "isNegated" not in a


def test_add_assertions_enrich_from_context():
    # nhánh NER: span+type -> assertion suy từ ngữ cảnh (ConText)
    from src.extract.enrich import add_assertions
    from src.metric.scorer import Concept
    raw = "1. Tiền sử bệnh\nCác bệnh lý mãn tính\n- tăng huyết áp\n"
    s = raw.index("tăng huyết áp")
    c = Concept("tăng huyết áp", "CHẨN_ĐOÁN", (s, s + len("tăng huyết áp")))
    out = add_assertions(raw, [c])
    assert "isHistorical" in out[0].assertions      # nằm dưới 'Tiền sử bệnh'
