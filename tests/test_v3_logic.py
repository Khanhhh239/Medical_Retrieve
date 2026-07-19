# -*- coding: utf-8 -*-
"""Test logic v3 (không GPU): abbrev + assertion data prep."""
from datagen.abbrev import expand
from datagen.assert_data import (mark, labels_of, build_examples,
                                 E_OPEN, E_CLOSE, ASSERT_LABELS)
from src.metric.scorer import Concept


def test_abbrev_expand():
    assert expand("tha") == "tăng huyết áp"
    assert expand("nmct") == "nhồi máu cơ tim"
    assert expand("bptnmt") == "bệnh phổi tắc nghẽn mạn tính"
    assert expand("khongphaiviettat") == "khongphaiviettat"   # không có -> giữ


def test_abbrev_in_linker_and_distant():
    from src.link.kb import load_icd10, load_rxnorm
    from src.link.linker import Linker
    from datagen.kb_distant import build_term_index, distant_label
    icd = Linker(load_icd10())
    assert icd.link("THA", "disease") == ["I10"]
    assert icd.link("NMCT", "disease")[0].startswith("I21")
    idx = build_term_index(Linker(load_rxnorm()), icd)
    cs = distant_label("Bệnh nhân THA và NMCT cũ", idx)
    by = {c.text: c for c in cs}
    assert "THA" in by and by["THA"].candidates == ("I10",)


def test_labels_of():
    assert labels_of(["isNegated"]) == [1.0, 0.0, 0.0]
    assert labels_of(["isHistorical", "isFamily"]) == [0.0, 1.0, 1.0]
    assert labels_of([]) == [0.0, 0.0, 0.0]
    assert ASSERT_LABELS == ["isNegated", "isHistorical", "isFamily"]


def test_mark_wraps_entity_with_context():
    raw = "Bệnh nhân không đau ngực khi gắng sức"
    s = raw.index("đau ngực")
    m = mark(raw, s, s + len("đau ngực"))
    assert E_OPEN in m and E_CLOSE in m
    inner = m.split(E_OPEN, 1)[1].split(E_CLOSE, 1)[0].strip()
    assert inner == "đau ngực"
    assert "không" in m.split(E_OPEN, 1)[0]          # cue phủ định trước marker


def test_build_examples_only_assertable():
    raw = "sốt cao, WBC 12"
    cs = [Concept("sốt cao", "TRIỆU_CHỨNG", (0, 7), ("isHistorical",)),
          Concept("WBC", "TÊN_XÉT_NGHIỆM", (9, 12))]
    ex = build_examples([("1", raw, cs)])
    assert len(ex) == 1                              # WBC không assertable -> bỏ
    text, y = ex[0]
    assert y == [0.0, 1.0, 0.0] and "sốt cao" in text
