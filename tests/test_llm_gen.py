# -*- coding: utf-8 -*-
"""Test parser marker của LLM-gen (grounding-critical). Không cần LLM."""
from datagen.llm_gen import parse_marked
from src.metric.scorer import score_dataset


def test_parse_basic_and_grounding():
    marked = ('Bệnh nhân bị <c type="TRIỆU_CHỨNG">sốt</c> và '
              '<c type="THUỐC" a="isHistorical">aspirin</c>.')
    clean, cs = parse_marked(marked)
    assert clean == "Bệnh nhân bị sốt và aspirin."
    assert len(cs) == 2
    for c in cs:                                # grounding
        assert clean[c.start:c.end] == c.text
    d = {c.type: c for c in cs}
    assert d["TRIỆU_CHỨNG"].text == "sốt"
    assert d["THUỐC"].text == "aspirin"
    assert d["THUỐC"].assertions == ("isHistorical",)


def test_parse_invalid_type_detagged_but_dropped():
    marked = 'X <c type="BLAH">y</c> <c type="THUỐC">aspirin</c>'
    clean, cs = parse_marked(marked)
    assert clean == "X y aspirin"               # vẫn gỡ thẻ khỏi text
    assert len(cs) == 1 and cs[0].text == "aspirin"   # type sai -> bỏ concept


def test_parse_assertion_only_on_eligible_type():
    marked = '<c type="KẾT_QUẢ_XÉT_NGHIỆM" a="isNegated">1.4</c>'
    clean, cs = parse_marked(marked)
    assert cs[0].assertions == ()               # lab không mang assertion


def test_parse_multiline_grounded_selfscore_one():
    marked = ('- <c type="TRIỆU_CHỨNG">khó thở</c>\n'
              '- Không <c type="TRIỆU_CHỨNG" a="isNegated">sốt</c>\n'
              '<c type="TÊN_XÉT_NGHIỆM">kali</c> <c type="KẾT_QUẢ_XÉT_NGHIỆM">3.2</c>')
    clean, cs = parse_marked(marked)
    assert cs
    for c in cs:
        assert clean[c.start:c.end] == c.text
    ds = score_dataset([(cs, cs)])              # gold tự chấm phải 1.0
    assert ds.final == 1.0


def test_parse_no_tags():
    clean, cs = parse_marked("Không có thẻ nào ở đây.")
    assert clean == "Không có thẻ nào ở đây." and cs == []
