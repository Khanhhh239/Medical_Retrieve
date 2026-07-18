# -*- coding: utf-8 -*-
"""Test khử nhiễu v2: bỏ rác, cắt ngoặc, giữ grounding."""
from datagen.denoise import clean_concepts, is_garbage
from src.metric.scorer import Concept
from src.io.offsets import is_grounded


def _c(raw, text, typ):
    s = raw.index(text)
    return Concept(text, typ, (s, s + len(text)))


def test_is_garbage():
    assert is_garbage("(") and is_garbage("U") and is_garbage("m")
    assert is_garbage("bệnh nội khoa") and is_garbage("bệnh hiện tại")
    assert is_garbage("...")
    assert not is_garbage("khó thở") and not is_garbage("aspirin")


def test_strip_trailing_paren_keeps_grounding():
    raw = "Đang dùng atenolol (uống hôm nay) đều"
    c = _c(raw, "atenolol (uống hôm nay)", "THUỐC")
    out = clean_concepts(raw, [c])
    assert len(out) == 1
    assert out[0].text == "atenolol"
    assert is_grounded(raw, out[0].text, out[0].position)


def test_drop_garbage_and_filler():
    raw = "( bệnh nội khoa nhưng khó thở rõ"
    cs = [_c(raw, "(", "TRIỆU_CHỨNG"),
          _c(raw, "bệnh nội khoa", "CHẨN_ĐOÁN"),
          _c(raw, "khó thở", "TRIỆU_CHỨNG")]
    out = clean_concepts(raw, cs)
    texts = [c.text for c in out]
    assert "khó thở" in texts
    assert "(" not in texts and "bệnh nội khoa" not in texts


def test_dedup_same_span():
    raw = "sốt và sốt"
    s = raw.index("sốt")
    cs = [Concept("sốt", "TRIỆU_CHỨNG", (s, s + 3)),
          Concept("sốt", "TRIỆU_CHỨNG", (s, s + 3))]
    assert len(clean_concepts(raw, cs)) == 1


def test_kb_trim_to_linkable_subspan():
    # KB-trim: 'X (ghi chú)' vốn được cắt ngoặc; kiểm luôn nhánh trim từ cuối
    from src.link.kb import load_rxnorm, load_icd10
    from src.link.linker import Linker
    rx, icd = Linker(load_rxnorm()), Linker(load_icd10())
    raw = "cho metoprolol succinate liều thấp"
    c = _c(raw, "metoprolol succinate liều thấp", "THUỐC")
    out = clean_concepts(raw, [c], linkers=(rx, icd), kb_trim=True)
    assert len(out) == 1
    assert is_grounded(raw, out[0].text, out[0].position)
    assert out[0].text.startswith("metoprolol")     # cắt về sub-span link được
