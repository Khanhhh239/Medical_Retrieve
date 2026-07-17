# -*- coding: utf-8 -*-
"""Test lõi Stage B: ground mention vào raw thật + vote self-consistency (không cần GPU)."""
from datagen.llm_annotate import annotate_to_concepts, vote, chunk_labeled
from src.metric.scorer import Concept
from src.io.offsets import is_grounded


def _all_grounded(raw, concepts):
    return all(is_grounded(raw, c.text, c.position) for c in concepts)


def test_ground_basic_and_assertion():
    raw = "Vào viện vì khó thở, không sốt."
    llm = ('Vào viện vì <c type="TRIỆU_CHỨNG">khó thở</c>, '
           'không <c type="TRIỆU_CHỨNG" a="isNegated">sốt</c>.')
    cs = annotate_to_concepts(raw, llm)
    assert _all_grounded(raw, cs)
    by = {c.text: c for c in cs}
    assert "khó thở" in by and "sốt" in by
    assert by["khó thở"].assertions == ()
    assert by["sốt"].assertions == ("isNegated",)


def test_forward_cursor_disambiguates_repeats():
    raw = "sốt cao, sau đó hết sốt hẳn"
    llm = ('<c type="TRIỆU_CHỨNG">sốt</c> cao, sau đó hết '
           '<c type="TRIỆU_CHỨNG">sốt</c> hẳn')
    cs = annotate_to_concepts(raw, llm)
    assert _all_grounded(raw, cs)
    starts = sorted(c.start for c in cs)
    assert len(starts) == 2 and starts[0] == 0 and starts[1] > 0   # 2 lần khác vị trí


def test_drop_hallucinated_mention():
    raw = "chỉ có khó thở"
    llm = ('<c type="THUỐC">aspirin</c> và <c type="TRIỆU_CHỨNG">khó thở</c>')
    cs = annotate_to_concepts(raw, llm)
    assert [c.text for c in cs] == ["khó thở"]                     # aspirin không có -> bỏ


def test_ground_is_case_accent_insensitive():
    raw = "Tiền sử TĂNG HUYẾT ÁP nhiều năm"
    llm = 'Tiền sử <c type="CHẨN_ĐOÁN" a="isHistorical">tăng huyết áp</c> nhiều năm'
    cs = annotate_to_concepts(raw, llm)
    assert len(cs) == 1
    assert cs[0].text == "TĂNG HUYẾT ÁP"        # lấy đúng chuỗi RAW, không phải chữ LLM
    assert is_grounded(raw, cs[0].text, cs[0].position)


def test_vote_drops_singleton_keeps_majority():
    raw = "bệnh nhân khó thở và ho"
    a = '<c type="TRIỆU_CHỨNG">khó thở</c>'
    both = '<c type="TRIỆU_CHỨNG">khó thở</c> và <c type="TRIỆU_CHỨNG">ho</c>'
    cs = vote(raw, [both, a, a], min_votes=2)
    texts = {c.text for c in cs}
    assert "khó thở" in texts        # 3 phiếu (both,a,a)
    assert "ho" not in texts         # 1 phiếu (chỉ both) -> loại


def test_vote_assertion_by_majority():
    raw = "không đau ngực"
    neg = '<c type="TRIỆU_CHỨNG" a="isNegated">đau ngực</c>'
    pos = '<c type="TRIỆU_CHỨNG">đau ngực</c>'
    cs = vote(raw, [neg, neg, pos], min_votes=2)
    assert len(cs) == 1
    assert cs[0].assertions == ("isNegated",)     # 2/3 phiếu isNegated


def test_vote_overlap_longer_wins_on_tie():
    raw = "chẩn đoán sốt cao chưa rõ"
    short = '<c type="TRIỆU_CHỨNG">sốt</c>'
    long = '<c type="CHẨN_ĐOÁN">sốt cao</c>'
    cs = vote(raw, [short, short, long, long], min_votes=2)   # hoà 2-2, chồng nhau
    assert len(cs) == 1
    assert cs[0].text == "sốt cao" and cs[0].type == "CHẨN_ĐOÁN"


def test_chunk_labeled_rebase_and_ground():
    # 3 dòng; concept ở dòng 1 và dòng 3; target nhỏ -> mỗi dòng 1 window
    raw = "sốt cao\nkhám bình thường\ndùng aspirin 81mg\n"
    concepts = [
        Concept("sốt cao", "TRIỆU_CHỨNG", (0, 7)),
        Concept("aspirin 81mg", "THUỐC", (raw.index("aspirin"), raw.index("aspirin") + 12)),
    ]
    assert all(is_grounded(raw, c.text, c.position) for c in concepts)   # tiền đề
    chunks = chunk_labeled(raw, concepts, target_chars=12)
    assert len(chunks) == 2                          # dòng 2 không có concept -> bỏ
    for sub, cs in chunks:
        for c in cs:
            assert is_grounded(sub, c.text, c.position)   # offset re-base đúng trong window
    assert chunks[0][1][0].position == (0, 7)        # window dòng 1: offset về 0


def test_chunk_labeled_groups_lines_under_target():
    raw = "a khó thở b\nc ho d\ne sốt f\n"
    concepts = [Concept("khó thở", "TRIỆU_CHỨNG", (2, 9))]
    assert is_grounded(raw, concepts[0].text, concepts[0].position)
    chunks = chunk_labeled(raw, concepts, target_chars=1000)   # gộp hết vào 1 window
    assert len(chunks) == 1
    sub, cs = chunks[0]
    assert is_grounded(sub, cs[0].text, cs[0].position)
