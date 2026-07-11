# -*- coding: utf-8 -*-
"""
Test TOÁN cho metric harness (P0). Mọi giá trị kỳ vọng đều tính tay.
Chạy: pytest tests/test_metric.py -v
"""
import math

import pytest

from src.metric.scorer import (
    Concept, ScorerConfig,
    edit_distance, wer, jaccard, align,
    score_sample, score_dataset,
)


def C(text, typ, start, end, assertions=(), candidates=()):
    return Concept(text=text, type=typ, position=(start, end),
                   assertions=tuple(assertions), candidates=tuple(candidates))


# --------------------------------------------------------------------------- #
#  edit_distance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b,exp", [
    (["a", "b", "c"], ["a", "b", "c"], 0),
    (["a", "b", "c"], ["a", "x", "c"], 1),   # 1 substitution
    (["a", "b", "c"], ["a", "c"], 1),        # 1 deletion
    (["a", "b"], ["a", "b", "c"], 1),        # 1 insertion
    ([], ["a", "b"], 2),
    (["a", "b"], [], 2),
    (list("kitten"), list("sitting"), 3),    # kinh điển
])
def test_edit_distance(a, b, exp):
    assert edit_distance(a, b) == exp


# --------------------------------------------------------------------------- #
#  WER — kể cả tính chất KHÔNG chặn trên 1
# --------------------------------------------------------------------------- #
def test_wer_perfect():
    assert wer("a b c", "a b c") == 0.0


def test_wer_substitution():
    assert wer("a b c", "a x c") == pytest.approx(1 / 3)


def test_wer_deletion():
    assert wer("a b c", "a c") == pytest.approx(1 / 3)


def test_wer_unbounded_above_one():
    # ref 3 từ, hyp thừa 4 từ -> 4 insertion / 3 = 1.333 > 1 (đặc tính phạt over-predict)
    assert wer("a b c", "a b c d e f g") == pytest.approx(4 / 3)


def test_wer_empty_ref():
    assert wer("", "") == 0.0
    assert wer("", "a") == 1.0


# --------------------------------------------------------------------------- #
#  Jaccard — 4 ca rỗng + đa mã
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("g,p,exp", [
    ([], [], 1.0),                       # rỗng/rỗng -> 1
    ([], ["a"], 0.0),                    # gt rỗng, pred có -> 0 (phạt over-predict)
    (["a"], [], 0.0),                    # gt có, pred rỗng -> 0
    (["a", "b"], ["a"], 0.5),            # |∩|=1, |∪|=2
    (["a", "b"], ["a", "b"], 1.0),
    (["K21.0", "K21.9"], ["K21.9"], 0.5),  # đa mã ICD, thiếu 1
    (["a", "b", "c"], ["b", "c", "d"], 0.5),
])
def test_jaccard(g, p, exp):
    assert jaccard(g, p) == pytest.approx(exp)


# --------------------------------------------------------------------------- #
#  Align — cùng type mới ghép
# --------------------------------------------------------------------------- #
def test_align_overlap_same_type():
    gt = [C("sốt", "TRIỆU_CHỨNG", 0, 3)]
    pred = [C("sốt", "TRIỆU_CHỨNG", 0, 3)]
    matched, og, op = align(pred, gt, "overlap")
    assert len(matched) == 1 and not og and not op


def test_align_type_mismatch_no_match():
    gt = [C("sốt", "TRIỆU_CHỨNG", 0, 3)]
    pred = [C("sốt", "CHẨN_ĐOÁN", 0, 3)]
    matched, og, op = align(pred, gt, "overlap")
    assert not matched and len(og) == 1 and len(op) == 1


# --------------------------------------------------------------------------- #
#  Self-score = 1.0 (bất biến then chốt): feed GT làm luôn prediction
# --------------------------------------------------------------------------- #
def _gerd_example():
    # Dựng từ ví dụ đề (rút gọn) — đủ 5 type, có assertion & candidate đa mã.
    return [
        C("ho đờm xanh", "TRIỆU_CHỨNG", 40, 51),
        C("tức ngực", "TRIỆU_CHỨNG", 53, 61),
        C("bệnh trào ngược dạ dày - thực quản", "CHẨN_ĐOÁN", 100, 134,
          candidates=["K21.0", "K21.9"]),
        C("Chlorpheniramine 0.4 MG/ML", "THUỐC", 160, 186,
          assertions=["isHistorical"], candidates=["360047"]),
        C("WBC", "TÊN_XÉT_NGHIỆM", 200, 203),
        C("14,43", "KẾT_QUẢ_XÉT_NGHIỆM", 210, 215),
    ]


@pytest.mark.parametrize("wer_mode", ["aligned", "concat"])
def test_self_score_is_one(wer_mode):
    g = _gerd_example()
    cfg = ScorerConfig(wer_mode=wer_mode)
    ds = score_dataset([(g, g)], cfg)
    assert ds.text_score == pytest.approx(1.0)
    assert ds.assertions_score == pytest.approx(1.0)
    assert ds.candidates_score == pytest.approx(1.0)
    assert ds.final == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
#  Sai type -> 0 cả 3 metric, tính 2 lần (ghi chú đề)
# --------------------------------------------------------------------------- #
def test_wrong_type_zero_all_three_aligned():
    gt = [C("sốt", "TRIỆU_CHỨNG", 0, 3)]
    pred = [C("sốt", "CHẨN_ĐOÁN", 0, 3, candidates=["R50"])]
    ds = score_dataset([(pred, gt)], ScorerConfig(wer_mode="aligned"))
    # aligned: 2 orphan -> WER [1,1] -> text 0
    assert ds.text_score == pytest.approx(0.0)
    assert ds.assertions_score == pytest.approx(0.0)
    assert ds.candidates_score == pytest.approx(0.0)
    assert ds.final == pytest.approx(0.0)


def test_wrong_type_text_divergence_concat():
    # Minh chứng vì sao concat KHÔNG trung thực với ghi chú đề: text vẫn khớp.
    gt = [C("sốt", "TRIỆU_CHỨNG", 0, 3)]
    pred = [C("sốt", "CHẨN_ĐOÁN", 0, 3, candidates=["R50"])]
    ds = score_dataset([(pred, gt)], ScorerConfig(wer_mode="concat"))
    assert ds.text_score == pytest.approx(1.0)      # <-- type-agnostic
    assert ds.assertions_score == pytest.approx(0.0)
    assert ds.candidates_score == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
#  Trọng số candidates theo (len(gt)+1), weighted-mean qua sample
# --------------------------------------------------------------------------- #
def test_candidates_weighting_across_samples():
    # Sample A: chẩn đoán đa mã, đoán thiếu 1 -> J=0.5, weight = len({K1,K2})+1 = 3
    gtA = [C("dx", "CHẨN_ĐOÁN", 0, 2, candidates=["K1", "K2"])]
    prA = [C("dx", "CHẨN_ĐOÁN", 0, 2, candidates=["K2"])]
    # Sample B: thuốc 1 mã, đúng -> J=1.0, weight = 1+1 = 2
    gtB = [C("rx", "THUỐC", 0, 2, candidates=["R1"])]
    prB = [C("rx", "THUỐC", 0, 2, candidates=["R1"])]
    ds = score_dataset([(prA, gtA), (prB, gtB)], ScorerConfig(wer_mode="aligned"))
    # candidates_score = (0.5*3 + 1.0*2) / (3+2) = 3.5/5 = 0.7
    assert ds.candidates_score == pytest.approx(0.7)
    # per-sample check
    assert ds.per_sample[0].candidates == pytest.approx(0.5)
    assert ds.per_sample[0].weight == pytest.approx(3.0)
    assert ds.per_sample[1].candidates == pytest.approx(1.0)
    assert ds.per_sample[1].weight == pytest.approx(2.0)


def test_over_predict_penalized_heavily():
    # gt 1 triệu chứng; pred đúng nó + bịa thêm 1 chẩn đoán có candidate sai.
    gt = [C("sốt", "TRIỆU_CHỨNG", 0, 3)]
    pred = [C("sốt", "TRIỆU_CHỨNG", 0, 3),
            C("ung thư", "CHẨN_ĐOÁN", 10, 17, candidates=["C80"])]
    ds = score_dataset([(pred, gt)], ScorerConfig(wer_mode="aligned"))
    # text: 1 matched (WER0) + 1 orphan_pred (WER1) -> mean 0.5 -> text 0.5
    assert ds.per_sample[0].text == pytest.approx(0.5)
    # candidates: orphan_pred chẩn đoán -> J 0, weight 1 -> cand 0
    assert ds.candidates_score == pytest.approx(0.0)


def test_aligned_wer_capped_per_concept():
    # cặp ghép: gt 1 từ, pred 5 từ -> WER 4.0. cap -> 1.0 -> text 0 (KHÔNG âm)
    gt = [C("sốt", "TRIỆU_CHỨNG", 0, 3)]
    pred = [C("sốt nặng kéo dài nhiều", "TRIỆU_CHỨNG", 0, 3)]
    ds_cap = score_dataset([(pred, gt)], ScorerConfig(wer_mode="aligned", cap_wer=True))
    assert ds_cap.text_score == pytest.approx(0.0)
    ds_raw = score_dataset([(pred, gt)], ScorerConfig(wer_mode="aligned", cap_wer=False))
    assert ds_raw.text_score < 0.0        # không cap -> âm (minh hoạ vì sao cần cap)


def test_final_weighting_formula():
    # kiểm tra đúng 0.3/0.3/0.4 và sắc thái QUAN TRỌNG của candidates_score:
    #   W_i weight theo SAMPLE, không theo concept. Trong 1 sample, J_cand(i) là
    #   TRUNG BÌNH KHÔNG TRỌNG SỐ của jaccard từng concept (giống assertions).
    g = _gerd_example()
    p = [c for c in g]
    p[2] = C(g[2].text, g[2].type, g[2].start, g[2].end,
             candidates=["K21.9"])  # thiếu K21.0 -> J=0.5 cho concept chẩn đoán đó
    ds = score_dataset([(p, g)], ScorerConfig(wer_mode="aligned"))
    assert ds.text_score == pytest.approx(1.0)
    assert ds.assertions_score == pytest.approx(1.0)
    # 1 sample: J_cand = mean(0.5 [chẩn đoán], 1.0 [thuốc]) = 0.75 ; weighting
    #   giữa-sample vô hiệu vì chỉ 1 sample. (Weight chỉ tách bạch khi >1 sample —
    #   xem test_candidates_weighting_across_samples.)
    assert ds.candidates_score == pytest.approx(0.75)
    exp_final = 0.3 * 1.0 + 0.3 * 1.0 + 0.4 * 0.75
    assert ds.final == pytest.approx(exp_final)  # = 0.9
