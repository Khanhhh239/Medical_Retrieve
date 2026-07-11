# -*- coding: utf-8 -*-
"""
P0 — Metric harness (oracle).

Cài đặt CHÍNH XÁC metric của BTC:

    final = 0.30 * text_score + 0.30 * assertions_score + 0.40 * candidates_score

    text_score       = (1/N) * Σ_i (1 - WER(i))
    assertions_score = (1/N) * Σ_i J_assert(i)
    candidates_score = [ Σ_i J_cand(i) * W_i ] / [ Σ_i W_i ]
                        với W_i = Σ_k (len(gt_candidates(k)) + 1)

Quy tắc Jaccard rỗng:
    gt=∅ ∧ pred=∅ -> 1
    gt=∅ ∧ pred≠∅ -> 0
    còn lại        -> |∩| / |∪|

Sai type: concept không align được -> tính 2 lần (orphan GT + orphan pred), mỗi lần 0.

Tài liệu này cố ý để lộ MỌI giả định gây tranh cãi qua `ScorerConfig`
(§2.2 medical.md): 2 biến thể WER, 2 cách align concept. Mặc định chọn cách
đọc literal nhất của công thức; đổi cờ khi BTC xác nhận.

Không phụ thuộc thư viện ngoài (chỉ stdlib) để test toán minh bạch & không lỗi.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

# 5 loại type hợp lệ (spec-defined, được phép hardcode — §3.2 medical.md)
VALID_TYPES = ("TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "CHẨN_ĐOÁN", "THUỐC")
# Chỉ 3 type này mang assertions
ASSERTION_TYPES = frozenset({"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"})
# Chỉ 2 type này mang candidates
CANDIDATE_TYPES = frozenset({"CHẨN_ĐOÁN", "THUỐC"})
VALID_ASSERTIONS = frozenset({"isNegated", "isFamily", "isHistorical"})

WEIGHTS = (0.30, 0.30, 0.40)  # text, assertions, candidates (spec-defined)


# --------------------------------------------------------------------------- #
#  Concept
# --------------------------------------------------------------------------- #
@dataclass
class Concept:
    text: str
    type: str
    position: Tuple[int, int]           # [start, end) — nửa mở
    assertions: Tuple[str, ...] = ()
    candidates: Tuple[str, ...] = ()

    @property
    def start(self) -> int:
        return self.position[0]

    @property
    def end(self) -> int:
        return self.position[1]

    @staticmethod
    def from_dict(d: dict) -> "Concept":
        pos = d.get("position") or [0, 0]
        return Concept(
            text=d.get("text", ""),
            type=d.get("type", ""),
            position=(int(pos[0]), int(pos[1])),
            assertions=tuple(d.get("assertions", []) or []),
            candidates=tuple(d.get("candidates", []) or []),
        )


# --------------------------------------------------------------------------- #
#  Config — mọi giả định gây tranh cãi nằm ở đây
# --------------------------------------------------------------------------- #
@dataclass
class ScorerConfig:
    # WER: "aligned" = WER per-concept (cặp đã align), orphan -> WER 1.0, MỖI concept
    #      cap ở 1.0 (sàn "0 điểm" theo ghi chú đề). Đây là cách ĐÚNG: ghi chú
    #      "sai type -> 0 điểm cả 3 metric kể cả text" CHỈ thành lập khi text chấm
    #      per-concept (concat thì text 'X' vẫn khớp bất kể type). => aligned mặc định.
    #      "concat" = nối text theo position rồi 1 WER/sample (kém trung thực với ghi
    #      chú; giữ lại để đối chiếu). §11-Q1 medical.md.
    wer_mode: str = "aligned"
    cap_wer: bool = True              # cap WER mỗi concept ở 1.0 (sàn 0 điểm/concept)
    # Align concept: "overlap" = cùng type & position giao nhau; "exact_text" =
    #      cùng type & text trùng (bỏ khoảng trắng thừa).
    align_mode: str = "overlap"
    # Chuẩn hoá token khi tính WER. Mặc định False = literal (span khớp -> WER 0).
    wer_lowercase: bool = False
    # Có phạt orphan (concept lệch type / thiếu / thừa) bằng 0 cho assert/cand.
    penalize_orphans: bool = True


# --------------------------------------------------------------------------- #
#  Toán cơ bản
# --------------------------------------------------------------------------- #
def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein trên chuỗi token bất kỳ (sub=del=ins=1). DP O(len_a*len_b)."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost, # substitution
            )
        prev = cur
    return prev[lb]


def _tok(text: str, lowercase: bool) -> List[str]:
    if lowercase:
        text = text.lower()
    return [t for t in re.split(r"\s+", text.strip()) if t]


def wer(ref_text: str, hyp_text: str, lowercase: bool = False) -> float:
    """
    Word Error Rate = edit_distance(ref_words, hyp_words) / len(ref_words).

    KHÔNG chặn trên 1 khi ref khác rỗng (insertions có thể đẩy WER > 1) — đây là
    tính chất metric khiến over-predict bị phạt nặng (§2.1a medical.md).
    Quy ước ref rỗng: 0 nếu hyp cũng rỗng, ngược lại 1.0.
    """
    r = _tok(ref_text, lowercase)
    h = _tok(hyp_text, lowercase)
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def jaccard(gt: Iterable[str], pred: Iterable[str]) -> float:
    """Jaccard với quy tắc rỗng của BTC."""
    g, p = set(gt), set(pred)
    if not g and not p:
        return 1.0
    if not g and p:
        return 0.0
    if g and not p:
        return 0.0
    return len(g & p) / len(g | p)


# --------------------------------------------------------------------------- #
#  Align concept giữa prediction và ground-truth
# --------------------------------------------------------------------------- #
def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).strip().lower())


def _overlap(a: Concept, b: Concept) -> int:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    return max(0, hi - lo)


def align(pred: Sequence[Concept], gt: Sequence[Concept], mode: str = "overlap"
          ) -> Tuple[List[Tuple[Concept, Concept]], List[Concept], List[Concept]]:
    """
    Ghép 1-1 pred<->gt. BẮT BUỘC cùng type (khác type -> không ghép -> orphan,
    hiện thực hoá quy tắc "sai type tính 2 lần").

    Trả về (matched_pairs, orphan_gt, orphan_pred).
    Greedy: overlap lớn nhất trước (mode overlap) / trùng text (mode exact_text).
    """
    used_pred = [False] * len(pred)
    matched: List[Tuple[Concept, Concept]] = []
    orphan_gt: List[Concept] = []

    for g in gt:
        best_j, best_key = -1, None
        for j, p in enumerate(pred):
            if used_pred[j] or p.type != g.type:
                continue
            if mode == "overlap":
                ov = _overlap(g, p)
                if ov > 0:
                    key = ov
                    if best_key is None or key > best_key:
                        best_key, best_j = key, j
            elif mode == "exact_text":
                if _norm_text(p.text) == _norm_text(g.text):
                    # ưu tiên position gần nhất
                    key = -abs(p.start - g.start)
                    if best_key is None or key > best_key:
                        best_key, best_j = key, j
            else:
                raise ValueError(f"align mode không hợp lệ: {mode}")
        if best_j >= 0:
            used_pred[best_j] = True
            matched.append((g, pred[best_j]))
        else:
            orphan_gt.append(g)

    orphan_pred = [p for j, p in enumerate(pred) if not used_pred[j]]
    return matched, orphan_gt, orphan_pred


# --------------------------------------------------------------------------- #
#  Chấm 1 sample
# --------------------------------------------------------------------------- #
@dataclass
class SampleScore:
    text: float          # 1 - WER
    assertions: float     # J_assert(i)
    candidates: float     # J_cand(i)
    weight: float         # W_i cho candidates
    wer: float


def _concat_text(concepts: Sequence[Concept]) -> str:
    ordered = sorted(concepts, key=lambda c: (c.start, c.end))
    return " ".join(c.text for c in ordered)


def score_sample(pred: Sequence[Concept], gt: Sequence[Concept],
                 cfg: Optional[ScorerConfig] = None) -> SampleScore:
    cfg = cfg or ScorerConfig()
    matched, orphan_gt, orphan_pred = align(pred, gt, cfg.align_mode)

    # ---- text (WER) ----
    if cfg.wer_mode == "concat":
        w = wer(_concat_text(gt), _concat_text(pred), cfg.wer_lowercase)
    elif cfg.wer_mode == "aligned":
        def _w(g, p):
            v = wer(g.text, p.text, cfg.wer_lowercase)
            return min(1.0, v) if cfg.cap_wer else v      # sàn 0 điểm/concept
        vals = [_w(g, p) for g, p in matched]
        # orphan: text thiếu/thừa (kể cả sai type) -> WER 1.0 (0 điểm text)
        vals += [1.0] * (len(orphan_gt) + len(orphan_pred))
        w = sum(vals) / len(vals) if vals else 0.0
    else:
        raise ValueError(f"wer_mode không hợp lệ: {cfg.wer_mode}")
    text_score = 1.0 - w

    # ---- assertions (chỉ ASSERTION_TYPES) ----
    a_vals: List[float] = []
    for g, p in matched:
        if g.type in ASSERTION_TYPES:
            a_vals.append(jaccard(g.assertions, p.assertions))
    if cfg.penalize_orphans:
        a_vals += [0.0] * sum(1 for c in orphan_gt if c.type in ASSERTION_TYPES)
        a_vals += [0.0] * sum(1 for c in orphan_pred if c.type in ASSERTION_TYPES)
    j_assert = sum(a_vals) / len(a_vals) if a_vals else 1.0  # vacuous: không có gì để chấm

    # ---- candidates (chỉ CANDIDATE_TYPES) — có weight ----
    c_vals: List[float] = []
    weight = 0.0
    for g, p in matched:
        if g.type in CANDIDATE_TYPES:
            c_vals.append(jaccard(g.candidates, p.candidates))
            weight += len(set(g.candidates)) + 1
    if cfg.penalize_orphans:
        for c in orphan_gt:
            if c.type in CANDIDATE_TYPES:
                c_vals.append(0.0)
                weight += len(set(c.candidates)) + 1
        for c in orphan_pred:
            if c.type in CANDIDATE_TYPES:
                c_vals.append(0.0)
                weight += 1  # gt không tồn tại -> len(gt)=0 -> +1
    j_cand = sum(c_vals) / len(c_vals) if c_vals else 1.0

    return SampleScore(text=text_score, assertions=j_assert,
                       candidates=j_cand, weight=weight, wer=w)


# --------------------------------------------------------------------------- #
#  Chấm cả tập
# --------------------------------------------------------------------------- #
@dataclass
class DatasetScore:
    final: float
    text_score: float
    assertions_score: float
    candidates_score: float
    n: int
    per_sample: List[SampleScore] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "final": self.final,
            "text_score": self.text_score,
            "assertions_score": self.assertions_score,
            "candidates_score": self.candidates_score,
            "n": self.n,
        }


def score_dataset(samples: Sequence[Tuple[Sequence[Concept], Sequence[Concept]]],
                  cfg: Optional[ScorerConfig] = None) -> DatasetScore:
    """samples = list các (pred_concepts, gt_concepts). N cố định = len(samples)."""
    cfg = cfg or ScorerConfig()
    n = len(samples)
    if n == 0:
        return DatasetScore(0.0, 0.0, 0.0, 0.0, 0)

    per = [score_sample(p, g, cfg) for p, g in samples]

    text_score = sum(s.text for s in per) / n
    assertions_score = sum(s.assertions for s in per) / n

    wsum = sum(s.weight for s in per)
    if wsum > 0:
        candidates_score = sum(s.candidates * s.weight for s in per) / wsum
    else:
        candidates_score = 1.0  # không có concept nào cần map

    final = WEIGHTS[0] * text_score + WEIGHTS[1] * assertions_score + WEIGHTS[2] * candidates_score
    return DatasetScore(final, text_score, assertions_score, candidates_score, n, per)


# tiện dụng: chấm từ list[dict] (định dạng JSON output của BTC)
def score_dataset_dicts(samples: Sequence[Tuple[Sequence[dict], Sequence[dict]]],
                        cfg: Optional[ScorerConfig] = None) -> DatasetScore:
    conv = [([Concept.from_dict(x) for x in pred], [Concept.from_dict(x) for x in gt])
            for pred, gt in samples]
    return score_dataset(conv, cfg)
