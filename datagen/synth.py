# -*- coding: utf-8 -*-
"""
P3 — Sinh bệnh án synthetic tiếng Việt có nhãn (offset CHÍNH XÁC by-construction).

Mô hình HÓA CẤU TRÚC data thật (17 header, bullet, 5 type, assertion, nhiễu) — ĐỘC
LẬP với baseline.py (không circular). Gold sinh ra thoả raw[start:end]==text tuyệt đối
vì string được bồi từng mảnh và span lấy đúng vị trí bồi.

Dùng cho: (a) dev set để ĐO pipeline (P6), (b) data train NER (P4).
KHÔNG phải phân phối test thật — chỉ in-distribution-ish. Ghi rõ khi báo cáo điểm.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.metric.scorer import Concept
from . import lexicon as L


class NoteBuilder:
    """Bồi text từng mảnh, ghi span khái niệm theo đúng vị trí đã bồi."""

    def __init__(self):
        self._buf: List[str] = []
        self._len = 0
        self.concepts: List[Concept] = []

    def emit(self, text: str):
        self._buf.append(text)
        self._len += len(text)

    def emit_concept(self, text: str, ctype: str,
                     assertions: Tuple[str, ...] = (),
                     candidates: Tuple[str, ...] = ()):
        start = self._len
        self.emit(text)
        self.concepts.append(
            Concept(text, ctype, (start, self._len), tuple(assertions), tuple(candidates))
        )

    def build(self) -> Tuple[str, List[Concept]]:
        return "".join(self._buf), self.concepts


@dataclass
class GenConfig:
    p_negate_symptom: float = 0.15
    p_family_section: float = 0.15
    p_double_space: float = 0.15      # nhiễu khoảng trắng (không đụng span khái niệm)
    p_narrative: float = 0.5          # câu free-text (baseline bỏ qua -> dev trung thực)
    min_symptoms: int = 2
    max_symptoms: int = 6


def _sp(rng: random.Random, cfg: GenConfig) -> str:
    """1 khoảng trắng, thỉnh thoảng đôi (mô phỏng nhiễu — nằm NGOÀI span khái niệm)."""
    return "  " if rng.random() < cfg.p_double_space else " "


def _val(rng: random.Random, lo: float, hi: float) -> str:
    v = round(rng.uniform(lo, hi), rng.choice([1, 2]))
    s = f"{v}"
    if rng.random() < 0.15:            # thỉnh thoảng dùng phẩy thập phân kiểu VN
        s = s.replace(".", ",")
    return s


def generate_note(rng: random.Random, cfg: Optional[GenConfig] = None
                  ) -> Tuple[str, List[Concept]]:
    cfg = cfg or GenConfig()
    b = NoteBuilder()

    # --- Section 1: Tiền sử + thuốc (isHistorical) + bệnh mãn tính ---
    b.emit(rng.choice(L.H_HISTORY) + "\n")
    b.emit(rng.choice(L.SUB_DRUG) + "\n")
    for name, rxcui in rng.sample(L.DRUGS, rng.randint(1, 4)):
        b.emit("- ")
        sig = rng.choice(L.SIGS)
        b.emit_concept(f"{name} {sig}", "THUỐC",
                       assertions=("isHistorical",), candidates=(rxcui,))
        b.emit("\n")
    b.emit(rng.choice(L.SUB_CHRONIC) + "\n")
    for name, icd in rng.sample(L.DISEASES, rng.randint(1, 3)):
        b.emit("- ")
        b.emit_concept(name, "CHẨN_ĐOÁN", assertions=("isHistorical",), candidates=(icd,))
        b.emit("\n")

    # --- (tuỳ chọn) Tiền sử gia đình (isFamily) ---
    if rng.random() < cfg.p_family_section:
        b.emit(rng.choice(L.SUB_FAMILY) + "\n")
        name, icd = rng.choice(L.DISEASES)
        b.emit("- ")
        b.emit_concept(name, "CHẨN_ĐOÁN", assertions=("isFamily",), candidates=(icd,))
        b.emit("\n")

    # --- Section 2: Bệnh sử hiện tại + triệu chứng ---
    b.emit(rng.choice(L.H_PRESENT) + "\n")
    b.emit(rng.choice(L.SUB_REASON))
    syms = rng.sample(L.SYMPTOMS, rng.randint(cfg.min_symptoms, cfg.max_symptoms))
    b.emit(" ")
    for i, s in enumerate(syms[:2]):     # vài triệu chứng ngay sau "Lý do nhập viện:"
        if i:
            b.emit(", ")
        b.emit_concept(s, "TRIỆU_CHỨNG")
    b.emit("\n")
    b.emit(rng.choice(L.SUB_SYMPTOM) + "\n")
    for s in syms:
        b.emit("- ")
        neg = rng.random() < cfg.p_negate_symptom
        if neg:
            b.emit("Không ")                # cue NẰM NGOÀI span khái niệm
            b.emit_concept(s, "TRIỆU_CHỨNG", assertions=("isNegated",))
        else:
            b.emit_concept(s, "TRIỆU_CHỨNG")
        b.emit("\n")

    # --- Câu free-text (KHÔNG bullet): baseline bỏ qua, cần model NER (P4) ---
    if rng.random() < cfg.p_narrative:
        b.emit("Bệnh nhân nhập viện vì tình trạng ")
        b.emit_concept(rng.choice(syms), "TRIỆU_CHỨNG")
        b.emit(" tăng dần trong vài ngày qua.\n")

    # --- Section 3: Đánh giá + xét nghiệm (tên + kết quả) ---
    b.emit(rng.choice(L.H_ASSESS) + "\n")
    b.emit(rng.choice(L.SUB_LAB) + "\n")
    for lname, unit, (lo, hi) in rng.sample(L.LABS, rng.randint(2, 5)):
        b.emit("- ")
        b.emit_concept(lname, "TÊN_XÉT_NGHIỆM")
        b.emit(_sp(rng, cfg))
        val = _val(rng, lo, hi) + (f" {unit}" if unit else "")
        b.emit_concept(val, "KẾT_QUẢ_XÉT_NGHIỆM")
        b.emit("\n")

    return b.build()


def generate_dataset(n: int, seed: int = 0, cfg: Optional[GenConfig] = None):
    rng = random.Random(seed)
    cfg = cfg or GenConfig()
    out = []
    for i in range(n):
        text, concepts = generate_note(rng, cfg)
        out.append((f"synth_{i}", text, concepts))
    return out
