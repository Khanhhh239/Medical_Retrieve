# -*- coding: utf-8 -*-
"""P4 — Suy luận NER: text -> Concept (span+type) GROUNDED (text=raw[s:e]).

W7: file dài hơn max_length token -> CHIA THEO DÒNG (line-based chunking) rồi ghép.
Khái niệm y khoa không vắt qua ranh giới dòng -> chia theo dòng KHÔNG cắt cụt entity,
và các chunk rời nhau nên KHÔNG cần khử trùng lặp.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from ..metric.scorer import Concept
from .labels import bio_to_spans_conf


class NERPredictor:
    def __init__(self, model_dir: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

    def _chunk_by_lines(self, text: str, max_length: int) -> List[Tuple[int, str]]:
        """Gom dòng thành chunk <= budget token. Trả [(char_offset, chunk_text)]."""
        budget = max(8, max_length - 2)                 # chừa [CLS]/[SEP]
        lines, off = [], 0
        for ln in text.splitlines(keepends=True):
            lines.append((off, ln)); off += len(ln)

        chunks: List[Tuple[int, str]] = []
        cur_start, cur_text, cur_tok = None, "", 0
        for start, ln in lines:
            ntok = len(self.tok(ln, add_special_tokens=False)["input_ids"])
            if cur_start is None:
                cur_start = start
            if cur_text and cur_tok + ntok > budget:    # đóng chunk trước
                chunks.append((cur_start, cur_text))
                cur_start, cur_text, cur_tok = start, "", 0
            cur_text += ln; cur_tok += ntok
        if cur_text:
            chunks.append((cur_start, cur_text))
        return chunks or [(0, text)]

    @torch.no_grad()
    def _predict_chunk(self, text: str, max_length: int):
        enc = self.tok(text, return_offsets_mapping=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        probs = self.model(**inputs).logits[0].softmax(-1)
        conf, pred = probs.max(-1)                        # độ tin cậy + nhãn / token
        return bio_to_spans_conf(pred.tolist(), offsets, conf.tolist())

    def predict_with_conf(self, text: str, max_length: int = 512
                          ) -> List[Tuple[Concept, float]]:
        """Chạy model 1 lần, trả [(Concept, confidence)] — để sweep nhiều ngưỡng khỏi chạy lại."""
        out: List[Tuple[Concept, float]] = []
        for base, chunk in self._chunk_by_lines(text, max_length):
            for s, e, typ, cf in self._predict_chunk(chunk, max_length):
                gs, ge = base + s, base + e                 # offset toàn cục
                txt = text[gs:ge]
                if txt.strip():                            # grounded by construction
                    out.append((Concept(txt, typ, (gs, ge)), cf))
        return out

    def predict(self, text: str, max_length: int = 512,
                min_prob: float = 0.0) -> List[Concept]:
        """min_prob>0: BỎ span có confidence trung bình < ngưỡng (cắt span thừa/rác mà
        model không chắc). min_prob=0 -> y hệt bản 34.60."""
        return [c for c, cf in self.predict_with_conf(text, max_length) if cf >= min_prob]
