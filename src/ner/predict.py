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
from .labels import bio_to_spans


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
    def _predict_chunk(self, text: str, max_length: int) -> List[Tuple[int, int, str]]:
        enc = self.tok(text, return_offsets_mapping=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        pred_ids = self.model(**inputs).logits[0].argmax(-1).tolist()
        return bio_to_spans(pred_ids, offsets)

    def predict(self, text: str, max_length: int = 512) -> List[Concept]:
        out: List[Concept] = []
        for base, chunk in self._chunk_by_lines(text, max_length):
            for s, e, typ in self._predict_chunk(chunk, max_length):
                gs, ge = base + s, base + e             # offset toàn cục
                txt = text[gs:ge]
                if txt.strip():                          # grounded by construction
                    out.append(Concept(txt, typ, (gs, ge)))
        return out
