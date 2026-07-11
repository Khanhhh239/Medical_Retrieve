# -*- coding: utf-8 -*-
"""P4 — Suy luận NER: text -> Concept (span+type) GROUNDED (text=raw[s:e])."""
from __future__ import annotations

from typing import List, Optional

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

    @torch.no_grad()
    def predict(self, text: str, max_length: int = 512) -> List[Concept]:
        enc = self.tok(text, return_offsets_mapping=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**inputs).logits[0]
        pred_ids = logits.argmax(-1).tolist()

        out: List[Concept] = []
        for s, e, typ in bio_to_spans(pred_ids, offsets):
            txt = text[s:e]
            if txt.strip():                       # grounded by construction
                out.append(Concept(txt, typ, (s, e)))
        return out
