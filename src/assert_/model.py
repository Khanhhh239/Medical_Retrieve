# -*- coding: utf-8 -*-
"""
v3 · Khối D — ASSERTION HỌC (encoder + multi-label head), thay ConText regex.

Bài toán: cho ngữ cảnh có marker [E]khái niệm[/E] -> {isNegated, isHistorical, isFamily}
(đa nhãn độc lập -> 3 sigmoid, BCE). Encoder mặc định ViHealthBERT-syllable (kiến thức y
tế VN; assertion KHÔNG cần offset nên word/syllable-seg vô hại). Fallback bất kỳ encoder.

Toán: logits (B,3) -> sigmoid -> ngưỡng 0.5 mỗi nhãn. HF tự dùng BCEWithLogitsLoss khi
problem_type='multi_label_classification'.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dataclasses import replace
from typing import List, Sequence, Tuple

import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

from datagen.assert_data import ASSERT_LABELS, ASSERTABLE, E_OPEN, E_CLOSE, mark

DEFAULT_MODEL = "demdecuong/vihealthbert-base-syllable"


class _DS(torch.utils.data.Dataset):
    def __init__(self, examples, tok, max_length):
        self.ex, self.tok, self.max_length = examples, tok, max_length

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        text, y = self.ex[i]
        enc = self.tok(text, truncation=True, max_length=self.max_length)
        enc["labels"] = [float(v) for v in y]
        return enc


def train(model_name: str, examples: Sequence[Tuple[str, List[float]]], out_dir: str,
          epochs: float = 3.0, batch_size: int = 16, lr: float = 2e-5,
          max_length: int = 256, optim: str = "adamw_torch") -> str:
    use_cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    n_added = tok.add_special_tokens({"additional_special_tokens": [E_OPEN, E_CLOSE]})
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(ASSERT_LABELS),
        problem_type="multi_label_classification")
    if n_added:
        model.resize_token_embeddings(len(tok))

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr,
        fp16=use_cuda, optim=optim, save_strategy="no", logging_steps=50, report_to=[],
    )
    Trainer(model=model, args=args, train_dataset=_DS(examples, tok, max_length),
            data_collator=DataCollatorWithPadding(tok)).train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[assert] train {len(examples)} ví dụ -> {out_dir} (CUDA={use_cuda})")
    return out_dir


class AssertionModel:
    """Suy luận: gán assertion cho concept assertable bằng model (thay ConText)."""

    def __init__(self, model_dir: str, device=None, threshold: float = 0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.threshold = threshold

    @torch.no_grad()
    def predict_texts(self, marked_texts: List[str], batch_size: int = 32,
                      max_length: int = 256) -> List[Tuple[str, ...]]:
        out: List[Tuple[str, ...]] = []
        for i in range(0, len(marked_texts), batch_size):
            enc = self.tok(marked_texts[i:i + batch_size], truncation=True,
                           max_length=max_length, padding=True, return_tensors="pt").to(self.device)
            probs = self.model(**enc).logits.sigmoid().cpu().tolist()
            for p in probs:
                out.append(tuple(ASSERT_LABELS[k] for k, v in enumerate(p) if v >= self.threshold))
        return out

    def annotate(self, raw: str, concepts):
        """Gán assertion cho concept assertable (type khác giữ nguyên)."""
        idx = [i for i, c in enumerate(concepts) if c.type in ASSERTABLE]
        if not idx:
            return list(concepts)
        preds = self.predict_texts([mark(raw, concepts[i].start, concepts[i].end) for i in idx])
        out = list(concepts)
        for j, i in enumerate(idx):
            out[i] = replace(concepts[i], assertions=preds[j])
        return out
