# -*- coding: utf-8 -*-
"""
P4 — Train NER token-classification (CUDA). §S3a medical.md.

Model mặc định để SMOKE-TEST là nhỏ (distilbert multilingual). Cho THI: đổi
--model sang ViHealthBERT (demdecuong/vihealthbert-base-syllable) — encoder y tế VN
SOTA, hoặc XLM-R. Xem README §Train.
"""
from __future__ import annotations

import os

# giảm phân mảnh VRAM (giúp card nhỏ 6GB) — phải đặt TRƯỚC khi import torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
)

from ..io.jsonl import load_labeled
from .dataset import NERDataset
from .labels import LABELS, LABEL2ID, ID2LABEL, NUM_LABELS


def train(model_name: str,
          train_path: str,
          out_dir: str,
          epochs: float = 3.0,
          batch_size: int = 4,       # nhỏ cho card 6GB; bù bằng grad_accum
          grad_accum: int = 4,       # batch hiệu dụng = batch_size * grad_accum
          lr: float = 3e-5,
          max_length: int = 256,     # giảm để tiết kiệm VRAM (đủ cho phần lớn note)
          gradient_checkpointing: bool = True,   # đánh đổi tính toán lấy VRAM
          optim: str = "adafactor",  # Adafactor: ít VRAM hơn AdamW ~10x (vừa card 6GB)
          eval_path: str = None):
    use_cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
    )

    train_ds = NERDataset(load_labeled(train_path), tok, max_length)
    eval_ds = NERDataset(load_labeled(eval_path), tok, max_length) if eval_path else None
    collator = DataCollatorForTokenClassification(tok)

    if gradient_checkpointing:
        model.config.use_cache = False       # cần khi bật gradient checkpointing

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=lr,
        fp16=use_cuda,                       # RTX 4050: fp16 tiết kiệm VRAM
        optim=optim,
        logging_steps=25,
        save_strategy="no",                  # KHÔNG lưu checkpoint giữa chừng (tiết kiệm disk);
        report_to=[],                        # chỉ lưu model cuối bằng trainer.save_model() bên dưới
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=eval_ds, data_collator=collator)
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[OK] Đã lưu model -> {out_dir} (CUDA={use_cuda})")
    return out_dir
