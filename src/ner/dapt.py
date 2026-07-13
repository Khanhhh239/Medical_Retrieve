# -*- coding: utf-8 -*-
"""
DAPT — Domain-Adaptive Pretraining (MLM tiếp trên văn bản MIỀN thật, KHÔNG nhãn).

Vì sao: NER train trên synthetic sập 0.91->0.25 vì domain gap (từ vựng/văn phong thật
khác data giả). DAPT chạy MLM (masked language modeling) trên chính 100 file test (+
text lâm sàng VN nếu có) để encoder QUEN miền thật TRƯỚC khi fine-tune NER. Đây là đòn
người thắng n2c2 2022 dùng (domain-adaptive pretraining), KHÔNG cần nhãn.

Luồng: dapt(XLM-R) -> models/dapt -> train_ner.py --model models/dapt -> NER.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from typing import List, Sequence

import torch
from transformers import (AutoTokenizer, AutoModelForMaskedLM, TrainingArguments,
                          Trainer, DataCollatorForLanguageModeling)


def make_blocks(texts: Sequence[str], tok, block_size: int = 512) -> List[List[int]]:
    """Nối token của mọi văn bản rồi cắt thành block đều block_size (kiểu chuẩn MLM)."""
    ids: List[int] = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id)
    n = (len(ids) // block_size) * block_size
    blocks = [ids[i:i + block_size] for i in range(0, max(n, block_size), block_size)]
    # nếu văn bản quá ngắn (< 1 block) vẫn giữ 1 block (pad ở collator)
    if not blocks:
        blocks = [ids] if ids else []
    return blocks


class MLMBlocks(torch.utils.data.Dataset):
    def __init__(self, blocks: List[List[int]]):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        return {"input_ids": self.blocks[i]}


def dapt(base_model: str, texts: Sequence[str], out_dir: str,
         epochs: float = 5.0, batch_size: int = 8, block_size: int = 512,
         lr: float = 5e-5, mlm_prob: float = 0.15,
         gradient_checkpointing: bool = True, optim: str = "adafactor") -> str:
    use_cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(base_model)
    if gradient_checkpointing:
        model.config.use_cache = False

    blocks = make_blocks(texts, tok, block_size)
    ds = MLMBlocks(blocks)
    collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=mlm_prob)

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr,
        fp16=use_cuda, gradient_checkpointing=gradient_checkpointing, optim=optim,
        save_strategy="no", logging_steps=25, report_to=[],
    )
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[DAPT] {len(blocks)} block x {block_size} tok -> {out_dir} (CUDA={use_cuda})")
    return out_dir
