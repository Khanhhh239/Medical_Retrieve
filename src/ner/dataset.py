# -*- coding: utf-8 -*-
"""P4 — Encode (text, concepts) -> feature token-classification BIO, giữ offset."""
from __future__ import annotations

from typing import List

import torch

from ..metric.scorer import Concept
from .labels import LABEL2ID


def encode_example(text: str, concepts: List[Concept], tokenizer, max_length: int = 256):
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    max_length=max_length)
    offsets = enc["offset_mapping"]
    labels = ["O"] * len(offsets)

    for c in concepts:
        # token có offset giao [c.start, c.end)
        tok_idx = [i for i, (a, b) in enumerate(offsets) if a != b and a < c.end and b > c.start]
        for j, i in enumerate(tok_idx):
            labels[i] = f"{'B' if j == 0 else 'I'}-{c.type}"

    label_ids = [LABEL2ID[labels[i]] if offsets[i][0] != offsets[i][1] else -100
                 for i in range(len(offsets))]
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": label_ids,
    }


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, items, tokenizer, max_length: int = 256):
        # items: list[(doc_id, text, [Concept])]
        self.data = [encode_example(text, concepts, tokenizer, max_length)
                     for _, text, concepts in items]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
