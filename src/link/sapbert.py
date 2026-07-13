# -*- coding: utf-8 -*-
"""
P5+ — SapBERT cross-lingual encoder (W2). ⚠️ EXPERIMENTAL — KHÔNG dùng trong pipeline.

KẾT QUẢ ÂM (đã đo thật): VN→ICD-10 chỉ đúng ~3/20 top-1 (cả [CLS] lẫn mean, và e5 cũng
~4/20). Tên bệnh tiếng Việt ngắn khớp đúng subcode trong 71k mã ICD tiếng Anh quá nhiễu,
tiếng Việt lại thưa trong UMLS -> embedding không align tốt. => KHÔNG wire vào linking;
ICD dùng từ điển VN (exact/fuzzy) chính xác hơn cho bệnh phổ biến. Giữ file này để tài
liệu hoá thử nghiệm + tái dùng nếu sau này fine-tune trên corpus VN.

Ý tưởng gốc: nhúng mô tả ICD tiếng Anh + diagnosis VN vào cùng không gian, cosine gần nhất.

Toán: dùng vector [CLS], CHUẨN HOÁ L2, cosine = dot product. index (N, d) nhân query
(d,) → điểm (N,), argmax.

Model tải từ HuggingFace (offline data-prep/inference-local đều ổn — KHÔNG phải API
ngoài lúc chấm; đây là model self-host ≤9B). Encoder nặng nên lazy import torch.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

DEFAULT_MODEL = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"


class SapBERTEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None,
                 batch_size: int = 64, max_length: int = 32,
                 tokenizer_name: str = "xlm-roberta-base", pooling: str = "cls"):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self._torch = torch
        self.pooling = pooling               # "cls" | "mean"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # SapBERT-XLMR fine-tune TỪ xlm-roberta-base -> CÙNG vocab. Dùng tokenizer của
        # xlm-roberta-base (load fast bình thường) để né bug transformers 5.0 hiểu nhầm
        # sentencepiece của SapBERT thành tiktoken. Token id giống hệt -> embedding đúng.
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """-> ma trận (len(texts), d) đã chuẩn hoá L2 (cosine = dot)."""
        out = []
        torch = self._torch
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state          # (B, T, d)
                if self.pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    rep = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                else:
                    rep = hidden[:, 0, :]                              # [CLS]
                rep = torch.nn.functional.normalize(rep, p=2, dim=1)
            out.append(rep.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 768), np.float32)


class IcdSapBERTLinker:
    """Index (code, term-embedding). link(text) -> mã ICD gần nhất (cosine)."""

    def __init__(self, encoder: SapBERTEncoder, codes: List[str],
                 emb: np.ndarray, min_score: float = 0.0):
        assert len(codes) == emb.shape[0], "codes và emb phải cùng số dòng"
        self.encoder = encoder
        self.codes = codes
        self.emb = emb                      # (N, d) đã chuẩn hoá
        self.min_score = min_score

    @classmethod
    def from_kb(cls, encoder: SapBERTEncoder, code_term_pairs: Sequence,
                cache_path: Optional[str] = None, min_score: float = 0.0):
        """code_term_pairs: list (code, term). Cache embedding ra .npz để khỏi nhúng lại."""
        codes = [c for c, _ in code_term_pairs]
        terms = [t for _, t in code_term_pairs]
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            if list(data["codes"]) == codes:          # cache khớp đúng bộ term
                return cls(encoder, codes, data["emb"], min_score)
        emb = encoder.encode(terms)
        if cache_path:
            np.savez(cache_path, codes=np.array(codes, dtype=object), emb=emb)
        return cls(encoder, codes, emb, min_score)

    def link(self, text: str, top_k: int = 1) -> List[str]:
        if not text.strip() or self.emb.shape[0] == 0:
            return []
        q = self.encoder.encode([text])[0]            # (d,) chuẩn hoá
        sims = self.emb @ q                            # (N,) cosine
        order = np.argsort(-sims)
        out, seen = [], set()
        for i in order:
            if sims[i] < self.min_score:
                break
            c = self.codes[int(i)]
            if c not in seen:
                seen.add(c)
                out.append(c)
            if len(out) >= top_k:
                break
        return out
