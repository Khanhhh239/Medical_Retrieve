# -*- coding: utf-8 -*-
"""
P5 — Nạp cơ sở tri thức (KB) cho linking.

Hỗ trợ:
  * CSV seed (code,term) — bundled, chạy được ngay.
  * RxNorm RXNCONSO.RRF (pipe-delimited) — thả file đầy đủ vào data/kb/ để mở rộng.
  * ICD-10 CSV đầy đủ (code,term[,lang]).

Engine linking KHÔNG phụ thuộc kích thước KB — coverage scale theo file bạn cắm vào.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_DIR = os.path.join(ROOT, "data", "kb")


@dataclass
class KB:
    name: str
    code_to_terms: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, code: str, term: str):
        code, term = code.strip(), term.strip()
        if code and term:
            self.code_to_terms.setdefault(code, []).append(term)

    def __len__(self):
        return len(self.code_to_terms)


def load_csv(path: str, name: str) -> KB:
    kb = KB(name)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kb.add(row["code"], row["term"])
    return kb


def load_rxnorm_rrf(path: str,
                    keep_tty=("IN", "PIN", "BN", "SCD", "SBD", "SCDC"),
                    lang="ENG") -> KB:
    """RXNCONSO.RRF: RXCUI=0, LAT=1, SAB=11, TTY=12, STR=14 (pipe-delimited)."""
    kb = KB("rxnorm")
    keep = set(keep_tty)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) < 15:
                continue
            if p[1] != lang or p[12] not in keep:
                continue
            kb.add(p[0], p[14])
    return kb


def _merge(dst: KB, src: KB) -> None:
    for code, terms in src.code_to_terms.items():
        for t in terms:
            dst.add(code, t)


def load_rxnorm(kb_dir: str = KB_DIR) -> KB:
    """Ưu tiên api-cache (RxNav) -> rồi RRF đầy đủ (nếu có) hoặc seed."""
    kb = KB("rxnorm")
    # THỨ TỰ = ƯU TIÊN khi trùng term (mã nạp trước đứng đầu -> linker trả). Nguồn CHÍNH
    # XÁC trước: getAllConcepts / RRF (mã IN/SCD chuẩn) rồi mới tới cache approximate cũ.
    full = os.path.join(kb_dir, "rxnorm_full.csv")   # getAllConcepts (phủ mạnh, chính xác)
    if os.path.exists(full):
        _merge(kb, load_csv(full, "rxnorm"))
    rrf = os.path.join(kb_dir, "RXNCONSO.RRF")
    if os.path.exists(rrf):
        _merge(kb, load_rxnorm_rrf(rrf))
    # cache RxNav approximate CHỈ dùng khi CHƯA có nguồn chính xác (nó có mã sai như
    # omeprazole->esomeprazole -> đừng để nó chèn vào KB đầy đủ).
    if len(kb) == 0:
        api = os.path.join(kb_dir, "rxnorm_api.csv")
        if os.path.exists(api):
            _merge(kb, load_csv(api, "rxnorm"))
    if len(kb) == 0:
        _merge(kb, load_csv(os.path.join(kb_dir, "rxnorm_seed.csv"), "rxnorm"))
    return kb


def load_icd10(kb_dir: str = KB_DIR) -> KB:
    full = os.path.join(kb_dir, "icd10.csv")
    path = full if os.path.exists(full) else os.path.join(kb_dir, "icd10_seed.csv")
    return load_csv(path, "icd10")
