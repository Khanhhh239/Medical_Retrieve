# -*- coding: utf-8 -*-
"""P1/S1 — Loader: đọc file input thô, giữ nguyên bytes -> str UTF-8 (không sửa)."""
from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    doc_id: str          # "1", "2", ... (tên file không đuôi)
    raw: str             # nội dung THÔ, không chỉnh sửa (hệ quy chiếu offset)
    path: str

    def __len__(self) -> int:
        return len(self.raw)


def load_document(path: str) -> Document:
    # newline="" để KHÔNG dịch \r\n -> \n (giữ offset đúng byte gốc nếu có CRLF)
    with open(path, "r", encoding="utf-8", newline="") as f:
        raw = f.read()
    doc_id = os.path.splitext(os.path.basename(path))[0]
    return Document(doc_id=doc_id, raw=raw, path=path)


def _numkey(path: str):
    m = re.search(r"(\d+)", os.path.basename(path))
    return (int(m.group(1)) if m else 1 << 30, os.path.basename(path))


def load_dataset(input_dir: str) -> List[Document]:
    """Nạp toàn bộ *.txt trong thư mục input, sắp theo số thứ tự file."""
    paths = glob.glob(os.path.join(input_dir, "*.txt"))
    if not paths:
        raise FileNotFoundError(f"Không thấy file .txt nào trong: {input_dir}")
    return [load_document(p) for p in sorted(paths, key=_numkey)]
