# -*- coding: utf-8 -*-
"""
Sinh data train bằng LLM ≤9B làm ANNOTATOR (không phải extractor — research: encoder
fine-tuned thắng LLM few-shot ở NER; LLM đáng giá nhất ở khâu sinh DATA đa dạng).

LLM sinh bệnh án VN với khái niệm gắn thẻ inline:
    <c type="THUỐC" a="isHistorical">aspirin 81mg</c>
`parse_marked` gỡ thẻ -> text sạch + concept với OFFSET CHÍNH XÁC (grounded
by-construction, giống NoteBuilder). Parser test được ở local; sinh chạy trên Kaggle.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from src.metric.scorer import (Concept, VALID_TYPES, ASSERTION_TYPES, VALID_ASSERTIONS)

_TAG = re.compile(r'<c\s+type="([^"]+)"(?:\s+a="([^"]*)")?\s*>(.*?)</c>', re.S)


def parse_marked(marked: str) -> Tuple[str, List[Concept]]:
    """Text gắn thẻ -> (text sạch, [Concept]) với offset đúng. Bỏ thẻ hỏng/type sai."""
    parts: List[str] = []
    concepts: List[Concept] = []
    pos, last = 0, 0
    for m in _TAG.finditer(marked):
        plain = marked[last:m.start()]
        parts.append(plain); pos += len(plain)
        typ = m.group(1).strip()
        a = m.group(2) or ""
        inner = m.group(3)
        start = pos
        parts.append(inner); pos += len(inner)
        last = m.end()
        if typ in VALID_TYPES and inner.strip():
            asserts = ()
            if typ in ASSERTION_TYPES:
                asserts = tuple(x.strip() for x in a.split(",")
                                if x.strip() in VALID_ASSERTIONS)
            concepts.append(Concept(inner, typ, (start, start + len(inner)), asserts, ()))
    parts.append(marked[last:])
    clean = "".join(parts)
    # chặn cứng grounding (an toàn — về nguyên tắc đã đúng do dựng offset lúc gỡ thẻ)
    concepts = [c for c in concepts if clean[c.start:c.end] == c.text]
    return clean, concepts


# ---- Prompt (few-shot). Đa dạng hoá bằng danh sách bối cảnh khoa/bệnh. ----
SYSTEM = (
    "Bạn là bác sĩ tạo bệnh án MẪU tiếng Việt để huấn luyện AI trích xuất khái niệm y tế. "
    "Gắn thẻ MỌI khái niệm y tế trong bài bằng đúng cú pháp:\n"
    '<c type="LOẠI" a="ASSERTION">cụm khái niệm</c>\n'
    "LOẠI ∈ {TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC}. "
    "a (tuỳ chọn, chỉ cho TRIỆU_CHỨNG/CHẨN_ĐOÁN/THUỐC) ∈ {isNegated, isFamily, isHistorical}, "
    "cách nhau bởi dấu phẩy. Thuốc giữ tên gốc + liều. Viết tự nhiên, có cả câu văn xuôi "
    "lẫn gạch đầu dòng, có thể có lỗi chính tả nhẹ như bệnh án thật. CHỈ trả về bệnh án đã gắn thẻ."
)

_EXAMPLE = (
    'Bệnh nhân nam 65 tuổi, có tiền sử <c type="CHẨN_ĐOÁN" a="isHistorical">tăng huyết áp</c> '
    'và <c type="CHẨN_ĐOÁN" a="isHistorical">đái tháo đường type 2</c>. '
    'Nhập viện vì <c type="TRIỆU_CHỨNG">khó thở</c> tăng dần, kèm <c type="TRIỆU_CHỨNG">ho có đờm</c>. '
    'Không <c type="TRIỆU_CHỨNG" a="isNegated">đau ngực</c>.\n'
    'Thuốc đang dùng:\n'
    '- <c type="THUỐC" a="isHistorical">amlodipine 5mg po daily</c>\n'
    'Kết quả xét nghiệm: <c type="TÊN_XÉT_NGHIỆM">creatinine</c> '
    '<c type="KẾT_QUẢ_XÉT_NGHIỆM">1.4 mg/dL</c>.'
)

SPECIALTIES = ["tim mạch", "hô hấp", "tiêu hoá", "nội tiết", "thận - tiết niệu",
               "thần kinh", "cơ xương khớp", "huyết học", "truyền nhiễm", "cấp cứu",
               "ung bướu", "lão khoa"]


def build_prompt(specialty: str) -> List[dict]:
    """Trả messages (chat) — đa dạng theo chuyên khoa."""
    user = (f"Tạo 1 bệnh án khoa {specialty}, dài 6–15 dòng, đa dạng khái niệm. "
            f"Theo đúng ĐỊNH DẠNG GẮN THẺ như ví dụ:\n\n{_EXAMPLE}")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]
