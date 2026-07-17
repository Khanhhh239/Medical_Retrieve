# -*- coding: utf-8 -*-
"""
Stage B (lõi SOTA) — LLM 32B làm ANNOTATOR trên 100 file THẬT -> silver labels.

Khác llm_gen (sinh note giả): ở đây LLM ĐỌC văn bản thật rồi gắn thẻ, ta GROUND lại
mention vào raw thật (không tin text LLM in ra) + VOTE self-consistency qua N mẫu để
khử nhiễu. Kết quả: (raw thật, concepts grounded trong raw thật) — data đúng miền, đánh
gục domain gap (synthetic từng sập 0.91->0.25).

Toàn bộ hàm ở đây THUẦN LOGIC (không cần GPU) nên test được local. LLM chỉ cấp text.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import List, Sequence, Tuple

from src.metric.scorer import Concept, VALID_TYPES, ASSERTION_TYPES, VALID_ASSERTIONS
from src.io.offsets import CharView
from datagen.llm_gen import parse_marked

_LINE = re.compile(r"[^\n]*(?:\n|$)")

# ---- Prompt chú thích: đọc note thật, chèn thẻ, GIỮ NGUYÊN chữ gốc ----
ANNOTATE_SYSTEM = (
    "Bạn là chuyên gia chú thích bệnh án tiếng Việt cho hệ thống trích xuất khái niệm y tế. "
    "Cho một bệnh án, hãy chèn thẻ đánh dấu MỌI khái niệm y tế, theo đúng cú pháp:\n"
    '<c type="LOẠI" a="ASSERTION">cụm khái niệm</c>\n'
    "LOẠI ∈ {TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC}.\n"
    "a (tuỳ chọn, CHỈ cho TRIỆU_CHỨNG/CHẨN_ĐOÁN/THUỐC) ∈ {isNegated, isFamily, isHistorical}, "
    "nhiều giá trị cách nhau bởi dấu phẩy.\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. GIỮ NGUYÊN từng ký tự của bệnh án gốc (kể cả lỗi chính tả, dính từ, dấu cách). "
    "CHỈ được chèn thẻ <c ...> và </c>, TUYỆT ĐỐI không sửa/thêm/bớt chữ.\n"
    "2. Cụm trong thẻ phải là ĐÚNG chuỗi con xuất hiện trong bệnh án.\n"
    "3. isNegated nếu bị phủ định (không, chưa, loại trừ); isHistorical nếu tiền sử/đã từng; "
    "isFamily nếu của người thân.\n"
    "CHỈ trả về bệnh án đã chèn thẻ, không giải thích."
)

# ví dụ ngắn để LLM bám định dạng (dùng chung với llm_gen._EXAMPLE tinh thần)
_ANN_EXAMPLE_IN = (
    "Tiền sử tăng huyết áp. Vào viện vì khó thở, không sốt.\n"
    "creatinine 1.4 mg/dL. Đang dùng amlodipine 5mg."
)
_ANN_EXAMPLE_OUT = (
    'Tiền sử <c type="CHẨN_ĐOÁN" a="isHistorical">tăng huyết áp</c>. '
    'Vào viện vì <c type="TRIỆU_CHỨNG">khó thở</c>, '
    'không <c type="TRIỆU_CHỨNG" a="isNegated">sốt</c>.\n'
    '<c type="TÊN_XÉT_NGHIỆM">creatinine</c> '
    '<c type="KẾT_QUẢ_XÉT_NGHIỆM">1.4 mg/dL</c>. '
    'Đang dùng <c type="THUỐC" a="isHistorical">amlodipine 5mg</c>.'
)


def build_annotate_prompt(note: str) -> List[dict]:
    """messages (chat) yêu cầu LLM gắn thẻ CHÍNH note này."""
    user = (f"Ví dụ:\nĐẦU VÀO:\n{_ANN_EXAMPLE_IN}\n\nĐẦU RA:\n{_ANN_EXAMPLE_OUT}\n\n"
            f"Giờ chú thích bệnh án sau, GIỮ NGUYÊN chữ gốc:\nĐẦU VÀO:\n{note}\n\nĐẦU RA:\n")
    return [{"role": "system", "content": ANNOTATE_SYSTEM},
            {"role": "user", "content": user}]


def annotate_to_concepts(raw: str, llm_output: str) -> List[Concept]:
    """
    1 mẫu LLM -> concepts GROUNDED trong `raw` thật.

    Không tin text LLM: gỡ thẻ lấy (mention, type, assertions), rồi tìm mention trong
    raw bằng con trỏ TIẾN (giữ thứ tự, phân biệt lần xuất hiện lặp). raw[s:e] là text.
    """
    _, cs = parse_marked(llm_output)
    view = CharView(raw)
    cursor = 0
    out: List[Concept] = []
    for c in cs:
        m = view.first(c.text, start=cursor, ws_flexible=True)   # ưu tiên sau con trỏ
        if m is None:
            m = view.first(c.text, start=0, ws_flexible=True)     # fallback: từ đầu
        if m is None:
            continue                                              # LLM bịa chữ -> bỏ
        asserts = tuple(a for a in c.assertions if a in VALID_ASSERTIONS) \
            if c.type in ASSERTION_TYPES else ()
        out.append(Concept(m.raw, c.type, (m.start, m.end), asserts, ()))
        cursor = m.end
    return out


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def vote(raw: str, samples: Sequence[str], min_votes: int = 2) -> List[Concept]:
    """
    Self-consistency: gộp N mẫu LLM, giữ span xuất hiện >= min_votes lần. Assertion lấy
    theo đa số. Giải chồng lấn: span nhiều phiếu hơn (rồi dài hơn) thắng.
    """
    votes: Counter = Counter()                      # (s,e,type) -> số mẫu
    asserts: dict = defaultdict(Counter)            # (s,e,type) -> Counter[assert-tuple]
    text_of: dict = {}
    for s in samples:
        seen = set()
        for c in annotate_to_concepts(raw, s):
            key = (c.start, c.end, c.type)
            if key in seen:                         # mỗi mẫu tính 1 phiếu / span
                continue
            seen.add(key)
            votes[key] += 1
            asserts[key][tuple(sorted(c.assertions))] += 1
            text_of[key] = c.text

    kept = [(k, v) for k, v in votes.items() if v >= min_votes]
    # nhiều phiếu trước, span dài hơn trước (ổn định thứ tự)
    kept.sort(key=lambda kv: (-kv[1], -(kv[0][1] - kv[0][0]), kv[0][0]))

    chosen: List[Concept] = []
    taken: List[Tuple[int, int]] = []
    for (s, e, t), _v in kept:
        if any(_overlap((s, e), sp) for sp in taken):   # bỏ span chồng span đã chọn
            continue
        a = asserts[(s, e, t)].most_common(1)[0][0]
        chosen.append(Concept(text_of[(s, e, t)], t, (s, e), a, ()))
        taken.append((s, e))
    chosen.sort(key=lambda c: c.start)                  # đọc theo thứ tự văn bản
    return chosen


def _line_spans(raw: str) -> List[Tuple[int, int]]:
    """(start,end) mỗi dòng KỂ CẢ '\\n' cuối — giữ offset đúng dù CRLF."""
    return [(m.start(), m.end()) for m in _LINE.finditer(raw) if m.end() > m.start()]


def chunk_labeled(raw: str, concepts: Sequence[Concept],
                  target_chars: int = 1200) -> List[Tuple[str, List[Concept]]]:
    """
    Cắt tài liệu dài -> nhiều window theo DÒNG (không vắt qua dòng), offset RE-BASE về
    0 của window. Dùng để silver (file thật dài) không bị trainer cắt cụt ở max_length.
    Chỉ giữ window có >=1 concept (nhãn miền thật là phần giá trị nhất). Bất biến giữ:
    sub[c.start:c.end] == c.text.
    """
    spans = _line_spans(raw)
    if not spans:
        return [(raw, list(concepts))] if (raw.strip() and concepts) else []
    windows: List[Tuple[int, int]] = []
    i, n = 0, len(spans)
    while i < n:
        w0 = spans[i][0]
        j = i
        while j < n and spans[j][1] - w0 <= target_chars:
            j += 1
        j = max(j, i + 1)                                # tối thiểu 1 dòng
        windows.append((w0, spans[j - 1][1]))
        i = j
    out: List[Tuple[str, List[Concept]]] = []
    for w0, w1 in windows:
        sub = raw[w0:w1]
        cs = [Concept(c.text, c.type, (c.start - w0, c.end - w0), c.assertions, ())
              for c in concepts if w0 <= c.start and c.end <= w1]
        if cs:
            out.append((sub, cs))
    return out
