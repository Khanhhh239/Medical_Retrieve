# -*- coding: utf-8 -*-
"""
P2 — Baseline structural extractor (training-free, §S3/S4 medical.md).

Khai thác tín hiệu CẤU TRÚC đã đo (79% bullet gán được section; 53% bullet triệu
chứng ngắn <=6 từ). KHÔNG train, KHÔNG dùng KB (candidates = [] — chờ P5).

Precision-first (metric phạt over-predict không chặn trên):
  * Chỉ lấy khái niệm từ BULLET và phần sau ':' của HEADER, trong section đã định type.
  * Section -> type: SYMPTOM->TRIỆU_CHỨNG, DIAGNOSIS->CHẨN_ĐOÁN, DRUG->THUỐC,
    LAB->(TÊN_XÉT_NGHIỆM + KẾT_QUẢ_XÉT_NGHIỆM).
  * BULLET = 1 khái niệm (không tách phẩy — tránh cắt nhầm 'bệnh thận mạn, không đặc hiệu').
  * HEADER-content (sau ':') = list -> tách theo phẩy (đúng kiểu 'sốt, đau').
  * Cắt cue phủ định mở đầu -> isNegated. Thuốc: bỏ đuôi chỉ định 'điều trị ...'.
  * Grounding: mọi span thoả raw[start:end]==text (writer chặn cứng lần cuối).

KHÔNG hardcode tri thức về 100 file test — toàn bộ là hàm thuần của input + config.
KHÔNG xử lý free-narrative (không bullet/section) — đó là việc của model NER (P4).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..io.loader import Document
from ..segment.sections import segment, Segmentation, match_canonical, header_of
from ..parse.numbers import find_numbers
from ..assert_.context import detect_assertions, leading_negation_len
from ..metric.scorer import Concept
from ..config import get as _cfg

SECTION_TYPE = {
    "SYMPTOM": "TRIỆU_CHỨNG",
    "DIAGNOSIS": "CHẨN_ĐOÁN",
    "DRUG": "THUỐC",
}
_BULLET = re.compile(r"^(\s*[-*•]\s*)")
_HEADER_COLON = re.compile(r"^\s*[^:\n]{3,60}?\s*:\s*")
_TRIM_CHARS = " \t\r\n,;.:•-"
_DRUG_INDICATION = re.compile(r"\s+(điều trị|cho|dùng cho|nếu|khi cần)\b.*$", re.I)
_DRUG_PAREN = re.compile(r"\s*\([^)]*\)?\s*$")    # đuôi '(...)' kể cả ngoặc CHƯA đóng
_DRUG_PREFIX = re.compile(r"^\s*(đang dùng|thường dùng|đã dùng|tiếp tục dùng|"
                          r"bắt đầu dùng|tiếp tục|bắt đầu|dùng)\s+", re.I)
_MAX_LEN = _cfg("baseline", "max_len", 120)              # span dài hơn -> coi là câu
_MAX_WORDS = _cfg("baseline", "max_words_symptom", 10)   # triệu chứng/chẩn đoán > N từ -> bỏ
_MAX_WORDS_DRUG = _cfg("baseline", "max_words_drug", 8)
_DRUG_ASCII_RE = re.compile(r"[A-Za-z]{%d,}" % _cfg("baseline", "drug_min_ascii", 4))


def _lines(raw: str):
    off = 0
    for line in raw.splitlines(keepends=True):
        yield off, line.rstrip("\r\n")
        off += len(line)


def _trim(raw: str, s: int, e: int) -> Tuple[int, int, str]:
    while s < e and raw[s] in _TRIM_CHARS:
        s += 1
    while e > s and raw[e - 1] in _TRIM_CHARS:
        e -= 1
    return s, e, raw[s:e]


def _content_span(raw: str, line_start: int, line: str) -> Optional[Tuple[int, int, bool]]:
    """(start, end, is_bullet) của phần nội dung; None nếu không phải bullet/header-':'."""
    mb = _BULLET.match(line)
    if mb:
        return line_start + mb.end(), line_start + len(line), True
    mh = _HEADER_COLON.match(line)
    if mh:
        cs = line_start + mh.end()
        if cs < line_start + len(line):
            return cs, line_start + len(line), False
    return None


def _iter_parts(raw: str, s: int, e: int, split: bool):
    if not split:
        yield _trim(raw, s, e)
        return
    i = s
    for part in re.split(r"[,;]", raw[s:e]):
        ps, pe = i, i + len(part)
        yield _trim(raw, ps, pe)
        i = pe + 1


def _emit_symptom_dx(raw, ctype, s, e, line, line_start, section, header, doc_section, split):
    out = []
    for p0, p1, _ in _iter_parts(raw, s, e, split):
        if p1 <= p0:
            continue
        # 'X: mô tả' -> khái niệm là X (phần TRƯỚC ':'; mô tả sau ':' bỏ đi)
        colon = raw.find(":", p0, p1)
        if colon != -1:
            p1 = colon
        neg_len = leading_negation_len(raw[p0:p1])
        cstart, cend, ctext = _trim(raw, p0 + neg_len, p1)
        # triệu chứng/chẩn đoán là cụm ngắn; quá dài/nhiều từ -> câu văn xuôi (nhiễu) -> bỏ
        if not ctext or len(ctext) > _MAX_LEN or len(ctext.split()) > _MAX_WORDS:
            continue
        asserts = detect_assertions(
            ctype, line, cstart - line_start, section=section,
            section_header=header, doc_section=doc_section,
            negated_by_prefix=(neg_len > 0),
        )
        out.append(Concept(ctext, ctype, (cstart, cend), tuple(asserts), ()))
    return out


def _emit_drug(raw, s, e, line, line_start, section, header, doc_section):
    s, e, txt = _trim(raw, s, e)
    if not txt:
        return []
    mpre = _DRUG_PREFIX.match(txt)             # bỏ đầu 'đang dùng/đã dùng ...'
    if mpre:
        s, e, txt = _trim(raw, s + mpre.end(), e)
    m = _DRUG_INDICATION.search(txt)          # bỏ đuôi chỉ định (là triệu chứng)
    if m:
        s, e, txt = _trim(raw, s, s + m.start())
    mp = _DRUG_PAREN.search(txt)              # bỏ đuôi '(dose decreased...)'
    if mp:
        s, e, txt = _trim(raw, s, s + mp.start())
    if not txt or len(txt) > _MAX_LEN or len(txt.split()) > _MAX_WORDS_DRUG:
        return []
    # tên thuốc thật là Latin/tiếng Anh; loại câu tiếng Việt thuần ('Ăn uống kém',
    # 'Điều trị bắt đầu chạy thận nhân tạo') bị lọt vào mục thuốc
    if not _DRUG_ASCII_RE.search(txt):
        return []
    asserts = detect_assertions("THUỐC", line, s - line_start,
                                section=section, section_header=header,
                                doc_section=doc_section)
    return [Concept(txt, "THUỐC", (s, e), tuple(asserts), ())]


def _emit_lab(raw, s, e):
    """Tách theo ';' rồi mỗi phần 'tên số[đơn vị]' -> tên + giá trị (chỉ khi có số)."""
    out = []
    i = s
    for part in re.split(r";", raw[s:e]):
        ps, pe = i, i + len(part)
        i = pe + 1
        ts, te, txt = _trim(raw, ps, pe)
        if not txt:
            continue
        nums = find_numbers(txt)
        if not nums:
            continue
        first = nums[0]
        ns, ne, name = _trim(raw, ts, ts + first.start)
        if not name or len(name) > 60:
            continue
        out.append(Concept(name, "TÊN_XÉT_NGHIỆM", (ns, ne), (), ()))
        vs, ve, vtxt = _trim(raw, ts + first.start, te)
        if vtxt:
            out.append(Concept(vtxt, "KẾT_QUẢ_XÉT_NGHIỆM", (vs, ve), (), ()))
    return out


def _lab_content_span(raw, line_start, line) -> Optional[Tuple[int, int]]:
    """
    Nội dung lab của 1 dòng. Xử lý ĐÚNG dạng 'WBC:14,43' (dấu ':' là DỮ LIỆU, giữ tên)
    và 'Kết quả xét nghiệm: kali 6.3' (dấu ':' là HEADER canonical -> lấy sau ':').
    """
    mb = _BULLET.match(line)
    start_off = mb.end() if mb else 0
    body = line[start_off:]
    m = re.match(r"^([^:\n]{3,60}?)\s*:\s*", body)
    if m and match_canonical(m.group(1))[0] is not None:   # 'Label:' là header canonical
        inner = start_off + m.end()
        return (line_start + inner, line_start + len(line)) if inner < len(line) else None
    if body.strip():                                        # 'kali 6.3' / 'WBC:14,43'
        return line_start + start_off, line_start + len(line)
    return None


def extract(doc: Document, seg: Optional[Segmentation] = None) -> List[Concept]:
    raw = doc.raw
    seg = seg or segment(raw, doc.doc_id)
    concepts: List[Concept] = []

    for line_start, line in _lines(raw):
        if not line.strip():
            continue
        sp = seg.span_at(line_start)
        section = sp.canonical if sp else "OTHER"
        header = sp.header_text if sp else ""
        doc_section = sp.doc_section if sp else "OTHER"
        h = header_of(line)                         # dòng này có phải TIÊU ĐỀ không?

        if section == "LAB":
            if h is not None and h[0] != "colon":   # bỏ tiêu đề bare/num, giữ 'Label: data'
                continue
            span = _lab_content_span(raw, line_start, line)
            if span:
                concepts += _emit_lab(raw, span[0], span[1])
            continue

        if section not in SECTION_TYPE:
            continue
        ctype = SECTION_TYPE[section]

        if h is not None:                           # dòng là TIÊU ĐỀ
            if h[0] != "colon":                     #   bare/num: chỉ là mốc -> KHÔNG nhả concept
                continue
            mcolon = re.search(r":\s*", line)        #   'Label: dữ liệu' -> lấy phần sau ':'
            cs, ce, split = line_start + mcolon.end(), line_start + len(line), True
            if cs >= ce:
                continue
        else:                                        # dòng DỮ LIỆU (bullet hoặc header-colon)
            cspan = _content_span(raw, line_start, line)
            if cspan is None:
                continue
            cs, ce, is_bullet = cspan
            split = not is_bullet

        if ctype == "THUỐC":
            concepts += _emit_drug(raw, cs, ce, line, line_start,
                                   section, header, doc_section)
        else:
            concepts += _emit_symptom_dx(raw, ctype, cs, ce, line, line_start,
                                         section, header, doc_section, split=split)

    return concepts
