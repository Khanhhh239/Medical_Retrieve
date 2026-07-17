# -*- coding: utf-8 -*-
"""
Client RxNav / RxNorm REST API (https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html).

DÙNG OFFLINE để XÂY KHO MÃ (data prep) — KHÔNG gọi lúc inference (luật thi: inference
self-host, không API ngoài). scripts/build_kb.py gọi cái này rồi lưu cache ra CSV;
pipeline inference chỉ đọc CSV.

Dùng urllib (stdlib) — không thêm dependency.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Optional

BASE = "https://rxnav.nlm.nih.gov/REST"


def _get(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "medical-kb-builder"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_exact(data: dict) -> Optional[str]:
    ids = (data.get("idGroup") or {}).get("rxnormId") or []
    return ids[0] if ids else None


def parse_approx(data: dict) -> Optional[str]:
    cands = (data.get("approximateGroup") or {}).get("candidate") or []
    return cands[0].get("rxcui") if cands else None


def get_all_concepts(ttys, timeout: int = 120):
    """
    getAllConcepts — LẤY TOÀN BỘ concept RxNorm theo term-type (1 call/TTY). Phủ MẠNH
    thuốc mà KHÔNG cần UMLS license. Trả list (rxcui, name, tty).

    TTY hay dùng: IN/PIN (hoạt chất), BN (brand), SCD/SBD (sản phẩm lâm sàng/brand),
    SCDC (thành phần). SCD chứa hàm lượng ('amlodipine 10 MG Oral Tablet') -> khớp mã
    cấp sản phẩm như ví dụ đề (308135).
    """
    tty = "+".join(ttys) if not isinstance(ttys, str) else ttys
    url = f"{BASE}/allconcepts.json?tty={urllib.parse.quote(tty)}"
    data = _get(url, timeout)
    out = []
    for c in (data.get("minConceptGroup") or {}).get("minConcept") or []:
        rxcui, name = c.get("rxcui"), c.get("name")
        if rxcui and name:
            out.append((rxcui, name, c.get("tty", "")))
    return out


def rxcui_for(name: str, timeout: int = 15, sleep: float = 0.05) -> Optional[str]:
    """
    Tên thuốc (có thể kèm hàm lượng) -> RxCUI. Thử EXACT (chuẩn hoá) trước cho tên
    hoạt chất sạch; không được thì APPROXIMATE (chịu 'aspirin 81 mg', brand, sai chính tả).
    Trả None nếu lỗi mạng/không thấy.
    """
    q = urllib.parse.quote((name or "").strip())
    if not q:
        return None
    try:
        c = parse_exact(_get(f"{BASE}/rxcui.json?name={q}&search=1", timeout))
        if c:
            time.sleep(sleep)
            return c
    except Exception:
        pass
    try:
        c = parse_approx(_get(f"{BASE}/approximateTerm.json?term={q}&maxEntries=1", timeout))
        time.sleep(sleep)
        return c
    except Exception:
        return None
