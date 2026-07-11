# -*- coding: utf-8 -*-
"""Test Writer (P2/S7)."""
from src.metric.scorer import Concept
from src.assemble.writer import to_records


def test_grounding_filter_drops_ungrounded():
    raw = "bệnh nhân bị sốt"
    good = Concept("sốt", "TRIỆU_CHỨNG", (raw.index("sốt"), raw.index("sốt") + 3))
    bad = Concept("ho", "TRIỆU_CHỨNG", (0, 2))     # raw[0:2] != "ho"
    recs = to_records([good, bad], raw)
    assert len(recs) == 1
    assert recs[0]["text"] == "sốt"


def test_candidates_key_only_for_dx_and_drug():
    raw = "aspirin sốt K219"
    rx = Concept("aspirin", "THUỐC", (0, 7), (), ("243670",))
    sym = Concept("sốt", "TRIỆU_CHỨNG", (8, 11))
    dx = Concept("K219", "CHẨN_ĐOÁN", (12, 16), (), ("K21.9",))
    recs = {r["type"]: r for r in to_records([rx, sym, dx], raw)}
    assert "candidates" in recs["THUỐC"]
    assert "candidates" in recs["CHẨN_ĐOÁN"]
    assert "candidates" not in recs["TRIỆU_CHỨNG"]


def test_sorted_and_no_dedupe():
    raw = "táo bón rồi lại táo bón"
    j = raw.rindex("táo bón")
    a = Concept("táo bón", "TRIỆU_CHỨNG", (0, 7))
    b = Concept("táo bón", "TRIỆU_CHỨNG", (j, j + 7))
    recs = to_records([b, a], raw)                  # đầu vào đảo thứ tự
    assert len(recs) == 2                            # KHÔNG dedupe
    assert recs[0]["position"][0] < recs[1]["position"][0]   # đã sort
