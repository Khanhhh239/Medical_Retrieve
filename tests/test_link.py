# -*- coding: utf-8 -*-
"""Test linker (P5). Kiểm tra CƠ CHẾ retrieve, độc lập độ chính xác mã (mã là data)."""
import os

from src.link.kb import load_csv
from src.link.linker import Linker
from src.link.pipeline import link_concepts
from src.metric.scorer import Concept

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "kb")


def _rx():
    return Linker(load_csv(os.path.join(KB_DIR, "rxnorm_seed.csv"), "rx"))


def _icd():
    return Linker(load_csv(os.path.join(KB_DIR, "icd10_seed.csv"), "icd"))


def test_drug_exact_brand_and_sig_stripping():
    rx = _rx()
    assert rx.link("aspirin", "drug") == ["1191"]
    assert rx.link("lasix 40 mg po daily", "drug") == ["4603"]   # brand + cắt liều
    assert rx.link("tylenol", "drug") == ["161"]                 # brand -> ingredient


def test_disease_exact_and_alias():
    icd = _icd()
    assert icd.link("bệnh trào ngược dạ dày thực quản", "disease") == ["K21.9"]
    assert icd.link("cao huyết áp", "disease") == ["I10"]        # alias -> I10


def test_no_match_returns_empty():
    assert _rx().link("khôngtồntạithuốc", "drug") == []


def test_drug_name_does_not_cut_glued_number():
    # fix #3: cắt liều ở khoảng-trắng+số, KHÔNG cắt số dính trong tên ('b12')
    rx = _rx()
    assert rx._drug_name("aspirin 81 mg po daily") == "aspirin"
    assert rx._drug_name("vitamin b12") == "vitamin b12"


def test_link_concepts_wires_only_dx_and_drug():
    cs = [
        Concept("aspirin 81 mg po daily", "THUỐC", (0, 22)),
        Concept("tăng huyết áp", "CHẨN_ĐOÁN", (23, 36)),
        Concept("sốt", "TRIỆU_CHỨNG", (37, 40)),
    ]
    d = {c.type: c for c in link_concepts(cs)}
    assert d["THUỐC"].candidates == ("1191",)
    assert d["CHẨN_ĐOÁN"].candidates == ("I10",)
    assert d["TRIỆU_CHỨNG"].candidates == ()          # không đụng
