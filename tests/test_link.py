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


def test_icd_vi_covers_common_diseases():
    from src.link.kb import load_icd10
    icd = Linker(load_icd10())
    assert len(icd.kb) > 1000                       # danh mục VN (Bộ Y tế) đã nạp
    assert icd.link("tăng huyết áp", "disease")[0] == "I10"
    assert icd.link("đái tháo đường type 2", "disease")[0] == "E11"
    assert icd.link("nhồi máu cơ tim", "disease")[0].startswith("I21")
    assert icd.link("xơ gan", "disease")[0].startswith("K7")


def test_drug_name_does_not_cut_glued_number():
    # fix #3: cắt liều ở khoảng-trắng+số, KHÔNG cắt số dính trong tên ('b12')
    rx = _rx()
    assert rx._drug_name("aspirin 81 mg po daily") == "aspirin"
    assert rx._drug_name("vitamin b12") == "vitamin b12"


def test_synonym_and_product_query():
    rx = _rx()                                  # helper thuần logic, không đụng KB
    assert rx._canon("paracetamol") == "acetaminophen"
    assert rx._canon("salbutamol") == "albuterol"
    assert rx._canon("amlodipine") == "amlodipine"          # không có synonym -> giữ
    assert rx._product_query("amlodipine 10mg") == "amlodipine 10 mg oral tablet"
    assert rx._product_query("paracetamol 500 mg") == "acetaminophen 500 mg oral tablet"
    assert rx._product_query("amlodipine") is None          # không liều -> None


def test_link_concepts_wires_only_dx_and_drug():
    cs = [
        Concept("aspirin 81 mg po daily", "THUỐC", (0, 22)),
        Concept("tăng huyết áp", "CHẨN_ĐOÁN", (23, 36)),
        Concept("sốt", "TRIỆU_CHỨNG", (37, 40)),
    ]
    d = {c.type: c for c in link_concepts(cs)}
    # kiểm WIRING (mã cụ thể là data, đổi theo KB): THUỐC nhận 1 RxCUI (số),
    # CHẨN_ĐOÁN nhận ICD I10, TRIỆU_CHỨNG KHÔNG đụng.
    assert len(d["THUỐC"].candidates) == 1 and d["THUỐC"].candidates[0].isdigit()
    assert d["CHẨN_ĐOÁN"].candidates == ("I10",)
    assert d["TRIỆU_CHỨNG"].candidates == ()          # không đụng
