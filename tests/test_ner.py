# -*- coding: utf-8 -*-
"""Test NER (P4). bio_to_spans thuần logic; predict cần model (skip nếu chưa train)."""
import os
import importlib.util

import pytest

from src.io.offsets import is_grounded
from src.ner.labels import bio_to_spans, LABEL2ID

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "ner_smoke")
_HAS_TF = importlib.util.find_spec("transformers") is not None
_HAS_MODEL = os.path.isdir(MODEL_DIR)


def test_bio_to_spans_merges_bi():
    labels = [LABEL2ID["O"], LABEL2ID["B-THUỐC"], LABEL2ID["I-THUỐC"], LABEL2ID["O"]]
    offsets = [(0, 0), (0, 7), (8, 10), (0, 0)]      # token0 đặc biệt
    assert bio_to_spans(labels, offsets) == [(0, 10, "THUỐC")]


def test_bio_to_spans_separate_entities():
    labels = [LABEL2ID["B-TRIỆU_CHỨNG"], LABEL2ID["B-THUỐC"]]
    offsets = [(0, 3), (4, 11)]
    assert bio_to_spans(labels, offsets) == [(0, 3, "TRIỆU_CHỨNG"), (4, 11, "THUỐC")]


@pytest.mark.skipif(not (_HAS_TF and _HAS_MODEL),
                    reason="Chưa có transformers hoặc models/ner_smoke (chạy train_ner.py)")
def test_ner_predictions_are_grounded():
    from src.ner.predict import NERPredictor
    p = NERPredictor(MODEL_DIR)
    text = ("2. Bệnh sử hiện tại\nCác triệu chứng hiện tại\n- sốt\n- khó thở\n"
            "3. Đánh giá tại bệnh viện\nKết quả xét nghiệm:\n- kali 6.3\n")
    for c in p.predict(text):
        assert is_grounded(text, c.text, c.position), (c.text, c.position)
