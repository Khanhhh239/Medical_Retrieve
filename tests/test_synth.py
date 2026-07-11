# -*- coding: utf-8 -*-
"""Test bộ sinh data synthetic (P3): gold phải grounded + tự nhất quán với scorer."""
import random

import pytest

from datagen.synth import generate_note, generate_dataset, GenConfig
from src.io.offsets import is_grounded
from src.metric.scorer import (
    score_dataset, ScorerConfig, ASSERTION_TYPES, CANDIDATE_TYPES, VALID_TYPES,
)


def test_generated_gold_is_grounded():
    rng = random.Random(0)
    for _ in range(50):
        text, concepts = generate_note(rng)
        assert concepts                      # note không rỗng
        for c in concepts:
            assert is_grounded(text, c.text, c.position), (c.text, c.position)


@pytest.mark.parametrize("mode", ["aligned", "concat"])
def test_self_score_is_one(mode):
    data = generate_dataset(30, seed=1)
    samples = [(cs, cs) for _, _, cs in data]
    ds = score_dataset(samples, ScorerConfig(wer_mode=mode))
    assert ds.final == pytest.approx(1.0)


def test_assertions_and_candidates_only_on_eligible_types():
    for _, _, cs in generate_dataset(40, seed=2):
        for c in cs:
            assert c.type in VALID_TYPES
            if c.assertions:
                assert c.type in ASSERTION_TYPES
            if c.candidates:
                assert c.type in CANDIDATE_TYPES


def test_all_three_assertions_appear():
    data = generate_dataset(80, seed=3,
                            cfg=GenConfig(p_negate_symptom=0.5, p_family_section=0.7))
    seen = set()
    for _, _, cs in data:
        for c in cs:
            seen.update(c.assertions)
    assert {"isNegated", "isFamily", "isHistorical"} <= seen
