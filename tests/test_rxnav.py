# -*- coding: utf-8 -*-
"""Test parser RxNav (không cần mạng). Live-call để skip nếu offline."""
import pytest

from src.link.rxnav import parse_exact, parse_approx


def test_parse_exact():
    data = {"idGroup": {"name": "aspirin", "rxnormId": ["1191"]}}
    assert parse_exact(data) == "1191"
    assert parse_exact({"idGroup": {}}) is None
    assert parse_exact({}) is None


def test_parse_approx():
    data = {"approximateGroup": {"candidate": [
        {"rxcui": "315431", "score": "12.0", "name": "aspirin 81 MG"}]}}
    assert parse_approx(data) == "315431"
    assert parse_approx({"approximateGroup": {}}) is None
    assert parse_approx({}) is None


@pytest.mark.skipif(True, reason="live call — bật tay khi có mạng")
def test_live_rxcui():
    from src.link.rxnav import rxcui_for
    assert rxcui_for("aspirin") == "1191"
