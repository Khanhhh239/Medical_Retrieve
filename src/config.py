# -*- coding: utf-8 -*-
"""Nạp ngưỡng tune-được từ configs/thresholds.yaml (1 chỗ duy nhất)."""
from __future__ import annotations

import os
import yaml

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "configs", "thresholds.yaml")

try:
    with open(_PATH, "r", encoding="utf-8") as f:
        _CFG = yaml.safe_load(f) or {}
except FileNotFoundError:
    _CFG = {}


def get(section: str, key: str, default):
    """Lấy ngưỡng; nếu thiếu file/khoá -> dùng default (giữ hành vi cũ)."""
    return (_CFG.get(section) or {}).get(key, default)
