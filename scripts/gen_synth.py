# -*- coding: utf-8 -*-
"""Sinh corpus synthetic -> data/synthetic/{train,dev}.jsonl. (P3)

Chạy: python scripts/gen_synth.py [--n_train 2000] [--n_dev 200]
"""
import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen.synth import generate_dataset            # noqa: E402
from src.io.jsonl import save_labeled                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "synthetic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_dev", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    train = generate_dataset(args.n_train, seed=42)
    dev = generate_dataset(args.n_dev, seed=1234)      # seed KHÁC train -> không trùng
    save_labeled(os.path.join(OUT, "train.jsonl"), train)
    save_labeled(os.path.join(OUT, "dev.jsonl"), dev)

    n_c = sum(len(c) for _, _, c in train)
    print(f"train: {len(train)} note, {n_c} concept -> {OUT}/train.jsonl")
    print(f"dev  : {len(dev)} note -> {OUT}/dev.jsonl")


if __name__ == "__main__":
    main()
