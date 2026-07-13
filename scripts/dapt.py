# -*- coding: utf-8 -*-
"""
DAPT trên 100 file test (+ text lâm sàng VN nếu có) -> models/dapt.

  python scripts/dapt.py --model xlm-roberta-large --epochs 5 --out models/dapt

Sau đó: python scripts/train_ner.py --model models/dapt ...  (NER khởi từ encoder đã DAPT)
"""
import os
import sys
import glob
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io.loader import load_dataset             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN = os.path.join(ROOT, "data", "test", "input")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-large")
    ap.add_argument("--input_dir", default=DEFAULT_IN)
    ap.add_argument("--extra_glob", default="",
                    help="thêm text lâm sàng VN (vd 'data/vn_clinical/*.txt')")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "dapt"))
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--block_size", type=int, default=512)
    args = ap.parse_args()

    texts = [d.raw for d in load_dataset(args.input_dir)]
    if args.extra_glob:
        for p in glob.glob(args.extra_glob):
            with open(p, encoding="utf-8", errors="replace") as f:
                texts.append(f.read())
    print(f"DAPT trên {len(texts)} văn bản miền thật")

    from src.ner.dapt import dapt
    dapt(args.model, texts, args.out, epochs=args.epochs,
         batch_size=args.batch_size, block_size=args.block_size)


if __name__ == "__main__":
    main()
