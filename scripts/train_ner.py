# -*- coding: utf-8 -*-
"""
Train NER (P4). Chạy:
  # SMOKE (chứng minh loop chạy trên CUDA, nhanh):
  python scripts/train_ner.py --limit 300 --epochs 1 --out models/ner_smoke
  # THẬT (đổi model sang ViHealthBERT/XLM-R, full data):
  python scripts/train_ner.py --model demdecuong/vihealthbert-base-syllable --epochs 3 --out models/ner
"""
import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-multilingual-cased")
    ap.add_argument("--train", default=os.path.join(ROOT, "data", "synthetic", "train.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "ner"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=4)      # 6GB-friendly
    ap.add_argument("--grad_accum", type=int, default=4)      # batch hiệu dụng = 4*4=16
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--optim", default="adafactor",
                    help="adafactor (6GB) | adamw_torch (GPU >=16GB như Kaggle)")
    ap.add_argument("--limit", type=int, default=0, help="giới hạn số mẫu (smoke)")
    args = ap.parse_args()

    # nếu --limit: cắt bớt train (smoke) sang file tạm
    train_path = args.train
    if args.limit:
        from src.io.jsonl import load_labeled, save_labeled
        items = load_labeled(args.train)[:args.limit]
        train_path = os.path.join(ROOT, "data", "synthetic", "_train_smoke.jsonl")
        save_labeled(train_path, items)

    from src.ner.train import train
    train(args.model, train_path, args.out, epochs=args.epochs,
          batch_size=args.batch_size, grad_accum=args.grad_accum,
          max_length=args.max_length)


if __name__ == "__main__":
    main()
