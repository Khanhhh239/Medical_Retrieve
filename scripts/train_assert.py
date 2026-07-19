# -*- coding: utf-8 -*-
"""
v3 · Train ASSERTION head (Khối D) từ silver. CHẠY TRÊN KAGGLE (GPU).

  python scripts/train_assert.py --model demdecuong/vihealthbert-base-syllable \
      --train data/silver.jsonl,data/synthetic/llm.jsonl --out models/assert

Nhãn assertion lấy từ silver (LLM giỏi ngữ nghĩa phủ định/tiền sử). Encoder mặc định
ViHealthBERT-syllable (y tế VN); đổi --model xlm-roberta-base nếu tích hợp lỗi.
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
    ap.add_argument("--model", default="demdecuong/vihealthbert-base-syllable")
    ap.add_argument("--train", default="")     # jsonl silver, phẩy ngăn nhiều nguồn
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "assert"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--optim", default="adamw_torch")
    args = ap.parse_args()

    from src.io.jsonl import load_labeled
    from datagen.assert_data import build_examples, ASSERT_LABELS
    from src.assert_.model import train

    docs = []
    for p in [x for x in args.train.split(",") if x]:
        if os.path.exists(p):
            docs.extend(load_labeled(p))
    ex = build_examples(docs)
    # thống kê nhãn (sanity)
    import collections
    cnt = collections.Counter()
    for _t, y in ex:
        for k, v in enumerate(y):
            if v:
                cnt[ASSERT_LABELS[k]] += 1
    print(f"[assert] {len(ex)} ví dụ | nhãn dương: {dict(cnt)}")
    if not ex:
        print("!! không có ví dụ assertion (thiếu silver?)"); return
    train(args.model, ex, args.out, epochs=args.epochs, batch_size=args.batch_size,
          max_length=args.max_length, optim=args.optim)


if __name__ == "__main__":
    main()
