# -*- coding: utf-8 -*-
"""
Đo 1 pipeline trên tập có nhãn (gold JSONL) bằng metric BTC. (P6)

Chạy: python scripts/evaluate.py --gold data/synthetic/dev.jsonl --pipeline baseline

LƯU Ý TRUNG THỰC: điểm trên dev SYNTHETIC ≠ điểm test thật (phân phối khác). Đây là
tín hiệu dev hợp lệ để so sánh các phiên bản pipeline, KHÔNG phải điểm cuộc thi.
"""
import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io.jsonl import load_labeled                 # noqa: E402
from src.io.loader import Document                    # noqa: E402
from src.metric.scorer import score_dataset, ScorerConfig  # noqa: E402


def _pipeline(name):
    if name == "baseline":
        from src.extract.baseline import extract
        return lambda text, doc_id: extract(Document(doc_id, text, ""))
    if name == "baseline+link":
        from src.extract.baseline import extract
        from src.link.pipeline import link_concepts
        return lambda text, doc_id: link_concepts(extract(Document(doc_id, text, "")))
    if name.startswith("ner:"):        # "ner:models/ner" -> nhánh model đầy đủ
        model_dir = name.split(":", 1)[1]
        from src.ner.predict import NERPredictor
        from src.extract.enrich import add_assertions
        from src.link.pipeline import link_concepts
        predictor = NERPredictor(model_dir)
        return lambda text, doc_id: link_concepts(add_assertions(text, predictor.predict(text)))
    raise ValueError(f"pipeline không hợp lệ: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pipeline", default="baseline")
    ap.add_argument("--wer_mode", default="aligned", choices=["aligned", "concat"])
    args = ap.parse_args()

    gold = load_labeled(args.gold)
    run = _pipeline(args.pipeline)
    samples = [(run(text, doc_id), concepts) for doc_id, text, concepts in gold]

    cfg = ScorerConfig(wer_mode=args.wer_mode)
    ds = score_dataset(samples, cfg)

    print(f"Pipeline: {args.pipeline} | gold: {args.gold} ({ds.n} mẫu) | WER={args.wer_mode}")
    print("-" * 52)
    print(f"  text_score       (0.30): {ds.text_score:.4f}")
    print(f"  assertions_score (0.30): {ds.assertions_score:.4f}")
    print(f"  candidates_score (0.40): {ds.candidates_score:.4f}")
    print(f"  FINAL                  : {ds.final:.4f}")
    print("-" * 52)
    print("  (dev SYNTHETIC — không phải điểm test thật)")


if __name__ == "__main__":
    main()
