# -*- coding: utf-8 -*-
"""
Inference THỐNG NHẤT -> output/*.json (+ output.zip). (lời giải chính để nộp)

  # nhánh rule (chạy ngay, không cần train):
  python scripts/predict.py --pipeline baseline --zip
  # nhánh model (cần train NER trước):
  python scripts/predict.py --pipeline ner --model_dir models/ner --zip
"""
import os
import sys
import argparse
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io.loader import load_dataset               # noqa: E402
from src.io.offsets import is_grounded               # noqa: E402
from src.assemble.writer import save_json            # noqa: E402
from src.link.pipeline import link_concepts, get_linkers   # noqa: E402
from datagen.denoise import clean_concepts           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN = os.path.join(ROOT, "data", "test", "input")
DEFAULT_OUT = os.path.join(ROOT, "output")


def build_pipeline(name, model_dir, no_link, clean=False, min_prob=0.0, assert_model=""):
    if name == "baseline":
        from src.extract.baseline import extract
        def run(doc):
            return extract(doc)
    elif name == "ner":
        from src.ner.predict import NERPredictor
        predictor = NERPredictor(model_dir)
        if assert_model:                                    # v3: assertion HỌC (thay ConText)
            from src.assert_.model import AssertionModel
            am = AssertionModel(assert_model)
            def run(doc):
                spans = predictor.predict(doc.raw, min_prob=min_prob)
                return am.annotate(doc.raw, spans)
        else:                                               # ConText rule (bản 34.60)
            from src.extract.enrich import add_assertions
            def run(doc):
                spans = predictor.predict(doc.raw, min_prob=min_prob)
                return add_assertions(doc.raw, spans)
    else:
        raise ValueError(name)

    linkers = get_linkers() if clean else None

    def full(doc):
        concepts = run(doc)
        if clean:                                    # v2: khử nhiễu (bỏ rác, cắt ngoặc, KB-trim)
            concepts = clean_concepts(doc.raw, concepts, linkers=linkers, kb_trim=True)
        if not no_link:
            concepts = link_concepts(concepts)
        return concepts
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="baseline", choices=["baseline", "ner"])
    ap.add_argument("--model_dir", default=os.path.join(ROOT, "models", "ner"))
    ap.add_argument("--input_dir", default=DEFAULT_IN)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--no-link", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="BẬT khử nhiễu v2 (mặc định TẮT — đo trên leaderboard là xấu hơn)")
    ap.add_argument("--min_prob", type=float, default=0.0,
                    help="Ngưỡng tin cậy: bỏ span NER conf < ngưỡng (cắt span thừa). 0 = như 34.60. Thử 0.6/0.75/0.85.")
    ap.add_argument("--assert_model", default="",
                    help="v3: thư mục assertion head (Khối D). Rỗng = ConText rule (34.60).")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    run = build_pipeline(args.pipeline, args.model_dir, args.no_link,
                         clean=args.clean, min_prob=args.min_prob,
                         assert_model=args.assert_model)
    docs = load_dataset(args.input_dir)
    n_c = n_ung = 0
    for d in docs:
        concepts = run(d)
        recs = save_json(concepts, d.raw, os.path.join(args.out_dir, f"{d.doc_id}.json"))
        n_c += len(recs)
        n_ung += sum(0 if is_grounded(d.raw, r["text"], tuple(r["position"])) else 1
                     for r in recs)

    print(f"[{args.pipeline}] {len(docs)} file, {n_c} concept, ungrounded={n_ung}")
    if args.zip:
        base = os.path.basename(args.out_dir.rstrip("/\\")) or "output"
        zpath = os.path.join(ROOT, f"{base}.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for d in docs:                   # cấu trúc bên trong LUÔN là output/<id>.json (BTC yêu cầu)
                z.write(os.path.join(args.out_dir, f"{d.doc_id}.json"),
                        arcname=f"output/{d.doc_id}.json")
        print(f"-> {zpath}")
    sys.exit(0 if n_ung == 0 else 1)


if __name__ == "__main__":
    main()
