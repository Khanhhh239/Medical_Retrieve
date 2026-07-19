# -*- coding: utf-8 -*-
"""
v3 · SWEEP ngưỡng abstention — 1 lần chạy model, xuất NHIỀU zip (tiết kiệm quota).

  python scripts/sweep_prob.py --model_dir models/ner --assert_model models/assert \
      --thresholds 0,0.6,0.75,0.85

Chạy NER 1 lần/ file (lấy span + confidence), rồi VỚI MỖI ngưỡng: lọc span conf<τ ->
assertion -> linking -> output_ner_p{τ}.zip. Nộp từng zip để tìm τ điểm cao nhất.
Đòn: over-prediction bị phạt 3 tầng -> cắt span thừa (conf thấp) nâng cả 3 điểm.
"""
import os
import sys
import argparse
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io.loader import load_dataset              # noqa: E402
from src.assemble.writer import save_json           # noqa: E402
from src.link.pipeline import link_concepts         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.join(ROOT, "models", "ner"))
    ap.add_argument("--assert_model", default="")
    ap.add_argument("--input_dir", default=os.path.join(ROOT, "data", "test", "input"))
    ap.add_argument("--thresholds", default="0,0.6,0.75,0.85")
    args = ap.parse_args()

    from src.ner.predict import NERPredictor
    predictor = NERPredictor(args.model_dir)
    am = None
    if args.assert_model:
        from src.assert_.model import AssertionModel
        am = AssertionModel(args.assert_model)
    else:
        from src.extract.enrich import add_assertions

    docs = load_dataset(args.input_dir)
    # CHẠY MODEL 1 LẦN / file -> (Concept, conf)
    raw = {d.doc_id: predictor.predict_with_conf(d.raw) for d in docs}
    thr = [float(x) for x in args.thresholds.split(",") if x != ""]
    id2raw = {d.doc_id: d.raw for d in docs}

    for t in thr:
        out_dir = os.path.join(ROOT, f"output_ner_p{int(round(t*100)):02d}")
        os.makedirs(out_dir, exist_ok=True)
        n_c = 0
        for d in docs:
            concepts = [c for c, cf in raw[d.doc_id] if cf >= t]     # lọc theo confidence
            concepts = am.annotate(d.raw, concepts) if am else add_assertions(d.raw, concepts)
            concepts = link_concepts(concepts)
            recs = save_json(concepts, d.raw, os.path.join(out_dir, f"{d.doc_id}.json"))
            n_c += len(recs)
        zpath = os.path.join(ROOT, f"output_ner_p{int(round(t*100)):02d}.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for d in docs:
                z.write(os.path.join(out_dir, f"{d.doc_id}.json"), arcname=f"output/{d.doc_id}.json")
        print(f"[sweep] τ={t}: {n_c} concept ({round(n_c/len(docs),1)}/file) -> {zpath}")


if __name__ == "__main__":
    main()
