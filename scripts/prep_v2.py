# -*- coding: utf-8 -*-
"""
v2 — Chuẩn bị data train theo triết lý KB-làm-xương-sống + khử nhiễu.

  python scripts/prep_v2.py --silver data/silver.jsonl --llm data/synthetic/llm.jsonl \
      --input_dir data/test/input --out_dir data/v2

Sinh:
  - distant.jsonl : nhãn KB (thuốc/bệnh biên sạch + lexicon triệu chứng/XN) trên file THẬT.
  - silver_clean.jsonl / llm_clean.jsonl : silver & synth ĐÃ KHỬ NHIỄU (bỏ rác, cắt ngoặc,
    KB-trim về sub-span link được).
Stage C train trên gộp [distant, silver_clean, llm_clean, template].
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
    ap.add_argument("--silver", default="")
    ap.add_argument("--llm", default="")
    ap.add_argument("--input_dir", default=os.path.join(ROOT, "data", "test", "input"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data", "v2"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from src.link.kb import load_rxnorm, load_icd10
    from src.link.linker import Linker
    from src.io.loader import load_dataset
    from src.io.jsonl import load_labeled, save_labeled
    from datagen.kb_distant import build_term_index, distant_label
    from datagen.denoise import clean_concepts
    from datagen.lexicon import SYMPTOMS, LABS

    rx, icd = Linker(load_rxnorm()), Linker(load_icd10())
    linkers = (rx, icd)

    # 1) DISTANT: KB làm nhãn biên sạch trên 100 file thật
    idx = build_term_index(rx, icd, symptoms=SYMPTOMS, labs=[l[0] for l in LABS])
    dist, nd = [], 0
    for d in load_dataset(args.input_dir):
        cs = distant_label(d.raw, idx)
        dist.append((d.doc_id, d.raw, cs)); nd += len(cs)
    dpath = os.path.join(args.out_dir, "distant.jsonl")
    save_labeled(dpath, dist)
    print(f"[distant] {len(dist)} file, {nd} nhãn KB (biên sạch) -> {dpath}")

    # 2) KHỬ NHIỄU silver + llm
    def denoise(path, name):
        if not path or not os.path.exists(path):
            print(f"[{name}] bỏ qua (không có file)")
            return
        items = load_labeled(path)
        before = sum(len(c) for _, _, c in items)
        # training: chỉ bỏ rác + cắt ngoặc (biên sạch hơn để HỌC). KHÔNG kb_trim.
        clean = [(did, txt, clean_concepts(txt, cs, kb_trim=False))
                 for did, txt, cs in items]
        after = sum(len(c) for _, _, c in clean)
        out = os.path.join(args.out_dir, name + "_clean.jsonl")
        save_labeled(out, clean)
        print(f"[{name}] {before} -> {after} nhãn (bỏ {before - after} rác) -> {out}")

    denoise(args.silver, "silver")
    denoise(args.llm, "llm")
    print("XONG. Stage C train trên:", args.out_dir, "+ template.")


if __name__ == "__main__":
    main()
