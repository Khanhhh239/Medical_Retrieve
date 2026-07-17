# -*- coding: utf-8 -*-
"""
Stage B — LLM 32B (Qwen2.5-32B-Instruct-AWQ) làm ANNOTATOR trên 100 file THẬT.

  python scripts/gen_silver.py --model Qwen/Qwen2.5-32B-Instruct-AWQ \
      --n_samples 5 --min_votes 2 --n_synth 3000 \
      --out data/silver.jsonl --synth_out data/synthetic/llm.jsonl

- silver: mỗi file thật -> N mẫu (self-consistency) -> vote -> concepts GROUNDED trong
  raw thật (đánh gục domain gap). Đây là data quan trọng nhất.
- synth (tuỳ chọn, cùng 1 lần load model): sinh note ca hiếm để phủ từ vựng.

CHỈ chạy OFFLINE lúc chuẩn bị data (được dùng model to). Inference KHÔNG dùng cái này.
Recipe 32B-AWQ trên 2xT4: quantization='awq' (KHÔNG marlin - T4 Turing), tensor_parallel=2.
"""
import os
import sys
import argparse
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN = os.path.join(ROOT, "data", "test", "input")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--input_dir", default=DEFAULT_IN)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "silver.jsonl"))
    ap.add_argument("--synth_out", default=os.path.join(ROOT, "data", "synthetic", "llm.jsonl"))
    ap.add_argument("--n_samples", type=int, default=5, help="mẫu/ file (self-consistency)")
    ap.add_argument("--min_votes", type=int, default=2, help="giữ span xuất hiện >= n lần")
    ap.add_argument("--n_synth", type=int, default=3000, help="số note synth (0 = bỏ)")
    ap.add_argument("--min_concepts", type=int, default=4, help="lọc synth ít khái niệm")
    ap.add_argument("--tp", type=int, default=2, help="tensor parallel (2 T4)")
    ap.add_argument("--max_model_len", type=int, default=6144)
    ap.add_argument("--max_new_tokens", type=int, default=3072)
    ap.add_argument("--max_note_chars", type=int, default=5000, help="cắt note quá dài cho vừa ctx")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--gpu_mem", type=float, default=0.92)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from src.io.loader import load_dataset
    from src.io.jsonl import save_labeled
    from datagen.llm_annotate import build_annotate_prompt, vote, chunk_labeled

    docs = load_dataset(args.input_dir)
    print(f"[silver] {len(docs)} file thật; {args.n_samples} mẫu/file")

    # awq (không marlin — T4 Turing sm_75). half = fp16.
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, quantization="awq",
              dtype="half", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, trust_remote_code=True)
    tok = llm.get_tokenizer()

    # ---- gom prompt silver (N mẫu/file) ----
    prompts, owner = [], []
    for di, d in enumerate(docs):
        note = d.raw[:args.max_note_chars]
        if len(d.raw) > args.max_note_chars:
            print(f"  [warn] file {d.doc_id} dài {len(d.raw)} -> cắt {args.max_note_chars}")
        p = tok.apply_chat_template(build_annotate_prompt(note),
                                    tokenize=False, add_generation_prompt=True)
        for _ in range(args.n_samples):
            prompts.append(p); owner.append(di)

    sp = SamplingParams(temperature=args.temperature, top_p=0.9,
                        max_tokens=args.max_new_tokens)
    outs = llm.generate(prompts, sp)

    by_doc = defaultdict(list)
    for oi, o in zip(owner, outs):
        by_doc[oi].append(o.outputs[0].text)

    items, n_concept = [], 0
    for di, d in enumerate(docs):
        cs = vote(d.raw, by_doc[di], min_votes=args.min_votes)
        for k, (sub, subcs) in enumerate(chunk_labeled(d.raw, cs)):   # cắt window ngắn
            items.append((f"{d.doc_id}_w{k}", sub, subcs))
            n_concept += len(subcs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_labeled(args.out, items)
    print(f"[silver] {len(items)} window, {n_concept} concept (grounded) -> {args.out}")

    # ---- synth (tuỳ chọn, cùng model) ----
    if args.n_synth > 0:
        from datagen.llm_gen import build_prompt, parse_marked, SPECIALTIES
        from src.io.offsets import is_grounded
        sprompts = [tok.apply_chat_template(build_prompt(SPECIALTIES[i % len(SPECIALTIES)]),
                                            tokenize=False, add_generation_prompt=True)
                    for i in range(args.n_synth)]
        souts = llm.generate(sprompts, SamplingParams(temperature=0.9, top_p=0.95,
                                                      max_tokens=900))
        sitems, kept = [], 0
        for i, o in enumerate(souts):
            clean, cs = parse_marked(o.outputs[0].text)
            cs = [c for c in cs if is_grounded(clean, c.text, c.position)]
            if len(cs) >= args.min_concepts:
                sitems.append((f"llm_{i}", clean, cs)); kept += 1
        os.makedirs(os.path.dirname(args.synth_out), exist_ok=True)
        save_labeled(args.synth_out, sitems)
        print(f"[synth] {kept}/{args.n_synth} note hợp lệ -> {args.synth_out}")


if __name__ == "__main__":
    main()
