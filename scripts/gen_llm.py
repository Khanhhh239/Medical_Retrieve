# -*- coding: utf-8 -*-
"""
Sinh data train bằng LLM ≤9B (Qwen2.5-7B) làm ANNOTATOR. CHẠY TRÊN KAGGLE.

  python scripts/gen_llm.py --n 6000 --backend vllm --out data/synthetic/llm.jsonl

vLLM (mặc định) nhanh 10-20x cho bulk-gen; HF là dự phòng nếu vLLM lỗi version.
Mỗi output -> parse_marked -> grounded -> lọc chất lượng -> JSONL.
Research: encoder fine-tuned > LLM few-shot ở NER span -> LLM chỉ SINH DATA.
"""
import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _prompts(tok, n):
    from datagen.llm_gen import build_prompt, SPECIALTIES
    return [tok.apply_chat_template(build_prompt(SPECIALTIES[i % len(SPECIALTIES)]),
                                    tokenize=False, add_generation_prompt=True)
            for i in range(n)]


def gen_vllm(model, n, max_new):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, max_model_len=2048, gpu_memory_utilization=0.90, dtype="half")
    tok = llm.get_tokenizer()
    outs = llm.generate(_prompts(tok, n),
                        SamplingParams(temperature=0.9, top_p=0.95, max_tokens=max_new))
    return [o.outputs[0].text for o in outs]


def gen_hf(model, n, max_new, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    m = AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb, device_map="auto").eval()
    prompts, texts = _prompts(tok, n), []
    for i in range(0, n, batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(m.device)
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=max_new, do_sample=True,
                             temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
        for j in range(out.shape[0]):
            texts.append(tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        print(f"  {i + batch_size}/{n}", flush=True)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--hf_model", default="Qwen/Qwen2.5-7B-Instruct")  # cho backend hf (bnb 4-bit)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "synthetic", "llm.jsonl"))
    ap.add_argument("--max_new_tokens", type=int, default=900)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--min_concepts", type=int, default=4)
    args = ap.parse_args()

    if args.backend == "vllm":
        raw = gen_vllm(args.model, args.n, args.max_new_tokens)
    else:
        raw = gen_hf(args.hf_model, args.n, args.max_new_tokens, args.batch_size)

    from datagen.llm_gen import parse_marked
    from src.io.jsonl import save_labeled
    from src.io.offsets import is_grounded
    items, kept = [], 0
    for i, gen in enumerate(raw):
        clean, cs = parse_marked(gen)
        cs = [c for c in cs if is_grounded(clean, c.text, c.position)]
        if len(cs) >= args.min_concepts:
            items.append((f"llm_{i}", clean, cs)); kept += 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_labeled(args.out, items)
    print(f"[LLM-gen/{args.backend}] {kept}/{args.n} note hợp lệ (grounded) -> {args.out}")


if __name__ == "__main__":
    main()
