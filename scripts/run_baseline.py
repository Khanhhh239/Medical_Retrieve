# -*- coding: utf-8 -*-
"""
Chạy baseline structural (P2) trên 100 file -> output/*.json + diagnostics.
Chạy: python scripts/run_baseline.py [--input_dir DIR] [--out_dir DIR]

LƯU Ý TRUNG THỰC: không có gold cho 100 file test nên KHÔNG in "điểm". Chỉ in
thống kê cấu trúc + kiểm tra tính hợp lệ. Điểm thật xem test micro-eval trên ví
dụ BTC (tests/test_baseline.py) hoặc khi có gold.
"""
import os
import sys
import json
import argparse
import collections

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io.loader import load_dataset                    # noqa: E402
from src.io.offsets import is_grounded                    # noqa: E402
from src.extract.baseline import extract                  # noqa: E402
from src.assemble.writer import to_records, save_json     # noqa: E402
from src.metric.scorer import VALID_TYPES, CANDIDATE_TYPES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN = os.path.join(ROOT, "data", "test", "input")
DEFAULT_OUT = os.path.join(ROOT, "output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=DEFAULT_IN)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--no-link", action="store_true", help="bỏ bước link candidates")
    ap.add_argument("--zip", action="store_true", help="đóng output.zip nộp thi")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    link_fn = None
    if not args.no_link:
        from src.link.pipeline import link_concepts
        link_fn = link_concepts

    docs = load_dataset(args.input_dir)
    by_type = collections.Counter()
    assert_counts = collections.Counter()
    n_concepts = n_with_assert = n_with_cand = n_linkable = 0
    n_ungrounded = n_bad_schema = 0
    per_file = []

    for d in docs:
        concepts = extract(d)
        if link_fn:
            concepts = link_fn(concepts)
        recs = save_json(concepts, d.raw, os.path.join(args.out_dir, f"{d.doc_id}.json"))
        per_file.append(len(recs))
        for r in recs:
            n_concepts += 1
            by_type[r["type"]] += 1
            if r["assertions"]:
                n_with_assert += 1
                for a in r["assertions"]:
                    assert_counts[a] += 1
            # kiểm tra hợp lệ
            if not is_grounded(d.raw, r["text"], tuple(r["position"])):
                n_ungrounded += 1
            if r["type"] not in VALID_TYPES:
                n_bad_schema += 1
            if ("candidates" in r) != (r["type"] in CANDIDATE_TYPES):
                n_bad_schema += 1
            if r["type"] in CANDIDATE_TYPES:
                n_linkable += 1
                if r.get("candidates"):
                    n_with_cand += 1

    print(f"Đã xử lý {len(docs)} file -> {args.out_dir}")
    print(f"Tổng concept: {n_concepts} | TB {n_concepts/len(docs):.1f}/file "
          f"| min {min(per_file)} max {max(per_file)}")
    print("\nPhân bố type:")
    for t in VALID_TYPES:
        print(f"  {t:20s} {by_type.get(t,0)}")
    print(f"\nConcept có assertion: {n_with_assert} "
          f"({100*n_with_assert/max(n_concepts,1):.0f}%)")
    for a, c in assert_counts.most_common():
        print(f"  {a:14s} {c}")

    link_state = "TẮT (--no-link)" if link_fn is None else "BẬT (seed KB)"
    pct_cand = 100 * n_with_cand / max(n_linkable, 1)
    print(f"\nLinking: {link_state}")
    print(f"  CHẨN_ĐOÁN/THUỐC có candidate: {n_with_cand}/{n_linkable} = {pct_cand:.0f}% "
          f"(phần còn lại ngoài seed KB — cắm RxNorm/ICD đầy đủ để tăng)")

    print("\n--- Kiểm tra hợp lệ ---")
    print(f"  Ungrounded (raw[s:e]!=text): {n_ungrounded}  (kỳ vọng 0)")
    print(f"  Sai schema (type/candidates): {n_bad_schema}  (kỳ vọng 0)")

    if args.zip:
        import zipfile
        zpath = os.path.join(os.path.dirname(args.out_dir), "output.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for d in docs:
                z.write(os.path.join(args.out_dir, f"{d.doc_id}.json"),
                        arcname=f"output/{d.doc_id}.json")
        print(f"\nĐã đóng gói: {zpath}")

    ok = (n_ungrounded == 0 and n_bad_schema == 0)
    print("\n" + ("[OK] Output hợp lệ, grounded 100%." if ok else "[FAIL] Có lỗi hợp lệ."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
