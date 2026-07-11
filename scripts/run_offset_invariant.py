# -*- coding: utf-8 -*-
"""
Chạy kiểm định P1 trên 100 file THẬT:
  1. BẤT BIẾN CharView 1:1 (len(norm)==len(raw)) — phải 100/100.
  2. Section phủ liên tục toàn tài liệu + phân bố section (mục tiêu ~79% bullet gán được).
  3. Probe find_all round-trip: mọi span trả về thoả raw[s:e]==match.raw (grounding).

Chạy: python scripts/run_offset_invariant.py [--input_dir DIR]
"""
import os
import sys
import argparse
import collections

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io.loader import load_dataset          # noqa: E402
from src.io.offsets import CharView, is_grounded  # noqa: E402
from src.segment.sections import segment, OTHER   # noqa: E402

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "test", "input")

PROBES = ["táo bón", "khó thở", "đánh trống ngực", "metoprolol", "aspirin",
          "kali", "creatinine", "tăng huyết áp", "sốt", "buồn nôn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    docs = load_dataset(args.input_dir)
    print(f"Nạp {len(docs)} tài liệu từ {args.input_dir}\n")

    # 1) CharView 1:1
    cv_ok = 0
    for d in docs:
        v = CharView(d.raw)
        if len(v.norm) == len(d.raw):
            cv_ok += 1
        else:
            print(f"  [FAIL 1:1] {d.doc_id}: {len(v.norm)} != {len(d.raw)}")
    print(f"[1] CharView 1:1        : {cv_ok}/{len(docs)}")

    # 2) Section: phủ liên tục + phân bố bullet
    cover_ok = 0
    sec_hist = collections.Counter()
    bullets_total = bullets_assigned = 0
    for d in docs:
        seg = segment(d.raw, d.doc_id)
        ok = seg.spans[0].char_start == 0 and seg.spans[-1].char_end == len(d.raw)
        ok = ok and all(a.char_end == b.char_start for a, b in zip(seg.spans, seg.spans[1:]))
        cover_ok += int(ok)
        # phân bố bullet
        off = 0
        for line in d.raw.splitlines(keepends=True):
            s = line.strip()
            if s.startswith("-") or s.startswith("*"):
                bullets_total += 1
                sec = seg.section_at(off)
                sec_hist[sec] += 1
                if sec != OTHER:
                    bullets_assigned += 1
            off += len(line)
    print(f"[2] Section phủ liên tục : {cover_ok}/{len(docs)}")
    pct = 100 * bullets_assigned / max(bullets_total, 1)
    print(f"    Bullet gán được section: {bullets_assigned}/{bullets_total} = {pct:.0f}%")
    print("    Phân bố bullet theo section:")
    for sec, c in sec_hist.most_common():
        print(f"      {sec:16s} {c}")

    # 3) find_all round-trip grounding
    probe_hits, probe_fail = 0, 0
    for d in docs:
        v = CharView(d.raw)
        for needle in PROBES:
            for m in v.find_all(needle):
                probe_hits += 1
                if not (d.raw[m.start:m.end] == m.raw and is_grounded(d.raw, m.raw, (m.start, m.end))):
                    probe_fail += 1
                    print(f"  [FAIL ground] {d.doc_id} needle={needle!r}")
    print(f"\n[3] find_all grounding   : {probe_hits} match, {probe_fail} lỗi grounding")

    ok = (cv_ok == len(docs)) and (cover_ok == len(docs)) and (probe_fail == 0)
    print("\n" + ("[OK] TẤT CẢ BẤT BIẾN P1 GIỮ VỮNG" if ok else "[FAIL] CÓ LỖI — xem trên"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
