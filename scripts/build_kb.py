# -*- coding: utf-8 -*-
"""
XÂY KHO MÃ (offline data prep) — chạy 1 lần, cần internet. KHÔNG phải inference.

  python scripts/build_kb.py --rxnorm --icd

- --rxnorm: quét tên thuốc trong test + synthetic, gọi RxNav API -> data/kb/rxnorm_api.csv
- --icd   : tải ICD-10 (tiếng Anh) công khai -> data/kb/icd10_en.csv (code,term)

Sau đó pipeline inference chỉ ĐỌC các CSV này (không gọi API) — đúng luật thi.
"""
import os
import re
import io
import sys
import csv
import glob
import argparse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io.loader import load_dataset                 # noqa: E402
from src.io.offsets import normalize_str               # noqa: E402
from src.extract.baseline import extract               # noqa: E402
from datagen.lexicon import DRUGS, SIGS, COMMON_DRUGS   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(ROOT, "data", "kb")
TEST = os.path.join(ROOT, "data", "test", "input")
ICD_URL = "https://raw.githubusercontent.com/k4m1113/ICD-10-CSV/master/codes.csv"
# Danh mục ICD-10 tiếng Việt (Bộ Y tế, EN+VN có phân cấp) — phủ mạnh phần bệnh.
ICD_VI_URL = "https://raw.githubusercontent.com/tamton23/primekg-vn-icd10-omop/main/icd10_danh_muc.csv"
_ICD_CODE = re.compile(r"^([A-Z]\d\d(?:\.\d+)?)\*?$")

# token sig cần bỏ khi hỏi API (giữ tên + hàm lượng)
_SIG = re.compile(r"\b(po|iv|im|sc|sl|pr|bid|tid|qid|qd|qhs|qam|qpm|prn|daily|"
                  r"q\d+h|x\d+|hs|ac|pc|oral|suspension|tablet|cap|caps)\b", re.I)


def _clean_query(m: str) -> str:
    """Tên + hàm lượng; nếu ĐƯỜNG UỐNG + hàm lượng mg/g -> ép 'oral tablet' để lấy mã
    SCD (sản phẩm) đúng cấp độ GT (vd amlodipine 10mg -> 308135 khớp ví dụ đề)."""
    low = m.lower()
    base = re.sub(r"[:().]", " ", m)
    base = _SIG.sub(" ", base)
    base = re.sub(r"\s+", " ", base).strip()
    has_oral = bool(re.search(r"\b(po|oral|uống)\b", low))
    has_mg = bool(re.search(r"\d+\s*(mg|mcg|g)\b", low))
    if has_oral and has_mg:
        return base + " oral tablet"
    return base


def collect_drug_mentions() -> set:
    out = set()
    for d in load_dataset(TEST):
        for c in extract(d):
            if c.type == "THUỐC":
                out.add(c.text)
    for name, _ in DRUGS:                    # thêm tên + vài sig để phủ private test
        out.add(name)
        out.add(f"{name} {SIGS[0]}")
    for name in COMMON_DRUGS:                # danh sách thuốc thường gặp -> pre-cache rộng
        out.add(name)
    return out


def build_rxnorm(force=False):
    from src.link.rxnav import rxcui_for
    path = os.path.join(KB, "rxnorm_api.csv")
    done = {}
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                done[row["term"]] = row["code"]
    mentions = collect_drug_mentions()
    print(f"[rxnorm] {len(mentions)} mention thuốc; đã có {len(done)} trong cache")
    n_new = 0
    for m in sorted(mentions):
        term = normalize_str(m)
        if term in done:
            continue
        code = rxcui_for(_clean_query(m))
        if code:
            done[term] = code
            n_new += 1
            print(f"  {m!r:45s} -> {code}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "term"])
        for term, code in sorted(done.items()):
            w.writerow([code, term])
    print(f"[rxnorm] xong: {len(done)} mã -> {path} (+{n_new} mới)")


def build_rxnorm_full(ttys=("IN", "PIN", "BN", "SCD", "SBD", "SCDC")):
    """PHỦ MẠNH thuốc: getAllConcepts toàn bộ RxNorm prescribable (KHÔNG cần UMLS license)."""
    from src.link.rxnav import get_all_concepts
    path = os.path.join(KB, "rxnorm_full.csv")
    seen = {}
    for tty in ttys:
        try:
            concs = get_all_concepts([tty])
            for rxcui, name, _t in concs:
                seen[(rxcui, name)] = 1
            print(f"  {tty}: {len(concs)} concept")
        except Exception as e:
            print(f"  {tty} LỖI: {type(e).__name__} {str(e)[:60]}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "term"])
        for (code, term) in sorted(seen):
            w.writerow([code, term])
    print(f"[rxnorm_full] {len(seen)} (mã,tên) -> {path}")


def build_icd_vi():
    """Tải danh mục ICD-10 VN (Bộ Y tế) -> icd10_vi.csv. Mã ở cột i, tên VN ở i+2 (triple
    code/en/vn theo từng cấp phân loại); lấy mọi cấp có tên tiếng Việt."""
    path = os.path.join(KB, "icd10_vi.csv")
    print(f"[icd_vi] tải {ICD_VI_URL} ...")
    req = urllib.request.Request(ICD_VI_URL, headers={"User-Agent": "kb-builder"})
    raw = urllib.request.urlopen(req, timeout=90).read().decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(raw)))
    pairs = {}
    for r in rows:
        for i, c in enumerate(r):
            m = _ICD_CODE.match(c.strip())
            if not m:
                continue
            code = m.group(1)
            for j in (i + 2, i + 1):                     # tên VN cạnh mã (code/en/vn)
                if j < len(r):
                    term = " ".join(r[j].split()).strip()
                    if (term and re.search(r"[^\x00-\x7f]", term)   # có ký tự phi-ASCII (VN)
                            and len(term) > 3 and not _ICD_CODE.match(term)):
                        pairs[(code, term.lower())] = 1
                        break
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "term"])
        for (c, t) in pairs:
            w.writerow([c, t])
    print(f"[icd_vi] {len(pairs)} (mã,tên VN), {len({c for c,_ in pairs})} mã riêng -> {path}")


def build_icd(force=False):
    path = os.path.join(KB, "icd10_en.csv")
    if os.path.exists(path) and not force:
        print(f"[icd] đã có {path} (dùng --force để tải lại)")
        return
    print(f"[icd] tải {ICD_URL} ...")
    req = urllib.request.Request(ICD_URL, headers={"User-Agent": "kb-builder"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode("utf-8", "replace")
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "term"])
        for row in csv.reader(raw.splitlines()):
            if len(row) < 5:
                continue
            full = row[2].strip()                    # 'A000'
            code = full[:3] + "." + full[3:] if len(full) > 3 else full  # -> 'A00.0'
            term = (row[4] or row[3]).strip()        # long description
            if code and term:
                w.writerow([code, term])
                n += 1
    print(f"[icd] xong: {n} mã -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxnorm", action="store_true")
    ap.add_argument("--rxnorm_full", action="store_true", help="getAllConcepts: phủ TOÀN BỘ RxNorm")
    ap.add_argument("--icd", action="store_true", help="ICD-10 tiếng Anh (cho SapBERT)")
    ap.add_argument("--icd_vi", action="store_true", help="danh mục ICD-10 VN (Bộ Y tế) — phủ bệnh")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (args.rxnorm or args.rxnorm_full or args.icd or args.icd_vi):
        args.rxnorm = args.icd = True
    os.makedirs(KB, exist_ok=True)
    if args.rxnorm_full:
        build_rxnorm_full()
    if args.rxnorm:
        build_rxnorm(args.force)
    if args.icd_vi:
        build_icd_vi()
    if args.icd:
        build_icd(args.force)


if __name__ == "__main__":
    main()
