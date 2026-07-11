# -*- coding: utf-8 -*-
"""Kiểm tra môi trường: Python, deps, CUDA. Chạy: python scripts/check_env.py"""
import sys
import platform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    print("=" * 60)
    print("MEDICAL — kiểm tra môi trường")
    print("=" * 60)
    print(f"Python      : {sys.version.split()[0]} ({platform.system()} {platform.release()})")

    for mod in ["numpy", "rapidfuzz", "yaml", "regex", "pytest"]:
        try:
            m = __import__(mod)
            print(f"{mod:12s}: OK {getattr(m, '__version__', '?')}")
        except Exception:
            print(f"{mod:12s}: MISSING")

    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"torch       : {torch.__version__}")
        print(f"CUDA        : {cuda}")
        if cuda:
            p = torch.cuda.get_device_properties(0)
            print(f"GPU         : {p.name} ({round(p.total_memory / 1024**3, 1)} GB)")
        else:
            print("GPU         : (không thấy CUDA — P0/P1 vẫn chạy CPU bình thường)")
    except Exception as e:
        print(f"torch       : MISSING ({e})")
        print("  -> P0 (metric) & P1 (data layer) KHÔNG cần torch; chỉ P4+ cần.")

    print("=" * 60)


if __name__ == "__main__":
    main()
