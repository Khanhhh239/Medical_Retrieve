# Medical — Trích xuất & Chuẩn hoá Khái niệm Y khoa (tiếng Việt)

Hệ thống phát hiện khái niệm y tế trong văn bản lâm sàng free-form tiếng Việt, gán
loại (5 type), suy luận assertion (phủ định / người nhà / tiền sử) và ánh xạ chuẩn
**ICD-10** (chẩn đoán) / **RxNorm** (thuốc).

Kiến trúc & lý luận đầy đủ: xem [`../medical.md`](../medical.md).
Trạng thái: **P0→P6 hoàn tất, 112 test xanh, chạy end-to-end trên CUDA.**

---

## 0. Có gì trong này

| Phase | Nội dung | Module | Test |
|---|---|---|---|
| P0 | Metric BTC (oracle) | `src/metric/scorer.py` | `test_metric` |
| P1 | Data layer: offset 1:1, section, parser số | `src/io`, `src/segment`, `src/parse` | `test_offsets/segment/parse` |
| P2 | Baseline rule/section (training-free) | `src/extract/baseline.py` | `test_baseline` |
| P3 | Sinh data synthetic có nhãn | `datagen/` | `test_synth` |
| P4 | NER train + predict (CUDA) | `src/ner/` | `test_ner` |
| P5 | Linking ICD-10 / RxNorm | `src/link/` | `test_link` |
| P6 | Assertion ConText + writer grounding | `src/assert_`, `src/assemble` | `test_context/writer` |

Hai nhánh inference: **baseline** (rule, chạy ngay) và **ner** (model, cần train).

---

## 1. Cài đặt môi trường (conda + CUDA)

GPU dùng khi train: **NVIDIA RTX 4050 Laptop 6GB**, CUDA 12.4.

```bash
conda create -n medical python=3.11 -y
conda activate medical

# Data layer + metric + baseline + linking (CPU):
pip install -r requirements.txt

# Train NER (GPU): torch CUDA + transformers
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate seqeval
```

Kiểm tra:
```bash
python scripts/check_env.py      # in Python/deps/CUDA/GPU
```

---

## 2. Dữ liệu

```
data/
├── test/input/1.txt … 100.txt   # tập test BTC (giải nén input.zip vào đây)
├── kb/rxnorm_seed.csv           # KB seed (thay bằng RxNorm RRF đầy đủ — §6)
├── kb/icd10_seed.csv            # KB seed (thay bằng ICD-10 VN đầy đủ — §6)
└── synthetic/                   # data train tự sinh (tạo ở §5)
```

Giải nén tập test (nếu chưa có):
```bash
# Windows PowerShell:
Expand-Archive input.zip -DestinationPath data\test -Force
```

---

## 3. Chạy test (LÀM TRƯỚC — xanh mới tin kết quả)

```bash
pytest -q                                # 112 test: toán + data + baseline + link + ner
python scripts/run_offset_invariant.py   # bất biến offset + section trên 100 file thật
```
Kỳ vọng: `112 passed`; CharView 1:1 = 100/100; find_all grounding 0 lỗi.

---

## 4. CHẠY inference → nộp (output.zip)

### 4a. Nhánh baseline (rule — chạy ngay, KHÔNG cần train)
```bash
python scripts/predict.py --pipeline baseline --zip
```
Sinh `output/1.json … 100.json` + `output.zip` (đúng cấu trúc BTC:
`output/<id>.json`). Mọi concept thoả `raw[start:end]==text` (grounding),
candidates điền từ KB.

### 4b. Nhánh NER (model — cần train ở §5 trước)
```bash
python scripts/predict.py --pipeline ner --model_dir models/ner --zip
```

Cờ: `--no-link` (bỏ linking), `--input_dir`, `--out_dir`.

---

## 5. TRAIN mô hình NER

Đề yêu cầu dùng giải pháp ngoài để **sinh data huấn luyện** rồi train model
self-host ≤9B. Quy trình:

### 5a. Sinh corpus synthetic có nhãn
```bash
python scripts/gen_synth.py --n_train 2000 --n_dev 200
# -> data/synthetic/train.jsonl, dev.jsonl
```
Bộ sinh mô hình hoá CẤU TRÚC data thật (17 header, bullet, 5 type, assertion, nhiễu),
offset chính xác by-construction. Mở rộng lexicon/độ đa dạng trong `datagen/lexicon.py`.

### 5b. Train (đã tối ưu cho card 6GB — RTX 4050)
```bash
# THẬT — XLM-R (đa ngôn ngữ, đã kiểm chứng train + lưu OK trên 6GB):
python scripts/train_ner.py --model xlm-roberta-base --epochs 3 --out models/ner
```
Mặc định đã VỪA card 6GB: `batch_size=4`, `grad_accum=4` (batch hiệu dụng 16),
`max_length=256`, **optimizer Adafactor** (ít VRAM hơn AdamW ~10×), gradient
checkpointing, và **chỉ lưu model cuối** (không lưu checkpoint giữa chừng — tiết kiệm disk).

Lưu ý VRAM/disk (đã gặp thật, đã xử lý):
- Dùng **AdamW + batch 8 sẽ OOM** trên 6GB → giữ Adafactor (mặc định). Nếu vẫn thiếu:
  `--batch_size 2 --grad_accum 8`.
- Lưu model ~1.1GB cần **≥2GB trống**. Hết disk → dọn `pip cache purge`, xoá model cũ,
  hoặc bớt `~/.cache/huggingface`.
- **ViHealthBERT** (y tế VN, tốt hơn về lý thuyết): dựa trên PhoBERT, **có thể thiếu
  tokenizer fast** → code lấy-offset sẽ vỡ. Chưa kiểm chứng; cần sửa tokenizer trước khi dùng.

### 5c. Đo trên dev synthetic
```bash
python scripts/evaluate.py --gold data/synthetic/dev.jsonl --pipeline baseline
python scripts/evaluate.py --gold data/synthetic/dev.jsonl --pipeline baseline+link
python scripts/evaluate.py --gold data/synthetic/dev.jsonl --pipeline "ner:models/ner"
```
In text/assertions/candidates/FINAL theo đúng metric BTC (`0.3/0.3/0.4`).

---

## 6. Cắm KB đầy đủ (tăng candidates — 0.4 điểm)

Seed KB nhỏ → coverage thấp trên data thật (đo được ~15%). Thay bằng KB đầy đủ,
engine tự dùng, không đổi code:

- **RxNorm**: đặt `RXNCONSO.RRF` vào `data/kb/` → `src/link/kb.py` tự nạp
  (lọc `LAT=ENG`, `TTY∈{IN,PIN,BN,SCD,SBD,SCDC}`).
- **ICD-10**: đặt `data/kb/icd10.csv` (cột `code,term`, nên có term tiếng Việt).

*(Nguồn RxNorm/UMLS cần license PhysioNet/NLM — chú ý điều khoản khi nộp data cho
top-15. ICD-10 VN: dùng bản Bộ Y tế được phép phân phối.)*

Nâng cấp linker (tuỳ chọn): bật rerank cross-lingual bằng SapBERT-XLMR cho chẩn đoán
tiếng Việt (điểm mơ hồ) — khung ở `src/link/linker.py`.

---

## 7. Cấu trúc thư mục

```
Medical/
├── medical.md ............... (thư mục cha) kiến trúc + kế hoạch + câu hỏi BTC
├── requirements.txt
├── configs/
│   ├── sections.yaml ........ 17 header → 8+ section canonical (CONFIG)
│   └── assertion_cues.yaml .. cue ConText (có dấu + \b)
├── data/{test,kb,synthetic}/
├── src/
│   ├── metric/scorer.py ..... P0 metric (oracle)
│   ├── io/ .................. loader, offsets 1:1, jsonl
│   ├── segment/sections.py .. section segmenter fuzzy
│   ├── parse/numbers.py ..... parser số (phẩy thập phân vs nghìn)
│   ├── extract/ ............. baseline.py, enrich.py
│   ├── assert_/context.py ... ConText assertion
│   ├── link/ ................ kb, linker, pipeline (ICD-10 + RxNorm)
│   ├── ner/ ................. labels(BIO), dataset, train, predict
│   └── assemble/writer.py ... grounding + JSON output
├── datagen/ ................. lexicon, synth (sinh data)
├── scripts/ ................. check_env, predict, run_baseline, gen_synth,
│                              train_ner, evaluate, run_offset_invariant
└── tests/ .................. 112 test
```

---

## 8. Nguyên tắc (KHÔNG vi phạm)

- **Bất biến §3.1**: mọi concept xuất ra thoả `raw[start:end] == text`. Normalize
  CHỈ để tìm biên, không bao giờ ghi vào output. Writer chặn cứng lần cuối.
- **Không hardcode** tri thức về 100 file test (leakage). Lexicon induce/ingest; KB
  cắm ngoài; ngưỡng calibrate trên dev.
- **Precision-first**: WER phạt over-predict không chặn trên; Jaccard phạt đoán thừa.
  Không chắc thì bỏ; thuốc xuất 1 mã tốt nhất; TRIỆU_CHỨNG/tên/kết quả XN không có
  `candidates`.

---

## 9. TRUNG THỰC về điểm số (đọc kỹ)

- **Chưa có gold cho 100 file test** → KHÔNG có headline score thật. Tự gán rồi tự
  chấm = vô nghĩa (bẫy overfit). Điểm thật cần gold BTC / dev người gán.
- **Điểm trên dev SYNTHETIC là LẠC QUAN**: cấu trúc khớp baseline (text≈1.0) và seed
  KB phủ đúng vocab synthetic (cand≈1.0). Dùng để so sánh phiên bản & train NER,
  KHÔNG suy ra điểm thi.
- **Trên 100 file THẬT**: pipeline chạy, output grounded 100%, nhưng candidate
  coverage ~15% với seed KB (thuốc/bệnh thật ngoài seed) → cần §6.
- **NER hiện là smoke** (1-epoch/300 mẫu, chất lượng thấp). Train thật theo §5b.

Bốn câu hỏi cần BTC làm rõ (đổi cách tối ưu): xem `medical.md` §11 (cách tính WER,
align concept, quy tắc đa-mã ICD, chuẩn hoá text tên XN).

---

## 10. Bảng lệnh nhanh

| Việc | Lệnh |
|---|---|
| Kiểm tra env/CUDA | `python scripts/check_env.py` |
| Chạy test | `pytest -q` |
| Bất biến offset 100 file | `python scripts/run_offset_invariant.py` |
| Inference baseline → zip | `python scripts/predict.py --pipeline baseline --zip` |
| Sinh data train | `python scripts/gen_synth.py` |
| Train NER (thật) | `python scripts/train_ner.py --model demdecuong/vihealthbert-base-syllable --epochs 3 --out models/ner` |
| Đo trên dev | `python scripts/evaluate.py --gold data/synthetic/dev.jsonl --pipeline baseline+link` |
| Inference NER → zip | `python scripts/predict.py --pipeline ner --model_dir models/ner --zip` |
