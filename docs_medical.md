# Medical Concept Extraction & Normalization — Kiến trúc & Kế hoạch thực thi

> **Trạng thái:** bản chốt thiết kế (v1). Mọi con số chất lượng đều là **giả thuyết cho đến khi Phase 2 sinh ra điểm đo được từ chính metric của BTC**. Không tuyên bố SOTA trước khi có số. Tài liệu này là hợp đồng kỹ thuật để triển khai, không phải marketing.

---

## 0. TL;DR

- **Bài toán:** trích xuất 5 loại khái niệm y khoa từ văn bản lâm sàng tiếng Việt free-form, gán assertion ngữ cảnh, và ánh xạ về **ICD-10** (chẩn đoán) / **RxNorm** (thuốc).
- **Hình dạng lời giải (do đề kê đơn):** dùng công cụ mạnh **offline** để *sinh data huấn luyện* → train **một model NER ≤9B** chạy inference **self-host, không API ngoài**. → **Train-based hybrid**, không phải pure training-free.
- **Chia việc theo bản chất tín hiệu:**
  - *Học (train):* NER 5-type span + type (đặc biệt triệu chứng/chẩn đoán tiếng Việt).
  - *Tra cứu (dictionary):* RxNorm, ICD-10.
  - *Luật (rule):* assertion (ConText).
- **Thứ tự thực thi (bắt buộc):** **metric harness → data layer → baseline dictionary/rule → sinh data → train → linking → calibrate**. Metric và data làm trước tiên.

---

## 1. Đặc tả bài toán (locked — nguồn chân lý)

### 1.1 Input
- 100 file `.txt` (`input/1.txt … 100.txt`), UTF-8, LF, NFC. TB ~1.323 ký tự/file, max 4.428. **Toàn corpus < 1 context window.**
- Văn bản free-form: ghi chú bác sĩ, giấy xuất viện, kết quả XN, EHR. Mỗi file có **>1 khái niệm**.

### 1.2 Output — mỗi file `n.txt` → `n.json` là list các concept:

| Trường | Kiểu | Ràng buộc |
|---|---|---|
| `text` | string | **PHẢI** = `raw[start:end]` (bất biến §3.1) |
| `position` | `[start, end]` | offset **ký tự**, 0..n-1, nửa mở `[start,end)` |
| `type` | enum | 1 trong 5 nhãn dưới |
| `assertions` | list | ⊆ `{isNegated, isFamily, isHistorical}`; chỉ cho `CHẨN_ĐOÁN/THUỐC/TRIỆU_CHỨNG`; `[]` nếu không |
| `candidates` | list | chỉ cho `CHẨN_ĐOÁN` (ICD-10) và `THUỐC` (RxNorm); **bỏ hẳn key** với 3 type còn lại |

### 1.3 Năm loại `type`
| Nhãn | Ý nghĩa | candidates | assertions |
|---|---|---|---|
| `TRIỆU_CHỨNG` | triệu chứng | ✗ | ✓ |
| `TÊN_XÉT_NGHIỆM` | tên xét nghiệm | ✗ | ✗ |
| `KẾT_QUẢ_XÉT_NGHIỆM` | giá trị + đơn vị XN | ✗ | ✗ |
| `CHẨN_ĐOÁN` | chẩn đoán bệnh | **ICD-10** | ✓ |
| `THUỐC` | thuốc điều trị | **RxNorm** | ✓ |

**Đa mã hợp lệ:** ví dụ đề `"bệnh trào ngược dạ dày - thực quản"` → `{K21.0, K21.9}`. Chẩn đoán không đặc hiệu ⇒ tập nhiều subcode.

---

## 2. Metric — reverse-engineer, vì nó lái toàn bộ thiết kế

```
final = 0.30·text_score + 0.30·assertions_score + 0.40·candidates_score
```

- `text_score   = (1/N) · Σ_i (1 − WER(i))`
- `assertions_score = (1/N) · Σ_i J_assert(i)`
- `candidates_score = [ Σ_i J_cand(i)·W_i ] / [ Σ_i W_i ]`, với `W_i = Σ_k (len(gt_cand(k)) + 1)` trên các concept k của sample i.
- **Jaccard rỗng:** `gt=∅ ∧ pred=∅ → 1`; `gt=∅ ∧ pred≠∅ → 0`; còn lại `|∩|/|∪|`.
- **Sai type:** concept bị tính **2 lần** (một orphan GT + một orphan pred), mỗi lần **0 điểm cả 3 metric**.

### 2.1 Hệ quả không thể thương lượng (mỗi cái → 1 quyết định thiết kế)

| # | Tính chất metric | Quyết định |
|---|---|---|
| a | `WER=(S+D+I)/M` **không chặn trên 1** → 1 insertion có thể làm `1−WER < 0` | **Precision-first.** Không chắc thì **bỏ**, đừng đoán bừa |
| b | `gt=∅ ∧ pred≠∅ → J=0` | **Không bao giờ** xuất `candidates` cho triệu chứng/tên XN/kết quả XN |
| c | Jaccard ≠ recall@k: `J({A,B}\|{A})=0.5 < J({A}\|{A})=1` | Thuốc: **xuất đúng 1 mã tốt nhất**. Chẩn đoán: xuất **đúng tập** |
| d | Sai type = 0×3×2 lần | **Type accuracy là ưu tiên #1**; section-aware bắt buộc |
| e | `candidates` weight theo `len(gt)+1` | **Chẩn đoán đa-mã có trọng số cao nhất** → dồn đầu tư vào ICD-10 |
| f | tên XN + kết quả XN chỉ vào `text_score` nhưng **rất nhiều** (~229 bullet lab) | Vẫn phải trích tốt: bỏ sót = deletion hàng loạt kéo WER |

### 2.2 Điểm mơ hồ của metric — PHẢI cài 2 cách và test cả 2 (test toán)
- **WER align thế nào?** (i) nối toàn bộ `text` theo thứ tự position rồi 1 WER/sample, hay (ii) WER per-concept sau khi align rồi trung bình. Công thức viết `WER(i)` đơn trị/sample ⇒ **mặc định (i)**, nhưng để **cờ chuyển đổi** và unit-test cả hai.
- **Align concept theo gì?** `(type, text)` exact hay `(type, position-overlap)`. Ghi chú "sai type tính 2 lần" ⇒ key có type. Cài cả hai, mặc định overlap-position + type.
- → **Không đoán. Ghi câu hỏi cho BTC (§11) và giữ scorer modular.**

---

## 3. Nguyên tắc thiết kế (chống hardcode, chống drift)

### 3.1 BẤT BIẾN offset (hard invariant, có test phủ 100 file)
> Với mọi concept xuất ra: `raw[start:end] == text`. Không thỏa ⇒ **loại**.

Điều này khử hoàn toàn mơ hồ "xuất text gì": text **được quyết định hoàn toàn bởi span**. Mọi normalize chỉ dùng để *tìm* biên, **không bao giờ ghi vào output**. Normalizer phải giữ `map[i_clean] → i_raw`.

### 3.2 Được phép hardcode (spec-defined, chắc chắn đúng)
5 nhãn type; 3 nhãn assertion; trọng số `0.3/0.3/0.4`; công thức metric; JSON schema; danh sách section chuẩn (là **config**, không phải logic).

### 3.3 CẤM hardcode (phải induce/ingest/calibrate)
Lexicon thực thể (tra KB hoặc cảm sinh từ input); header thật (fuzzy match, config-driven); ngưỡng (calibrate trên dev); **bất kỳ tri thức nào về nội dung 100 file test** (= leakage BTC dựng private test để bắt).

### 3.4 Chống drift theo example đề — build theo TEST SET
Đo thực tế trên 100 file test (khác example của đề):

| | Example đề | **Test set thật** | Theo |
|---|---|---|---|
| Format lab | `WBC:14,43` (dấu `:`) | `- troponin 0.01` (space) | test |
| Thập phân | phẩy | **chấm (114) vs phẩy (12)** | chấm; phẩy `21,000` = phân cách nghìn |
| Thuốc | `Chlorpheniramine 0.4 MG/ML` chuẩn hoá | `metoprolol 25mg po bid`, `lasix` | dạng sig bẩn |

### 3.5 Data bẩn đã đo (28/100 file, 39 lỗi) — pipeline phải chịu được, không vá từng ca
Dính từ (`atenololtrong`, `klonopinclonidine` = 2 thuốc dính), dấu cách đôi (400 chỗ/86 file → lệch offset), cụm lặp (22 file), sai chính tả thuốc (`levafloxacin`, `doxycyclin`, `cotrimoxazol`), bullet `*` (file 36), vitals dồn (`VS98.3 12987 56 18 99RA`). → Xử lý bằng **thuật toán tổng quát** (Viterbi resegmentation, fuzzy), không regex vá điểm.

---

## 4. Kiến trúc pipeline

```
raw .txt
  │
  ▼
[S1] IO + Normalizer giữ offset ──────────────► map[i_clean→i_raw]
  │
  ▼
[S2] Section Segmenter (fuzzy 17 header / 90 sub → 8 canonical)
  │        └─ section prior (feature mạnh nhất cho type & isHistorical)
  ▼
[S3] Mention Detection (over-generate, recall-first)
  ├─ [S3a] NER model ≤9B (train)  ── triệu chứng / chẩn đoán / (xác nhận thuốc,lab)
  ├─ [S3b] Drug gazetteer + fuzzy ── RxNorm alias (vocab nhỏ: 57 INN + 13 brand)
  └─ [S3c] Lab extractor (regex+section) ── TÊN_XÉT_NGHIỆM + KẾT_QUẢ_XÉT_NGHIỆM
  │
  ▼
[S4] Type Resolver (section prior ⊕ model logit ⊕ nguồn mention)  ── quyết #1
  │
  ├────────────────────────┬───────────────────────────┐
  ▼                        ▼                           ▼
[S5] Assertion (ConText)  [S6a] RxNorm link (thuốc)   [S6b] ICD-10 link (chẩn đoán)
  rule, không train        retrieve→rerank, 1 mã        cross-lingual, ĐA-MÃ ★nặng nhất
  │                        │                           │
  └────────────────────────┴───────────────────────────┘
                           ▼
[S7] Assembler + Grounding check (raw[start:end]==text) + Writer JSON
                           ▼
[S8] Operating-point calibration (ngưỡng giữ/bỏ theo E[Δscore])
```

### S1 — IO + Normalizer giữ offset
Đọc UTF-8. Sinh "clean view" (gộp dấu cách đôi, tách token dính) **kèm mảng ánh xạ ngược**. Tách token dính bằng **Viterbi** trên từ điển (âm tiết VN ∪ gazetteer thuốc ∪ abbrev lab), tối đa hoá xác suất unigram — dùng để **đề xuất biên**, không sửa text. Test: round-trip offset chính xác trên cả 100 file.

### S2 — Section Segmenter
17 biến thể header × 90 sub-section → ~8 canonical (`TIEN_SU`, `HIEN_TAI`, `SYMPTOM`, `DIAGNOSIS`, `DRUG`, `LAB`, `PROC`, `EVENT`). RapidFuzz `token_set_ratio` trên chuỗi bỏ dấu (đo được: gán đúng **79%** bullet). Danh sách canonical để trong `configs/sections.yaml`. Xử lý bullet `-` và `*`.

### S3 — Mention Detection (recall-first, over-generate có chủ đích)
- **S3a NER model (train, §6):** trục chính cho triệu chứng/chẩn đoán tiếng Việt (span + type). Encoder fine-tuned cho biên span chính xác (bảo vệ WER).
- **S3b Drug gazetteer:** vocab thuốc cực nhỏ (57 INN + 13 brand/132KB) ⇒ **từ điển thắng model học**. Exact → char-3gram TF-IDF → RapidFuzz edit≤2 (bắt `levafloxacin`, `doxycyclin`). Regex ghép sig theo sau (`25mg po bid`).
- **S3c Lab extractor:** cặp `tên [±:] giá_trị[đơn_vị]` trong section LAB. Parser số: `.` thập phân, `,` phân cách nghìn (`21,000`), vitals dồn tách theo pattern. `TÊN_XÉT_NGHIỆM` = tên (có thể kèm mô tả trong ngoặc); `KẾT_QUẢ_XÉT_NGHIỆM` = giá trị+đơn vị.

### S4 — Type Resolver (quyết định #1 vì phạt nặng nhất)
Kết hợp: **section prior** (LAB→tên/kết quả XN; DRUG→thuốc; SYMPTOM→triệu chứng; DIAGNOSIS/chronic→chẩn đoán) ⊕ logit model ⊕ nguồn mention. Guard phản ví dụ đã biết (`"ecg bình thường"` nằm dưới mục triệu chứng nhưng **không** phải triệu chứng; `"kali"` = thuốc ở mục thuốc / tên XN ở mục lab).

### S5 — Assertion (ConText rule, không train)
Lý do rule: `isHistorical` 361 cue/94 file gần như do section; `isFamily` chỉ 10/7 → không đủ train; GT nhiều khả năng sinh bằng ConText nên rule-bắt-chước-rule khớp cao; phủ định là điểm yếu cố hữu của LLM zero-shot. Cài: cue lexicon VN (`không/phủ nhận/âm tính/chưa` ; `tiền sử/từng bị/đã dùng` ; `gia đình/bố/mẹ`) + scope window + terminator. Chỉ áp cho 3 type có assertion.

### S6a — RxNorm linking (thuốc)
Alias index từ RxNorm RRF (`IN/BN/SCD/SBD`) → retrieve (exact→ngram→fuzzy) → rerank **SapBERT** (dùng thẳng) → **≤9B reranker** chỉ cho ca mơ hồ (brand, tên dính). Granularity: **có liều+dạng → SCD; thiếu → ingredient RxCUI** (ví dụ đề: `nystatin→7597` ingredient; `Chlorpheniramine 0.4 MG/ML→360047` SCD). Xuất **1 mã**.

### S6b — ICD-10 linking (chẩn đoán) ★ đòn bẩy lớn nhất (weight `len(gt)+1`)
Chẩn đoán **tiếng Việt** → ICD-10. Cross-lingual: **SapBERT-XLM-R** embed trên alias ICD-10 (VN + EN) → retrieve → rerank. **Chiến lược đa-mã:** text đặc hiệu → 1 mã lá; text **không đặc hiệu** → tập subcode hợp lý (calibrate trên GT tay). Đây là sub-bài-toán khó nhất và nặng điểm nhất — đầu tư R&D lớn nhất ở đây.

### S7 — Assembler + Writer
Grounding check (§3.1) loại mọi concept lệch. Sort theo `start`. **Không dedupe** (cụm lặp giữ nguyên, position khác nhau). Bỏ key `candidates` cho 3 type không map. Ghi JSON đúng schema.

### S8 — Calibration điểm vận hành
Với xác suất hiệu chỉnh `p` mỗi concept, viết `E[Δfinal | giữ]` theo đúng `0.3/0.3/0.4` → chọn ngưỡng `p*` tại điểm đổi dấu. Trực giác: `p* > 0.5` (giữ nhầm phạt 3 trường, bỏ sót chỉ 1 deletion). Calibrate trên dev synthetic + 15 file tay.

---

## 5. Model & tooling — tuân thủ ≤9B (inference)

| Vai trò | Model | Params | ≤9B |
|---|---|---|---|
| NER 5-type (chính) | **ViHealthBERT-base** (PhoBERT+25M câu y tế VN, SOTA NER y tế VN) | ~135M | ✓ |
| Rerank linking | **SapBERT** / SapBERT-XLM-R | 110M / 550M | ✓ |
| Reranker ca khó / generative fallback | **Qwen3-8B** / SeaLLMs-v3-7B / Sailor2-8B | 7–8B | ✓ |
| MT sinh data (**offline**, không tính inference) | NLLB-1.3B / envit5 | — | n/a |
| ⚠️ CẤM | **Gemma-2-9B ≈ 9.24B** | 9.24B | ✗ vượt trần |

Hạ tầng: Windows 11 + CUDA local; Python venv; PyTorch CUDA; HF Transformers; RapidFuzz; spaCy/medspaCy (ConText); FAISS (retrieve). Corpus 132KB ⇒ inference cả 100 file trong **phút**, compute không phải ràng buộc.

---

## 6. Sinh data huấn luyện (phần "ngoài lời giải chính" đề yêu cầu)

Đề: *"cần dùng giải pháp ngoài lời giải chính để tạo thêm dữ liệu huấn luyện."* Hai nguồn, bổ trợ nhau:

1. **Translate-train (EasyProject label projection, Findings ACL 2023):** lấy corpus lâm sàng Anh có nhãn concept/assertion → chèn marker `[...]` quanh entity → dịch EN→VI bằng MT self-host → chiếu span về. Vượt word-alignment vì giữ biên span. Map nhãn Anh → 5 type VN.
2. **LLM sinh bệnh án VN tổng hợp:** prompt LLM mạnh (offline) sinh note đúng **template section đã đo** (17 header), phủ đủ 5 type, **bơm đúng nhiễu đã đo** (dính từ, chấm/phẩy, cụm lặp, sig bẩn). Nhãn có sẵn do sinh có kiểm soát.

**Ràng buộc dữ liệu (xử lý sớm):** n2c2/i2b2/MIMIC dưới DUA PhysioNet **cấm redistribute**, mà top-15 phải nộp "data nhóm dùng". Ưu tiên nguồn công khai (BC5CDR, NCBI-Disease, ADE, CADEC) + note tự sinh. **Hỏi BTC** trước khi đầu tư nguồn hạn chế.

**Không leakage:** 100 file test **không** được gán nhãn để train. Lexicon cảm sinh lúc inference (hàm thuần của input) thì hợp lệ.

---

## 7. Chiến lược test (bắt buộc — 4 nhóm)

| Nhóm | Nội dung | Tiêu chí |
|---|---|---|
| **Toán (metric)** | edit-distance WER trên cặp đã biết; 4 ca Jaccard rỗng; công thức weight `len(gt)+1`; double-count sai type; **tái tạo đúng ví dụ đề** | pass tuyệt đối; 2 biến thể WER (§2.2) đều test |
| **Data-processing** | bất biến offset `raw[start:end]==text` trên **cả 100 file**; round-trip normalizer; segmenter trên 17 header; parser số (`.`/`,`/`21,000`); Viterbi tách 6 ca thuốc dính | 100% pass |
| **Logic** | assertion (scope phủ định, family, historical) trên câu dựng; type resolver trên phản ví dụ (`ecg bình thường`, `kali`) | pass |
| **Integration** | e2e 100 file → JSON hợp schema, position hợp lệ, type ∈ enum | 100/100 |

Nguyên tắc: **metric harness phải xong & xanh trước khi viết bất kỳ extractor nào** — nó là oracle cho mọi vòng lặp.

---

## 8. Kế hoạch thực thi theo Phase (data + metric TRƯỚC)

| Phase | Việc | Deliverable | Gate |
|---|---|---|---|
| **P0** | **Metric harness** + unit test toán | `src/metric/scorer.py` tái tạo ví dụ đề | test toán xanh |
| **P1** | **Data layer**: loader, normalizer-offset, segmenter, ingest RxNorm+ICD-10; **gán tay 15 file (chỉ để ĐO)** | `src/io`, `src/segment`, KB index, `dev15/` | data-test xanh; bất biến offset 100/100 |
| **P2** | **Baseline dictionary+rule+section** (chưa train) → chạy scorer | **con số baseline trung thực** | có điểm đo được đầu tiên |
| **P3** | **Sinh data** (translate-train + LLM synth) + inject nhiễu | `datagen/`, corpus train | dev synthetic hợp lệ |
| **P4** | **Train NER** ViHealthBERT 5-type; đo vs baseline | `src/ner/`, weights | > baseline trên dev15 |
| **P5** | **Linking**: RxNorm + **ICD-10 đa-mã** (đầu tư lớn nhất) | `src/link/` | candidates_score tăng |
| **P6** | **Assertion ConText** + **calibrate ngưỡng** | `src/assert_`, `configs/thresholds.yaml` | final tăng, precision-first |
| **P7** | **Đóng gói tái lập** (README, weights, script) cho top-15 | repo chạy 1 lệnh | BTC dựng lại được |

Ưu tiên theo trọng số điểm: **ICD-10 đa-mã (0.4) > type accuracy > assertion (0.3) ≈ text (0.3)**.

---

## 9. Cấu trúc repo (`C:\Users\Admin\Downloads\Medical\`)

```
Medical/
├── README.md                 # cài đặt + 1 lệnh chạy (yêu cầu BTC)
├── requirements.txt
├── configs/
│   ├── sections.yaml         # 17→8 canonical (config, không hardcode logic)
│   ├── assertion_cues.yaml
│   └── thresholds.yaml       # calibrate, không cố định
├── data/
│   ├── kb/rxnorm/  kb/icd10_vn/
│   ├── synthetic/            # data train tự sinh
│   └── dev15/                # 15 file gán tay — CHỈ để đo, KHÔNG train
├── src/
│   ├── metric/scorer.py      # P0, oracle
│   ├── io/{loader,normalizer,offsets}.py
│   ├── segment/sections.py
│   ├── ner/{model,gazetteer,lab_extractor,train}.py
│   ├── link/{rxnorm,icd10,sapbert}.py
│   ├── assert_/context.py
│   └── assemble/{writer,grounding}.py
├── datagen/{translate_train,llm_synth,noise_inject}.py
├── tests/{test_metric,test_offsets,test_segment,test_parse,test_assert}.py
└── scripts/{run_inference,train,evaluate}.py
```

---

## 10. Rủi ro & suy biến
- **ICD-10 đa-mã**: sub-bài khó nhất, nặng điểm nhất, ít tiền lệ. Rủi ro cao → làm sớm (P5), calibrate trên dev15.
- **Không có train/dev gán sẵn** → nguy cơ overfit 15 file tay. Kỷ luật: tune ≤1–2 vòng, quyết định trên dev synthetic. *(Bài học đã trả giá: overfit tập nhỏ → val đẹp, thực tế sập.)*
- **Suy biến êm:** nếu BTC phát train set → thay data synthetic bằng data thật, kiến trúc giữ nguyên. Nếu ICD-10 VN không có/không được redistribute → fallback string-match + WHO ICD-10 EN qua dịch.

## 11. Câu hỏi cho BTC (đắt nếu đoán sai — hỏi sớm)
1. **WER tính thế nào** (nối chuỗi/sample hay per-concept trung bình) và **align concept theo `(type,text)` hay `(type,position)`**?
2. **Quy tắc đa-mã ICD**: khi text không đặc hiệu, GT lấy bao nhiêu subcode? Xin vài ví dụ GT.
3. **Chuẩn hoá text tên XN**? (ví dụ đề `WBC`→`TWBC`) — ảnh hưởng WER.
4. **ICD-10 tiếng Việt**: bản nào được phép dùng và **redistribute kèm code** nộp top-15?
5. Với **từ dính** (`atenololtrong`), GT ghi `text`/`position` là toàn token hay sub-span thuốc?

---

## 12. Nguồn tham khảo (đã kiểm chứng)
- ViHealthBERT — Pre-trained LM cho y tế tiếng Việt (LREC 2022)
- SapBERT / SapBERT-XLM-R — self-alignment, cross-lingual biomedical EL (NAACL/ACL 2021)
- BioELX — retrieve→rerank, mention-anchored prompting, R@1 50.1→54.8 (2026)
- EasyProject — "Frustratingly Easy Label Projection", translate-train (Findings ACL 2023)
- ViPubmed — Enriching Biomedical Knowledge via Large-Scale Translation (dịch quy mô lớn EN→VI)
- Frontiers Digital Health 2025 — BERT fine-tuned > zero-shot cho symptom/negation lâm sàng

> **Lưu ý trung thực:** GLiNER-multi/GLiNER-BioMed **không** hỗ trợ tiếng Việt (fine-tune trên Pile-NER tiếng Anh) → đã loại khỏi thiết kế. Các số của BioELX là trên **entity linking cross-lingual**, chưa được chứng minh cho **verify span NER tiếng Việt** → coi là giả thuyết, phải đo ở P4–P5.
