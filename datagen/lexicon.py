# -*- coding: utf-8 -*-
"""
Seed lexicon cho sinh data synthetic (P3) và test linker (P5).

Tri thức y khoa TỔNG QUÁT (KHÔNG lấy từ 100 file test — không leakage). Mã ICD-10 /
RxNorm là mã CÔNG KHAI phổ biến, dùng làm seed. Mục tiêu của lexicon LỚN: cho NER
nhiều TỪ VỰNG đa dạng để học (span + type); mã chỉ phụ (NER không dự đoán mã).
TRƯỚC KHI CHẠY THẬT: mã thuốc lấy từ RxNav API (scripts/build_kb.py), ICD từ file đầy đủ.
"""
from __future__ import annotations

# (tên bệnh tiếng Việt, mã ICD-10)
DISEASES = [
    ("tăng huyết áp", "I10"), ("đái tháo đường type 2", "E11"),
    ("bệnh trào ngược dạ dày thực quản", "K21.9"), ("viêm phổi", "J18.9"),
    ("bệnh thận mạn", "N18.9"), ("rung nhĩ", "I48.91"),
    ("bệnh phổi tắc nghẽn mạn tính", "J44.9"), ("rối loạn lipid máu", "E78.5"),
    ("suy tim", "I50.9"), ("nhồi máu cơ tim", "I21.9"), ("hen phế quản", "J45.909"),
    ("viêm dạ dày", "K29.70"), ("thiếu máu", "D64.9"), ("suy giáp", "E03.9"),
    ("cường giáp", "E05.90"), ("trầm cảm", "F32.9"), ("rối loạn lo âu", "F41.9"),
    ("nhiễm trùng đường tiết niệu", "N39.0"), ("loét dạ dày", "K25.9"),
    ("xơ gan", "K74.60"), ("viêm gan B", "B18.1"), ("viêm gan C", "B18.2"),
    ("nhồi máu não", "I63.9"), ("xuất huyết não", "I61.9"),
    ("ung thư đại tràng", "C18.9"), ("ung thư phổi", "C34.90"),
    ("ung thư vú", "C50.919"), ("ung thư dạ dày", "C16.9"),
    ("ung thư gan", "C22.9"), ("ung thư tuyến tiền liệt", "C61"),
    ("sỏi thận", "N20.0"), ("sỏi mật", "K80.20"), ("viêm tụy cấp", "K85.90"),
    ("viêm ruột thừa", "K37"), ("thoát vị đĩa đệm", "M51.9"),
    ("viêm khớp dạng thấp", "M06.9"), ("gout", "M10.9"), ("loãng xương", "M81.0"),
    ("bệnh mạch vành", "I25.10"), ("suy thận cấp", "N17.9"),
    ("động kinh", "G40.909"), ("parkinson", "G20"), ("sa sút trí tuệ", "F03.90"),
    ("béo phì", "E66.9"), ("suy dinh dưỡng", "E46"),
    ("viêm phế quản cấp", "J20.9"), ("lao phổi", "A15.0"),
    ("nhiễm khuẩn huyết", "A41.9"), ("viêm màng não", "G03.9"),
    ("tăng sản tuyến tiền liệt", "N40.0"), ("suy tim sung huyết", "I50.0"),
]

# (tên thuốc EN, RxCUI ingredient — placeholder; thật lấy từ RxNav)
DRUGS = [
    ("aspirin", "1191"), ("acetaminophen", "161"), ("amlodipine", "17767"),
    ("metoprolol", "6918"), ("atorvastatin", "83367"), ("rosuvastatin", "301542"),
    ("simvastatin", "36567"), ("lisinopril", "29046"), ("losartan", "52175"),
    ("valsartan", "69749"), ("furosemide", "4603"), ("spironolactone", "9997"),
    ("hydrochlorothiazide", "5487"), ("omeprazole", "7646"), ("pantoprazole", "40790"),
    ("esomeprazole", "283742"), ("amoxicillin", "723"), ("azithromycin", "18631"),
    ("ciprofloxacin", "2551"), ("levofloxacin", "82122"), ("ceftriaxone", "2193"),
    ("metronidazole", "6922"), ("metformin", "6809"), ("insulin", "5856"),
    ("gliclazide", "25789"), ("clopidogrel", "32968"), ("warfarin", "11289"),
    ("apixaban", "1364430"), ("rivaroxaban", "1114195"), ("enoxaparin", "67108"),
    ("levothyroxine", "10582"), ("prednisone", "8640"), ("prednisolone", "8638"),
    ("gabapentin", "25480"), ("pregabalin", "187832"), ("sertraline", "36437"),
    ("fluoxetine", "4493"), ("diazepam", "3322"), ("tramadol", "10689"),
    ("morphine", "7052"), ("salbutamol", "435"), ("budesonide", "19831"),
    ("allopurinol", "519"), ("digoxin", "3407"), ("nitroglycerin", "4917"),
]

# Thuốc thường gặp (tên generic) để PRE-CACHE rộng qua RxNav — phủ cả private test.
COMMON_DRUGS = [
    "aspirin", "acetaminophen", "ibuprofen", "naproxen", "diclofenac", "celecoxib",
    "tramadol", "morphine", "oxycodone", "fentanyl", "codeine", "gabapentin", "pregabalin",
    "amlodipine", "nifedipine", "diltiazem", "verapamil", "lisinopril", "enalapril",
    "ramipril", "captopril", "losartan", "valsartan", "telmisartan", "irbesartan",
    "metoprolol", "atenolol", "bisoprolol", "carvedilol", "propranolol", "labetalol",
    "furosemide", "bumetanide", "hydrochlorothiazide", "spironolactone", "indapamide",
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "ezetimibe", "fenofibrate",
    "clopidogrel", "ticagrelor", "warfarin", "apixaban", "rivaroxaban", "dabigatran",
    "enoxaparin", "heparin", "digoxin", "amiodarone", "nitroglycerin", "isosorbide",
    "metformin", "gliclazide", "glimepiride", "glipizide", "sitagliptin", "empagliflozin",
    "dapagliflozin", "insulin glargine", "insulin aspart", "insulin", "pioglitazone",
    "omeprazole", "pantoprazole", "esomeprazole", "lansoprazole", "rabeprazole", "ranitidine",
    "famotidine", "domperidone", "metoclopramide", "ondansetron", "loperamide", "lactulose",
    "amoxicillin", "amoxicillin clavulanate", "ampicillin", "cephalexin", "cefuroxime",
    "ceftriaxone", "cefepime", "cefixime", "azithromycin", "clarithromycin", "erythromycin",
    "ciprofloxacin", "levofloxacin", "moxifloxacin", "doxycycline", "metronidazole",
    "clindamycin", "vancomycin", "meropenem", "piperacillin tazobactam", "gentamicin",
    "trimethoprim sulfamethoxazole", "nitrofurantoin", "fluconazole", "acyclovir",
    "levothyroxine", "carbimazole", "methimazole", "prednisone", "prednisolone",
    "dexamethasone", "hydrocortisone", "methylprednisolone", "salbutamol", "salmeterol",
    "ipratropium", "tiotropium", "budesonide", "fluticasone", "montelukast", "theophylline",
    "sertraline", "fluoxetine", "escitalopram", "paroxetine", "amitriptyline", "mirtazapine",
    "diazepam", "lorazepam", "alprazolam", "clonazepam", "zolpidem", "quetiapine",
    "risperidone", "olanzapine", "haloperidol", "levetiracetam", "valproate", "phenytoin",
    "carbamazepine", "allopurinol", "colchicine", "levodopa", "donepezil", "memantine",
    "tamsulosin", "finasteride", "furosemide", "calcium carbonate", "vitamin d",
    "ferrous sulfate", "folic acid", "potassium chloride", "magnesium",
]

SIGS = ["10 mg po daily", "25 mg po bid", "40 mg po daily", "500 mg po q12h",
        "81 mg po daily", "5 mg po qhs", "20 mg iv daily", "50 mg po bid",
        "100 mg po tid", "1 g iv q8h", "2.5 mg po daily", "12.5 mg po bid"]

SYMPTOMS = [
    "đánh trống ngực", "khó thở", "khó thở khi gắng sức", "sốt", "sốt cao", "ho",
    "ho khan", "ho có đờm", "ho ra máu", "đau ngực", "đau ngực trái", "đau bụng",
    "đau bụng vùng thượng vị", "đau quặn bụng", "buồn nôn", "nôn", "nôn ra máu",
    "chóng mặt", "hoa mắt", "đau đầu", "đau nửa đầu", "mệt mỏi", "suy nhược",
    "táo bón", "tiêu chảy", "đi ngoài phân đen", "phù chân", "phù toàn thân",
    "đau thượng vị", "ợ hơi", "ợ chua", "chán ăn", "sụt cân", "khó nuốt",
    "nuốt nghẹn", "tê tay chân", "yếu nửa người", "đau lưng", "đau khớp",
    "sưng khớp", "cứng khớp buổi sáng", "mất ngủ", "lo âu", "hồi hộp",
    "vã mồ hôi", "khàn tiếng", "chảy máu cam", "tiểu buốt", "tiểu rắt",
    "tiểu ra máu", "tiểu đêm", "bí tiểu", "vàng da", "vàng mắt", "ngứa da",
    "nổi mẩn", "co giật", "lú lẫn", "ngất", "run tay",
]

# (tên xét nghiệm, đơn vị, khoảng giá trị mẫu)
LABS = [
    ("WBC", "", (4.0, 18.0)), ("RBC", "", (3.0, 6.0)),
    ("hemoglobin", "g/dL", (7.0, 15.0)), ("hematocrit", "%", (25.0, 50.0)),
    ("platelet", "", (100.0, 450.0)), ("creatinine", "mg/dL", (0.6, 6.0)),
    ("ure", "mmol/L", (2.0, 20.0)), ("glucose", "mg/dL", (70.0, 400.0)),
    ("HbA1c", "%", (5.0, 12.0)), ("kali", "mmol/L", (2.8, 6.5)),
    ("natri", "mmol/L", (128.0, 145.0)), ("calci", "mmol/L", (1.8, 2.8)),
    ("troponin", "ng/mL", (0.0, 2.0)), ("bnp", "pg/mL", (20.0, 5000.0)),
    ("alt", "U/L", (10.0, 400.0)), ("ast", "U/L", (10.0, 400.0)),
    ("bilirubin", "mg/dL", (0.3, 10.0)), ("albumin", "g/dL", (2.0, 5.0)),
    ("inr", "", (0.9, 3.5)), ("crp", "mg/L", (0.0, 200.0)),
    ("lactate", "mmol/L", (0.5, 5.0)), ("cholesterol", "mg/dL", (120.0, 300.0)),
]

# header biến thể (đúng như đo trên data) để note trông thật
H_HISTORY = ["1. Tiền sử bệnh", "1. Tiền sử bệnh nội khoa", "1. Tiền sử bệnh lý"]
H_PRESENT = ["2. Tiền sử bệnh hiện tại", "2. Bệnh sử hiện tại", "2. Lịch sử bệnh hiện tại"]
H_ASSESS = ["3. Đánh giá tại bệnh viện", "3. Khám tại bệnh viện"]
SUB_DRUG = ["Thuốc trước khi nhập viện:", "Thuốc trước khi nhập viện lần này:"]
SUB_CHRONIC = ["Các bệnh lý mãn tính", "Các bệnh lý mạn tính"]
SUB_SYMPTOM = ["Các triệu chứng hiện tại", "Triệu chứng hiện tại"]
SUB_REASON = ["Lý do nhập viện:", "Lý do vào viện:"]
SUB_LAB = ["Kết quả xét nghiệm:", "Kết quả xét nghiệm"]
SUB_FAMILY = ["Tiền sử gia đình:", "Tiền sử gia đình"]

# Mẫu câu VĂN XUÔI (free-text) — để model học trích entity trong câu, không chỉ bullet.
# '{c}' = chỗ chèn khái niệm (được ghi span chính xác lúc sinh).
NARRATIVE_SYMPTOM = [
    "Bệnh nhân nhập viện vì tình trạng {c} tăng dần trong vài ngày qua.",
    "Khoảng một tuần nay bệnh nhân xuất hiện {c} kèm theo mệt mỏi.",
    "Bệnh nhân than phiền {c} liên tục, ảnh hưởng sinh hoạt.",
    "Cách vào viện 3 ngày, người bệnh bắt đầu có {c}.",
    "Trong quá trình theo dõi, bệnh nhân vẫn còn {c}.",
    "Bệnh nhân được đưa đến cấp cứu do {c} đột ngột.",
]
NARRATIVE_DIAGNOSIS = [
    "Bệnh nhân được chẩn đoán {c} sau khi thăm khám và làm xét nghiệm.",
    "Tiền sử ghi nhận {c} đã điều trị nhiều năm.",
    "Kết luận của bác sĩ là {c}, cần theo dõi thêm.",
]
