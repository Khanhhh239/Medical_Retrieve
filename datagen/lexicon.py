# -*- coding: utf-8 -*-
"""
Seed lexicon cho sinh data synthetic (P3) và test linker (P5).

Tri thức y khoa TỔNG QUÁT (không lấy từ 100 file test — không leakage). Mã ICD-10 /
RxNorm là mã CÔNG KHAI, phổ biến, dùng làm seed minh hoạ. TRƯỚC KHI CHẠY THẬT: thay
bằng RxNorm RRF + ICD-10 đầy đủ (xem README §Train / data/kb/).
"""
from __future__ import annotations

# (tên hiển thị tiếng Việt, mã ICD-10)  — bệnh
DISEASES = [
    ("tăng huyết áp", "I10"),
    ("đái tháo đường type 2", "E11"),
    ("bệnh trào ngược dạ dày thực quản", "K21.9"),
    ("viêm phổi", "J18.9"),
    ("bệnh thận mạn", "N18.9"),
    ("rung nhĩ", "I48.91"),
    ("bệnh phổi tắc nghẽn mạn tính", "J44.9"),
    ("rối loạn lipid máu", "E78.5"),
    ("suy tim", "I50.9"),
    ("nhồi máu cơ tim", "I21.9"),
    ("hen phế quản", "J45.909"),
    ("viêm dạ dày", "K29.70"),
    ("thiếu máu", "D64.9"),
    ("suy giáp", "E03.9"),
    ("trầm cảm", "F32.9"),
    ("nhiễm trùng đường tiết niệu", "N39.0"),
    ("loét dạ dày", "K25.9"),
    ("xơ gan", "K74.60"),
    ("nhồi máu não", "I63.9"),
    ("ung thư đại tràng", "C18.9"),
]

# (tên thuốc, RxCUI ingredient)  — thuốc (tên EN giữ nguyên như trong data)
DRUGS = [
    ("aspirin", "1191"),
    ("acetaminophen", "161"),
    ("amlodipine", "17767"),
    ("metoprolol", "6918"),
    ("atorvastatin", "83367"),
    ("lisinopril", "29046"),
    ("furosemide", "4603"),
    ("omeprazole", "7646"),
    ("amoxicillin", "723"),
    ("metformin", "6809"),
    ("clopidogrel", "32968"),
    ("warfarin", "11289"),
    ("levothyroxine", "10582"),
    ("pantoprazole", "40790"),
    ("azithromycin", "18631"),
    ("ciprofloxacin", "2551"),
    ("prednisone", "8640"),
    ("gabapentin", "25480"),
    ("sertraline", "36437"),
    ("insulin", "5856"),
]

SIGS = ["10 mg po daily", "25 mg po bid", "40 mg po daily", "500 mg po q12h",
        "81 mg po daily", "5 mg po qhs", "20 mg iv daily", "50 mg po bid"]

SYMPTOMS = [
    "đánh trống ngực", "khó thở", "sốt", "ho", "ho khan", "ho có đờm", "đau ngực",
    "đau bụng", "buồn nôn", "nôn", "chóng mặt", "đau đầu", "mệt mỏi", "táo bón",
    "tiêu chảy", "phù chân", "đau thượng vị", "ợ hơi", "chán ăn", "sụt cân",
    "khó nuốt", "tê tay chân", "đau lưng", "mất ngủ", "lo âu", "hồi hộp",
    "vã mồ hôi", "đau khớp", "khàn tiếng", "chảy máu cam",
]

# (tên xét nghiệm, đơn vị, khoảng giá trị mẫu)
LABS = [
    ("WBC", "", (4.0, 18.0)),
    ("hemoglobin", "g/dL", (7.0, 15.0)),
    ("creatinine", "mg/dL", (0.6, 6.0)),
    ("glucose", "mg/dL", (70.0, 400.0)),
    ("kali", "mmol/L", (2.8, 6.5)),
    ("natri", "mmol/L", (128.0, 145.0)),
    ("troponin", "ng/mL", (0.0, 2.0)),
    ("bnp", "pg/mL", (20.0, 5000.0)),
    ("alt", "U/L", (10.0, 400.0)),
    ("ast", "U/L", (10.0, 400.0)),
    ("inr", "", (0.9, 3.5)),
    ("lactate", "mmol/L", (0.5, 5.0)),
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
