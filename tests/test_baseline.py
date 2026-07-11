# -*- coding: utf-8 -*-
"""Test cơ chế baseline extractor (P2). Đây là test CƠ CHẾ, không phải đo chất lượng."""
from src.io.loader import Document
from src.extract.baseline import extract
from src.io.offsets import is_grounded

RAW = (
    "1. Tiền sử bệnh\n"
    "Thuốc trước khi nhập viện:\n"
    "- metoprolol 25mg po bid\n"
    "- doxycycline cho viêm tuyến mồ hôi\n"
    "Các bệnh lý mãn tính\n"
    "- Tăng huyết áp\n"
    "- bệnh thận mạn, không đặc hiệu\n"
    "2. Bệnh sử hiện tại\n"
    "Lý do nhập viện: sốt, đau thượng vị\n"
    "Các triệu chứng hiện tại\n"
    "- đánh trống ngực\n"
    "- Không khó thở\n"
    "3. Đánh giá tại bệnh viện\n"
    "Kết quả xét nghiệm:\n"
    "- kali 6.3\n"
    "- troponin 0.01\n"
)


def _doc():
    return Document("t", RAW, "t.txt")


def _by_type(concepts):
    out = {}
    for c in concepts:
        out.setdefault(c.type, []).append(c.text)
    return out


def test_all_extracted_spans_grounded():
    for c in extract(_doc()):
        assert is_grounded(RAW, c.text, c.position), (c.text, c.position)


def test_extracts_expected_types():
    by = _by_type(extract(_doc()))
    assert "metoprolol 25mg po bid" in by["THUỐC"]
    assert "doxycycline" in by["THUỐC"]                       # đuôi chỉ định đã cắt
    assert "Tăng huyết áp" in by["CHẨN_ĐOÁN"]
    assert "bệnh thận mạn, không đặc hiệu" in by["CHẨN_ĐOÁN"]  # bullet KHÔNG tách phẩy
    assert "đánh trống ngực" in by["TRIỆU_CHỨNG"]
    assert "khó thở" in by["TRIỆU_CHỨNG"]                      # cue phủ định đã cắt
    assert "sốt" in by["TRIỆU_CHỨNG"]                          # list sau ':' đã tách
    assert "đau thượng vị" in by["TRIỆU_CHỨNG"]
    assert "kali" in by["TÊN_XÉT_NGHIỆM"]
    assert "6.3" in by["KẾT_QUẢ_XÉT_NGHIỆM"]


def test_no_qualifier_leaked_as_concept():
    # 'không đặc hiệu' KHÔNG được thành 1 CHẨN_ĐOÁN riêng
    dx = _by_type(extract(_doc())).get("CHẨN_ĐOÁN", [])
    assert "không đặc hiệu" not in dx


def test_negation_assertion_on_symptom():
    kt = [c for c in extract(_doc()) if c.text == "khó thở"][0]
    assert "isNegated" in kt.assertions


def test_historical_from_drug_section():
    met = [c for c in extract(_doc()) if c.text.startswith("metoprolol")][0]
    assert "isHistorical" in met.assertions


def test_baseline_no_candidates_yet():
    for c in extract(_doc()):
        assert c.candidates == ()        # chưa link — chờ P5


def test_symptom_colon_takes_part_before_colon():
    # 'X: mô tả' -> khái niệm là X (fix bug #2)
    raw = "Các triệu chứng hiện tại\n- đánh trống ngực: còn cảm giác khi nhập viện\n"
    by = _by_type(extract(Document("t", raw, "")))
    assert "đánh trống ngực" in by.get("TRIỆU_CHỨNG", [])
    assert not any("cảm giác" in t for t in by.get("TRIỆU_CHỨNG", []))


def test_lab_colon_form_keeps_name():
    # 'WBC:14,43' (dấu ':' là dữ liệu) -> giữ được tên xét nghiệm (fix bug #2)
    for raw in ["Kết quả xét nghiệm\n- WBC:14,43\n",
                "Kết quả xét nghiệm\nWBC:14,43\n"]:
        by = _by_type(extract(Document("t", raw, "")))
        assert "WBC" in by.get("TÊN_XÉT_NGHIỆM", []), raw
        assert "14,43" in by.get("KẾT_QUẢ_XÉT_NGHIỆM", []), raw


def test_lab_space_form_still_works():
    raw = "Kết quả xét nghiệm\n- kali 6.3\n"
    by = _by_type(extract(Document("t", raw, "")))
    assert "kali" in by.get("TÊN_XÉT_NGHIỆM", [])
    assert "6.3" in by.get("KẾT_QUẢ_XÉT_NGHIỆM", [])


def test_bullet_style_subheaders_extracted():
    # file 36 style: tiêu đề mục là bullet '*', item là bullet lồng sâu hơn (fix bug rỗng)
    raw = ("1. Tiền sử bệnh\n"
           "    *   Thuốc trước khi nhập viện\n"
           "        *   azathioprine\n"
           "        *   prograf (dose decreased from 5mg bid)\n"
           "3. Đánh giá tại bệnh viện\n"
           "    *   Kết quả xét nghiệm\n"
           "        *   creatinine 5.1\n"
           "        *   k 5.8\n")
    by = _by_type(extract(Document("t", raw, "")))
    assert "azathioprine" in by.get("THUỐC", [])
    assert "prograf" in by.get("THUỐC", [])                 # paren đã bỏ
    assert "creatinine" in by.get("TÊN_XÉT_NGHIỆM", [])
    assert "5.1" in by.get("KẾT_QUẢ_XÉT_NGHIỆM", [])
    # tiêu đề (là bullet) KHÔNG được nhả thành concept
    assert "Thuốc trước khi nhập viện" not in by.get("THUỐC", [])
    assert "Kết quả xét nghiệm" not in by.get("TÊN_XÉT_NGHIỆM", [])


def test_free_narrative_yields_little():
    # ví dụ BTC là free-narrative (không bullet/section) -> baseline gần như rỗng.
    # Đây là GIỚI HẠN CÓ CHỦ ĐÍCH: free text là việc của model NER (P4).
    narr = ("Bệnh nhân nam 70 tuổi bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, "
            "được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản.")
    cs = extract(Document("n", narr, "n.txt"))
    assert len(cs) <= 1
