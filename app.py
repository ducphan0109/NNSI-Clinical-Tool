import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN CHUẨN LÂM SÀNG
# ==========================================
st.set_page_config(page_title="NNSI Clinical Decision Support System", page_icon="⚕️", layout="wide")

# CSS tinh chỉnh cho phong cách báo cáo y khoa (Medical Report Style)
st.markdown("""
    <style>
    .main-title {font-size: 32px !important; font-weight: 800; color: #1E3A8A; margin-bottom: 5px;}
    .sub-title {font-size: 16px !important; font-style: italic; color: #4B5563; margin-top: 0px; margin-bottom: 30px;}
    .clinical-box {background-color: #F8FAFC; padding: 20px; border-radius: 8px; border-left: 6px solid #1E3A8A; margin-bottom: 25px;}
    .metric-label {font-size: 16px !important; font-weight: 600; color: #475569;}
    .metric-value {font-size: 36px !important; font-weight: 700;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. TẢI TÀI NGUYÊN MÔ HÌNH (CACHE)
# ==========================================
@st.cache_resource
def load_resources():
    model = joblib.load("xgb_model.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    training_cols = joblib.load("training_columns.pkl")
    return model, explainer, training_cols

try:
    model, explainer, training_cols = load_resources()
except FileNotFoundError:
    st.error("Lỗi Hệ thống: Không tìm thấy tệp trọng số mô hình. Vui lòng kiểm tra lại cấu trúc thư mục.")
    st.stop()

# ==========================================
# 3. THUẬT TOÁN ĐỊNH LƯỢNG NNSI
# ==========================================
NNSI_RULES = {
    'DR1TCAFF': {'cutoff': 317.00, 'dir': '>=', 'points': 37},
    'DR1TVB6': {'cutoff': 1.18, 'dir': '<', 'points': 17},
    'DR1TZINC': {'cutoff': 10.82, 'dir': '<', 'points': 8},
    'DR1TMAGN': {'cutoff': 290.97, 'dir': '<', 'points': 7},
    'DR1TFOLA': {'cutoff': 339.50, 'dir': '<', 'points': 7},
    'Carb_Pro_Ratio': {'cutoff': 3.67, 'dir': '>=', 'points': 5},
    'DR1TPROT': {'cutoff': 94.31, 'dir': '<', 'points': 4},
    'DR1TIRON': {'cutoff': 15.69, 'dir': '<', 'points': 3},
    'DR1TCARB': {'cutoff': 280.27, 'dir': '<', 'points': 2},
    'DR1TVB12': {'cutoff': 1.00, 'dir': '<', 'points': 1}
}

def calculate_nnsi(patient_df):
    score = 0
    patient_data = patient_df.iloc[0].to_dict()
    for vi_chat, rule in NNSI_RULES.items():
        if vi_chat in patient_data:
            val = patient_data[vi_chat]
            if rule['dir'] == '>=' and val >= rule['cutoff']:
                score += rule['points']
            elif rule['dir'] == '<' and val < rule['cutoff']:
                score += rule['points']
                
    # Phân tầng nguy cơ lâm sàng
    if score <= 30: return score, "Nguy cơ Thấp (Low Risk Phenotype)", "#16A34A"
    elif score <= 60: return score, "Nguy cơ Trung bình (Moderate Risk Phenotype)", "#D97706"
    else: return score, "Nguy cơ Cao (High Risk Phenotype)", "#DC2626"

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# ==========================================
# 4. SIDEBAR: HỒ SƠ BỆNH ÁN ĐIỆN TỬ
# ==========================================
st.sidebar.markdown("### 🏛️ THÔNG TIN DỰ ÁN")
st.sidebar.info(
    "**Chủ nhiệm:** Phan Minh Đức\n\n"
    "**Cơ quan chủ quản:** Bộ môn Dinh dưỡng, Khoa Y - Trường Đại học Quốc tế Hồng Bàng (HIU)\n\n"
    "**Chuyên ngành:** Dinh dưỡng Thần kinh (Nutritional Neuroscience)"
)
st.sidebar.markdown("---")

st.sidebar.header("📋 THIẾT LẬP THÔNG SỐ LÂM SÀNG")
with st.sidebar.form("patient_form"):
    st.subheader("I. Chỉ số Nhân khẩu & Thể trạng")
    age = st.number_input("Tuổi (Năm)", min_value=18, max_value=80, value=30)
    gender = st.selectbox("Giới tính", options=[1, 2], format_func=lambda x: "Nam giới" if x==1 else "Nữ giới")
    race = st.selectbox("Mã Chủng tộc (NHANES)", options=[1, 2, 3, 4, 6], index=2)
    bmi = st.number_input("Chỉ số Khối cơ thể (BMI)", min_value=10.0, max_value=80.0, value=22.5)
    pir = st.number_input("Tỷ lệ Thu nhập/Nghèo đói (PIR)", min_value=0.0, max_value=5.0, value=2.0)
    phq9 = st.slider("Thang điểm Trầm cảm (PHQ-9)", min_value=0, max_value=27, value=0)
    
    st.subheader("II. Dữ liệu Vi chất Thần kinh (24h Recall)")
    caff = st.number_input("Lượng Caffeine (mg/ngày)", min_value=0.0, value=150.0)
    iron = st.number_input("Sắt - Iron (mg/ngày)", min_value=0.0, value=12.0)
    magn = st.number_input("Magiê - Magnesium (mg/ngày)", min_value=0.0, value=250.0)
    zinc = st.number_input("Kẽm - Zinc (mg/ngày)", min_value=0.0, value=20.0)
    vb6 = st.number_input("Vitamin B6 (mg/ngày)", min_value=0.0, value=1.5)
    vb12 = st.number_input("Vitamin B12 (mcg/ngày)", min_value=0.0, value=2.4)
    fola = st.number_input("Folate (mcg/ngày)", min_value=0.0, value=400.0)
    prot = st.number_input("Tổng Protein (g/ngày)", min_value=0.0, value=70.0)
    carb = st.number_input("Tổng Carbohydrate (g/ngày)", min_value=0.0, value=250.0)
    
    submitted = st.form_submit_button("Tiến hành Phân tích & Trích xuất Báo cáo")

# ==========================================
# 5. KHUNG HIỂN THỊ KẾT QUẢ TRUNG TÂM
# ==========================================
st.markdown('<p class="main-title">Hệ thống Đánh giá Lâm sàng: Chỉ số Dinh dưỡng Thần kinh - Giấc ngủ (NNSI)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Công cụ hỗ trợ quyết định lâm sàng ứng dụng thuật toán phi tuyến tính Survey-Weighted XGBoost</p>', unsafe_allow_html=True)

st.markdown("""
<div class="clinical-box">
    <strong>Tóm tắt Học thuật:</strong> Chỉ số NNSI (Nutritional Neuroscience Sleep Index) là một hệ thống điểm Nomogram được trích xuất thông qua kỹ thuật giải mã hộp đen SHAP (SHapley Additive exPlanations). Công cụ này định lượng hóa xác suất xuất hiện Kiểu hình Giấc ngủ Bất lợi (Adverse Sleep Phenotype) dựa trên hồ sơ chuyển hóa vi chất và các biến số kiểm soát dịch tễ.
</div>
""", unsafe_allow_html=True)

if submitted:
    # Tiền xử lý dữ liệu đầu vào
    input_data = {'RIDAGEYR': age, 'RIAGENDR': gender, 'RIDRETH3': race, 'BMXBMI': bmi, 'INDFMPIR': pir, 'PHQ9_Score': phq9, 'DR1TCAFF': caff, 'DR1TIRON': iron, 'DR1TMAGN': magn, 'DR1TZINC': zinc, 'DR1TVB6': vb6, 'DR1TVB12': vb12, 'DR1TFOLA': fola, 'DR1TPROT': prot, 'DR1TCARB': carb}
    df_patient = pd.DataFrame([input_data])
    df_patient['Carb_Pro_Ratio'] = df_patient['DR1TCARB'] / df_patient['DR1TPROT']
    
    # Định dạng biến phân loại (Categorical) để tránh lỗi XGBoost
    df_patient['RIAGENDR'] = pd.Categorical(df_patient['RIAGENDR'], categories=[1, 2])
    df_patient['RIDRETH3'] = pd.Categorical(df_patient['RIDRETH3'], categories=[1, 2, 3, 4, 6])
    df_patient = pd.get_dummies(df_patient, columns=['RIAGENDR', 'RIDRETH3'], drop_first=True, dtype=int)
    
    # Đồng bộ hóa cấu trúc cột với tập huấn luyện
    df_model_ready = df_patient.reindex(columns=training_cols, fill_value=0)
    
    # Trích xuất dự đoán
    prob_risk = model.predict_proba(df_model_ready)[0][1] * 100
    nnsi_score, risk_cat, color = calculate_nnsi(df_model_ready)
    
    # Render Báo cáo Chỉ số
    st.markdown("### 📊 BÁO CÁO KẾT QUẢ ĐỊNH LƯỢNG")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<p class="metric-label">Tổng điểm Nomogram NNSI</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value" style="color:{color};">{nnsi_score} / 91</p>', unsafe_allow_html=True)
        st.markdown(f"**Kết luận Lâm sàng:** <span style='color:{color}; font-weight:bold;'>{risk_cat}</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f'<p class="metric-label">Xác suất Dự đoán Bất lợi Giấc ngủ (AI Model)</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value" style="color:#334155;">{prob_risk:.1f}%</p>', unsafe_allow_html=True)
        st.markdown("**Thuật toán Cốt lõi:** `XGBoost Classifier (n_estimators=500)`")
        
    st.divider()
    
    # Render Báo cáo Diễn giải (SHAP)
    st.markdown("### 🧠 PHÂN TÍCH KHẢ NĂNG DIỄN GIẢI (Explainable AI - SHAP Values)")
    st.caption("Biểu đồ lực (Force Plot) dưới đây bóc tách mức độ đóng góp của từng vi chất vào sự thay đổi xác suất nguy cơ. Các véc-tơ màu đỏ đẩy nguy cơ lên cao (chống lại giấc ngủ sinh lý), trong khi các véc-tơ màu xanh kéo nguy cơ xuống (bảo vệ giấc ngủ).")
    
    shap_vals_patient = explainer.shap_values(df_model_ready)
    st_shap(shap.force_plot(explainer.expected_value, shap_vals_patient[0], df_model_ready.iloc[0], matplotlib=False), height=150)
