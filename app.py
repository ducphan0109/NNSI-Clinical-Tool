import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components

st.sidebar.markdown("---")
st.sidebar.info(
    "👨‍🔬 **Tác giả:** Phan Minh Đức\n\n"
    "🏫 **Đơn vị:** Bộ môn Dinh dưỡng, Khoa Y - Trường Đại học Quốc tế Hồng Bàng (HIU)\n\n"
    "🔬 **Định hướng:** Dinh dưỡng Thần kinh (Nutritional Neuroscience)"
)
st.sidebar.markdown("---")
st.info("💡 **Chỉ số NNSI (Nutritional Neuroscience Sleep Index):** Là hệ thống điểm lâm sàng được trích xuất từ thuật toán XGBoost và SHAP, giúp định lượng nguy cơ rối loạn giấc ngủ dựa trên sự thiếu hụt hoặc dư thừa của các vi chất thần kinh.")

# 1. CẤU HÌNH GIAO DIỆN & LOAD TÀI NGUYÊN
st.set_page_config(page_title="NNSI Clinical Tool", page_icon="🧠", layout="wide")

@st.cache_resource
def load_resources():
    model = joblib.load("xgb_model.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    training_cols = joblib.load("training_columns.pkl")
    return model, explainer, training_cols

try:
    model, explainer, training_cols = load_resources()
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy file mô hình. Vui lòng kiểm tra lại.")
    st.stop()

# 2. HỆ THỐNG ĐIỂM NNSI
NNSI_RULES = {
'DR1TCAFF': {'cutoff': 317.00, 'dir': '>=', 'points': 37},  # Caffeine: Dư thừa (>=) gây kích thích thần kinh trung ương, cản trở giấc ngủ sóng chậm.
    'DR1TVB6': {'cutoff': 1.18, 'dir': '<', 'points': 17},      # Vitamin B6: Thiếu (<) làm giảm quá trình chuyển hoá Tryptophan thành Serotonin và Melatonin.
    'DR1TZINC': {'cutoff': 10.82, 'dir': '<', 'points': 8},     # Kẽm: Thiếu (<) suy giảm chất lượng giấc ngủ do Kẽm là đồng yếu tố tổng hợp Melatonin.
    'DR1TMAGN': {'cutoff': 290.97, 'dir': '<', 'points': 7},    # Magnesium: Thiếu (<) làm giảm hoạt động của thụ thể ức chế GABA, gây khó thư giãn.
    'DR1TFOLA': {'cutoff': 339.50, 'dir': '<', 'points': 7},    # Folate: Thiếu (<) ảnh hưởng hệ thần kinh và quá trình tổng hợp monoamine.
    'Carb_Pro_Ratio': {'cutoff': 3.67, 'dir': '>=', 'points': 5},# Tỷ lệ Carb/Đạm: Quá cao (>=) phản ánh chế độ ăn siêu chế biến, dư đường/thiếu đạm gây biến thiên đường huyết về đêm.
    'DR1TPROT': {'cutoff': 94.31, 'dir': '<', 'points': 4},     # Protein: Thiếu (<) dẫn đến hụt nguồn cung cấp axit amin thiết yếu (Tryptophan).
    'DR1TIRON': {'cutoff': 15.69, 'dir': '<', 'points': 3},     # Sắt: Thiếu (<) là cơ chế bệnh sinh trực tiếp gây suy giảm Dopamine, dẫn đến Hội chứng chân không yên (Restless Legs Syndrome) phá vỡ giấc ngủ.
    'DR1TCARB': {'cutoff': 280.27, 'dir': '<', 'points': 2},    # Carbohydrate: Thiếu (<) quá mức làm giảm lượng Insulin, cản trở Tryptophan vượt qua hàng rào máu não (BBB).
    'DR1TVB12': {'cutoff': 1.00, 'dir': '<', 'points': 1}       # Vitamin B12: Thiếu (<) liên quan đến rối loạn nhịp sinh học và thoái hoá myelin.
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
                
    if score <= 30: return score, "Nguy cơ Thấp", "green"
    elif score <= 60: return score, "Nguy cơ Trung bình", "orange"
    else: return score, "Nguy cơ Cao", "red"

# 3. GIAO DIỆN (SIDEBAR INPUT)
st.sidebar.header("📋 Hồ sơ Bệnh nhân")
with st.sidebar.form("patient_form"):
    st.subheader("Nhân khẩu & Tâm lý")
    age = st.number_input("Tuổi", min_value=18, max_value=80, value=30)
    gender = st.selectbox("Giới tính", options=[1, 2], format_func=lambda x: "Nam" if x==1 else "Nữ")
    race = st.selectbox("Chủng tộc", options=[1, 2, 3, 4, 6], index=2)
    bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=22.5)
    pir = st.number_input("Thu nhập PIR", min_value=0.0, max_value=5.0, value=2.0)
    phq9 = st.slider("Điểm Trầm cảm", min_value=0, max_value=27, value=0)
    
    st.subheader("Dinh dưỡng Thần kinh")
    caff = st.number_input("Caffeine (mg)", min_value=0.0, value=150.0)
    iron = st.number_input("Sắt (mg)", min_value=0.0, value=12.0)
    magn = st.number_input("Magnesium (mg)", min_value=0.0, value=250.0)
    zinc = st.number_input("Kẽm (mg)", min_value=0.0, value=10.0)
    vb6 = st.number_input("Vitamin B6 (mg)", min_value=0.0, value=1.5)
    vb12 = st.number_input("Vitamin B12 (mcg)", min_value=0.0, value=2.4)
    fola = st.number_input("Folate (mcg)", min_value=0.0, value=400.0)
    prot = st.number_input("Protein (g)", min_value=0.0, value=70.0)
    carb = st.number_input("Carbohydrate (g)", min_value=0.0, value=250.0)
    submitted = st.form_submit_button("Tính toán NNSI & Dự đoán")

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# 4. KẾT QUẢ
st.title("Phân tích Dinh dưỡng Thần kinh & Giấc ngủ")
if submitted:
    input_data = {'RIDAGEYR': age, 'RIAGENDR': gender, 'RIDRETH3': race, 'BMXBMI': bmi, 'INDFMPIR': pir, 'PHQ9_Score': phq9, 'DR1TCAFF': caff, 'DR1TIRON': iron, 'DR1TMAGN': magn, 'DR1TZINC': zinc, 'DR1TVB6': vb6, 'DR1TVB12': vb12, 'DR1TFOLA': fola, 'DR1TPROT': prot, 'DR1TCARB': carb}
    df_patient = pd.DataFrame([input_data])
    df_patient['Carb_Pro_Ratio'] = df_patient['DR1TCARB'] / df_patient['DR1TPROT']
    
    # Ép kiểu category trước khi One-Hot Encoding
    df_patient['RIAGENDR'] = pd.Categorical(df_patient['RIAGENDR'], categories=[1, 2])
    df_patient['RIDRETH3'] = pd.Categorical(df_patient['RIDRETH3'], categories=[1, 2, 3, 4, 6])
    df_patient = pd.get_dummies(df_patient, columns=['RIAGENDR', 'RIDRETH3'], drop_first=True, dtype=int)
    
    df_model_ready = df_patient.reindex(columns=training_cols, fill_value=0)
    prob_risk = model.predict_proba(df_model_ready)[0][1] * 100
    nnsi_score, risk_cat, color = calculate_nnsi(df_model_ready)
    
    col1, col2 = st.columns(2)
    with col1: 
        st.markdown(f"### 📊 Tổng điểm NNSI: **{nnsi_score} / 91**")
        st.markdown(f"**Phân loại:** <span style='color:{color}; font-size:18px; font-weight:bold;'>{risk_cat}</span>", unsafe_allow_html=True)
    with col2: 
        st.markdown(f"### 🤖 Xác suất Rối loạn giấc ngủ: **{prob_risk:.1f}%**")
        st.markdown("**Mô hình:** XGBoost Classifier")
    st.divider()        
    
    st.subheader("🔍 Khai phá Hộp đen (SHAP Analysis)")
    shap_vals_patient = explainer.shap_values(df_model_ready)
    st_shap(shap.force_plot(explainer.expected_value, shap_vals_patient[0], df_model_ready.iloc[0], matplotlib=False), height=150)
