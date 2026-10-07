import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & TỪ ĐIỂN SONG NGỮ
# ==========================================
st.set_page_config(page_title="NNSI Clinical Tool", page_icon="⚕️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,700;1,300&family=Inter:wght@400;600&display=swap');
    
    .academic-title {font-family: 'Merriweather', serif; font-size: 34px !important; font-weight: 700; color: #0f172a; margin-bottom: 5px; line-height: 1.3;}
    .academic-subtitle {font-family: 'Inter', sans-serif; font-size: 16px !important; color: #475569; margin-top: 0px; margin-bottom: 25px; font-weight: 400;}
    .abstract-box {font-family: 'Inter', sans-serif; background-color: #f8fafc; padding: 20px 25px; border-left: 5px solid #3b82f6; margin-bottom: 30px; font-size: 14px; color: #334155; line-height: 1.6;}
    .sidebar-info-box {background-color: #f1f5f9; padding: 15px; border-radius: 5px; font-size: 13px; font-family: 'Inter', sans-serif; color: #334155; margin-bottom: 15px; border: 1px solid #e2e8f0;}
    .metric-title {font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #64748b;}
    .metric-value {font-family: 'Inter', sans-serif; font-size: 38px; font-weight: 700; margin-top: -10px; margin-bottom: 0px;}
    .conclusion-text {font-family: 'Inter', sans-serif; font-size: 18px; font-weight: 600;}
    .divider {margin-top: 40px; margin-bottom: 40px; border-top: 1px solid #e2e8f0;}
    div[data-testid="stForm"] {border: none; padding: 0;}
    </style>
""", unsafe_allow_html=True)

LANG = {
    "EN": {
        "title": "Neuro-Nutritional Sleep Index (NNSI) Calculator",
        "subtitle": "An explainable AI-driven clinical decision support system using survey-weighted XGBoost",
        "abstract": "<b>Abstract:</b> The NNSI is a clinical nomogram derived from a machine learning framework trained on NHANES data. It quantifies the probability of an Adverse Sleep Phenotype based on neural micronutrient status and demographic controls. This tool is designed for research and adjunctive clinical decision-making.",
        "sidebar_info": "<b>Sole Author:</b> Phan Minh Duc<br><br><b>Affiliation:</b> Dept. of Nutrition, Faculty of Medicine, Hong Bang International University (HIU)<br><br><b>Domain:</b> Nutritional Neuroscience",
        "input_header": "📋 PATIENT PROFILING",
        "demo_header": "I. Demographics & Mental Health",
        "nutri_header": "II. Neural Micronutrients (24h Recall)",
        "age": "Age (Years)", "gender": "Gender", "male": "Male", "female": "Female",
        "race": "Race/Ethnicity", "bmi": "Body Mass Index (BMI)", 
        "pir": "Poverty Income Ratio (PIR)", 
        "pir_help": "Ratio of family income to the poverty threshold. <1.0 means below poverty level. 5.0 means income is 5x or more above poverty level.",
        "phq9": "PHQ-9 Depression Score",
        "phq9_help": "Total score from 0-27. Used as a confounding control in the AI model (does not directly add to NNSI Nomogram points).",
        "caff": "Caffeine (mg/d)", "iron": "Iron (mg/d)", "magn": "Magnesium (mg/d)", "zinc": "Zinc (mg/d)",
        "vb6": "Vitamin B6 (mg/d)", "vb12": "Vitamin B12 (mcg/d)", "fola": "Folate (mcg/d)", "prot": "Protein (g/d)", "carb": "Carbohydrate (g/d)",
        "btn": "Execute Analysis & Generate Report",
        "res_header": "📊 QUANTITATIVE CLINICAL REPORT",
        "score_label": "NOMOGRAM SCORE (NNSI)",
        "prob_label": "AI PREDICTED PROBABILITY (ADVERSE SLEEP)",
        "risk_low": "Low Risk Phenotype", "risk_mod": "Moderate Risk Phenotype", "risk_high": "High Risk Phenotype",
        "shap_header": "🧠 EXPLAINABLE AI (SHAP ANALYSIS)",
        "shap_desc": "The force plot below deconstructs the patient's specific risk profile. Red vectors indicate nutritional factors pushing toward sleep disturbance, while blue vectors represent protective factors.",
        "race_opts": {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other Race / Multi-Racial"}
    },
    "VI": {
        "title": "Hệ thống Đánh giá Lâm sàng: Chỉ số NNSI",
        "subtitle": "Công cụ hỗ trợ quyết định lâm sàng ứng dụng thuật toán phi tuyến tính Survey-Weighted XGBoost",
        "abstract": "<b>Tóm tắt:</b> Chỉ số NNSI (Nutritional Neuroscience Sleep Index) là một hệ thống điểm Nomogram được trích xuất thông qua kỹ thuật giải mã hộp đen SHAP. Công cụ định lượng hóa xác suất xuất hiện Kiểu hình Giấc ngủ Bất lợi dựa trên hồ sơ chuyển hóa vi chất và các biến số kiểm soát dịch tễ.",
        "sidebar_info": "<b>Tác giả độc lập:</b> Phan Minh Đức<br><br><b>Cơ quan:</b> Bộ môn Dinh dưỡng, Khoa Y - Trường Đại học Quốc tế Hồng Bàng (HIU)<br><br><b>Chuyên ngành:</b> Dinh dưỡng Thần kinh",
        "input_header": "📋 HỒ SƠ LÂM SÀNG",
        "demo_header": "I. Nhân khẩu & Tâm lý",
        "nutri_header": "II. Vi chất Thần kinh (24h Recall)",
        "age": "Tuổi (Năm)", "gender": "Giới tính", "male": "Nam giới", "female": "Nữ giới",
        "race": "Chủng tộc / Sắc tộc", "bmi": "Chỉ số Khối cơ thể (BMI)", 
        "pir": "Tỷ lệ Thu nhập/Nghèo đói (PIR)", 
        "pir_help": "Thang đo từ 0.0 - 5.0. Dưới 1.0 là sống dưới mức nghèo. 5.0 là thu nhập gấp 5 lần trở lên so với chuẩn nghèo.",
        "phq9": "Thang điểm Trầm cảm (PHQ-9)",
        "phq9_help": "Thang điểm từ 0-27. Đóng vai trò là biến kiểm soát nhiễu trong mô hình AI ngầm, không trực tiếp cộng vào Tổng điểm NNSI.",
        "caff": "Lượng Caffeine (mg/ngày)", "iron": "Sắt (mg/ngày)", "magn": "Magiê (mg/ngày)", "zinc": "Kẽm (mg/ngày)",
        "vb6": "Vitamin B6 (mg/ngày)", "vb12": "Vitamin B12 (mcg/ngày)", "fola": "Folate (mcg/ngày)", "prot": "Tổng Protein (g/ngày)", "carb": "Tổng Carbohydrate (g/ngày)",
        "btn": "Tiến hành Phân tích & Trích xuất Báo cáo",
        "res_header": "📊 BÁO CÁO KẾT QUẢ ĐỊNH LƯỢNG",
        "score_label": "TỔNG ĐIỂM NOMOGRAM NNSI",
        "prob_label": "XÁC SUẤT BẤT LỢI GIẤC NGỦ (AI MODEL)",
        "risk_low": "Kiểu hình Nguy cơ Thấp", "risk_mod": "Kiểu hình Nguy cơ Trung bình", "risk_high": "Kiểu hình Nguy cơ Cao",
        "shap_header": "🧠 KHAI PHÁ HỘP ĐEN (SHAP ANALYSIS)",
        "shap_desc": "Biểu đồ lực (Force Plot) dưới đây bóc tách mức độ đóng góp của từng vi chất vào sự thay đổi xác suất nguy cơ. Vectơ màu đỏ đẩy nguy cơ lên cao (chống lại giấc ngủ), vectơ màu xanh kéo nguy cơ xuống (bảo vệ giấc ngủ).",
        "race_opts": {1: "Người gốc Mexico", 2: "Người gốc Hispanic khác", 3: "Người Da trắng", 4: "Người Da đen", 6: "Người gốc Á", 7: "Đa chủng tộc / Khác"}
    }
}

lang_choice = st.sidebar.radio("🌐 Language / Ngôn ngữ", ["English", "Tiếng Việt"], horizontal=True)
lang = "EN" if lang_choice == "English" else "VI"
t = LANG[lang]

# ==========================================
# 2. TẢI TÀI NGUYÊN MÔ HÌNH
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
    st.error("Model resources not found / Không tìm thấy tệp trọng số mô hình.")
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
                
    if score <= 30: return score, t["risk_low"], "#16A34A"
    elif score <= 60: return score, t["risk_mod"], "#D97706"
    else: return score, t["risk_high"], "#DC2626"

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# ==========================================
# 4. GIAO DIỆN CHÍNH (MAIN APP)
# ==========================================
st.markdown(f'<div class="academic-title">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="academic-subtitle">{t["subtitle"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="abstract-box">{t["abstract"]}</div>', unsafe_allow_html=True)

st.sidebar.markdown(f'<div class="sidebar-info-box">{t["sidebar_info"]}</div>', unsafe_allow_html=True)

st.sidebar.markdown(f"### {t['input_header']}")
with st.sidebar.form("patient_form"):
    st.markdown(f"**{t['demo_header']}**")
    age = st.number_input(t["age"], min_value=18, max_value=80, value=30)
    gender = st.selectbox(t["gender"], options=[1, 2], format_func=lambda x: t["male"] if x==1 else t["female"])
    
    # Render biến Chủng tộc bằng chữ dễ hiểu
    race = st.selectbox(t["race"], options=[1, 2, 3, 4, 6, 7], index=4, format_func=lambda x: t["race_opts"][x])
    
    bmi = st.number_input(t["bmi"], min_value=10.0, max_value=80.0, value=22.5)
    
    # Bổ sung tooltip giải thích cho PIR và PHQ9
    pir = st.number_input(t["pir"], min_value=0.0, max_value=5.0, value=2.0, help=t["pir_help"])
    phq9 = st.slider(t["phq9"], min_value=0, max_value=27, value=0, help=t["phq9_help"])
    
    st.markdown(f"**{t['nutri_header']}**")
    caff = st.number_input(t["caff"], min_value=0.0, value=150.0)
    iron = st.number_input(t["iron"], min_value=0.0, value=12.0)
    magn = st.number_input(t["magn"], min_value=0.0, value=250.0)
    zinc = st.number_input(t["zinc"], min_value=0.0, value=20.0)
    vb6 = st.number_input(t["vb6"], min_value=0.0, value=1.5)
    vb12 = st.number_input(t["vb12"], min_value=0.0, value=2.4)
    fola = st.number_input(t["fola"], min_value=0.0, value=400.0)
    prot = st.number_input(t["prot"], min_value=0.0, value=70.0)
    carb = st.number_input(t["carb"], min_value=0.0, value=250.0)
    
    submitted = st.form_submit_button(t["btn"])

if submitted:
    input_data = {'RIDAGEYR': age, 'RIAGENDR': gender, 'RIDRETH3': race, 'BMXBMI': bmi, 'INDFMPIR': pir, 'PHQ9_Score': phq9, 'DR1TCAFF': caff, 'DR1TIRON': iron, 'DR1TMAGN': magn, 'DR1TZINC': zinc, 'DR1TVB6': vb6, 'DR1TVB12': vb12, 'DR1TFOLA': fola, 'DR1TPROT': prot, 'DR1TCARB': carb}
    df_patient = pd.DataFrame([input_data])
    df_patient['Carb_Pro_Ratio'] = df_patient['DR1TCARB'] / df_patient['DR1TPROT']
    
    df_patient['RIAGENDR'] = pd.Categorical(df_patient['RIAGENDR'], categories=[1, 2])
    df_patient['RIDRETH3'] = pd.Categorical(df_patient['RIDRETH3'], categories=[1, 2, 3, 4, 6])
    df_patient = pd.get_dummies(df_patient, columns=['RIAGENDR', 'RIDRETH3'], drop_first=True, dtype=int)
    
    df_model_ready = df_patient.reindex(columns=training_cols, fill_value=0)
    prob_risk = model.predict_proba(df_model_ready)[0][1] * 100
    nnsi_score, risk_cat, color = calculate_nnsi(df_model_ready)
    
    st.markdown(f"### {t['res_header']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-title">{t["score_label"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color:{color};">{nnsi_score} / 91</div>', unsafe_allow_html=True)
        st.markdown(f"<div class='conclusion-text' style='color:{color};'>{risk_cat}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-title">{t["prob_label"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color:#0f172a;">{prob_risk:.1f}%</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown(f"### {t['shap_header']}")
    st.caption(t["shap_desc"])
    shap_vals_patient = explainer.shap_values(df_model_ready)
    st_shap(shap.force_plot(explainer.expected_value, shap_vals_patient[0], df_model_ready.iloc[0], matplotlib=False), height=150)
