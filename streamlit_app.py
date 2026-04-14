import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

# 1. Advanced Page Configuration
st.set_page_config(
    page_title="Cardiac Intelligence Portal",
    layout="wide",
    page_icon="🩺"
)

# 2. Professional Dark/Light Mode CSS
st.markdown("""
    <style>
    /* Main Container Styling */
    .main {
        background-color: transparent;
    }
    
    /* Custom Card Styling */
    .clinical-card {
        border-radius: 12px;
        padding: 20px;
        background-color: rgba(150, 150, 150, 0.1);
        border: 1px solid rgba(150, 150, 150, 0.2);
        margin-bottom: 20px;
    }
    
    /* Risk Badge Styling */
    .status-badge {
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    /* Metric Enhancement */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Asset Loader (with optimized pathing)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('src/heart_model_calibrated.pkl')
        scaler = joblib.load('src/scaler.pkl')
        ood_detector = joblib.load('src/ood_detector.pkl')
        explainer = joblib.load('src/shap_explainer.pkl')
        return model, scaler, ood_detector, explainer
    except Exception as e:
        st.error(f"⚠️ Critical System Error: Missing clinical assets ({e})")
        return None, None, None, None

model, scaler, ood_detector, explainer = load_assets()

if model is None: st.stop()

# 4. Clinical Input Mapping (Aliases for better UX)
FEATURE_MAP = {
    'age': 'Patient Age',
    'sex': 'Biological Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Serum Cholesterol',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG Results',
    'thalach': 'Max Heart Rate',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression',
    'slope': 'Peak Exercise Slope',
    'ca': 'Major Vessels (0-3)',
    'thal': 'Thalassemia Status'
}

# 5. Sidebar: Input Grouping
with st.sidebar:
    st.title("📋 Patient Data")
    st.markdown("---")
    
    with st.expander("Physical Demographics", expanded=True):
        age = st.slider("Age", 20, 90, 50)
        sex = st.selectbox("Sex", (0, 1), format_func=lambda x: "Male" if x==1 else "Female")
        thal = st.selectbox("Thalassemia", (1, 2, 3), format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])

    with st.expander("Clinical Vitals", expanded=True):
        trestbps = st.number_input("BP (mm Hg)", 90, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", (0, 1), format_func=lambda x: "Yes" if x==1 else "No")
        
    with st.expander("Cardiac Diagnostics", expanded=False):
        cp = st.selectbox("Chest Pain Type", (0, 1, 2, 3))
        thalach = st.slider("Max Heart Rate", 60, 220, 150)
        restecg = st.selectbox("Resting ECG", (0, 1, 2))
        exang = st.selectbox("Exercise Angina", (0, 1), format_func=lambda x: "Yes" if x==1 else "No")
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
        slope = st.selectbox("ST Slope", (0, 1, 2))
        ca = st.selectbox("Major Vessels Colored", (0, 1, 2, 3))

    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
    input_df = pd.DataFrame(data, index=[0])

# 6. Processing Engine
input_scaled = scaler.transform(input_df)
# Map to Clinical Names for XAI
input_scaled_df = pd.DataFrame(input_scaled, columns=[FEATURE_MAP[c] for c in input_df.columns])

prob = model.predict_proba(input_scaled)[0][1]
is_ood = ood_detector.predict(input_scaled)[0] == -1

# 7. Main Dashboard Design
st.title("🩺 Cardiac Risk Assessment Intelligence")
st.write("---")

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("Diagnostic Results")
    
    # Reliability Badge
    if is_ood:
        st.markdown('<span class="status-badge" style="background-color: #F59E0B; color: black;">⚠️ LOW DATA RELIABILITY</span>', unsafe_allow_html=True)
        st.warning("**Data Divergence Detected:** This profile is statistically unique. Interpret with caution.")
    else:
        st.markdown('<span class="status-badge" style="background-color: #10B981; color: white;">✅ HIGH DATA RELIABILITY</span>', unsafe_allow_html=True)

    # Risk Metric
    risk_label = "CRITICAL" if prob > 0.7 else "HIGH" if prob > 0.5 else "LOW"
    delta_color = "inverse" if prob > 0.5 else "normal"
    st.metric(label="Calculated Cardiac Risk", value=f"{prob*100:.1f}%", delta=risk_label, delta_color=delta_color)
    
    st.markdown("---")
    st.markdown("### 🏥 Clinical Action Plan")
    if prob > 0.7:
        st.error("**Urgent Referral:** Immediate cardiology consultation and advanced imaging recommended.")
    elif prob > 0.3:
        st.warning("**Watchful Waiting:** Secondary testing (Stress Echo/CCTA) advised based on risk drivers.")
    else:
        st.success("**Preventative Care:** Maintain routine monitoring and lifestyle management.")
        
with col_right:
    st.subheader("Explainable AI (XAI)")
    st.write("Understanding why this patient received this cardiac risk score:")

    # 🌙 Detect theme (simple heuristic)
    is_dark = st.get_option("theme.base") == "dark"

    # 🎨 Adaptive colors
    if is_dark:
        red_bg = "#7f1d1d"
        green_bg = "#14532d"
        text_color = "white"
        plot_bg = "#0e1117"
    else:
        red_bg = "#fee2e2"
        green_bg = "#dcfce7"
        text_color = "black"
        plot_bg = "white"

    try:
        # Generate SHAP values
        shap_values = explainer(input_scaled_df)

        # Handle SHAP formats
        if hasattr(shap_values, "values"):
            if len(shap_values.values.shape) == 3:
                values_to_plot = shap_values.values[0, :, 1]
                base_value = shap_values.base_values[0][1]
            else:
                values_to_plot = shap_values.values[0]
                base_value = shap_values.base_values[0]
        else:
            shap_values_raw = explainer.shap_values(input_scaled)
            if isinstance(shap_values_raw, list):
                values_to_plot = shap_values_raw[1][0]
                base_value = explainer.expected_value[1]
            else:
                values_to_plot = shap_values_raw[0]
                base_value = explainer.expected_value

        feature_names = input_scaled_df.columns.tolist()

        # =========================
        # 🔝 TOP RISK DRIVERS
        # =========================
        st.markdown("### 🚨 Top Contributing Factors")

        top_features = sorted(
            zip(feature_names, values_to_plot),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        for name, val in top_features:
            if val > 0:
                st.markdown(
                    f"<div style='padding:10px; border-radius:10px; margin-bottom:6px; background-color:{red_bg}; color:{text_color};'>"
                    f"🔴 <b>{name}</b> — Increases Risk (+{val:.3f})"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='padding:10px; border-radius:10px; margin-bottom:6px; background-color:{green_bg}; color:{text_color};'>"
                    f"🟢 <b>{name}</b> — Reduces Risk ({val:.3f})"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # =========================
        # 🧠 NATURAL LANGUAGE SUMMARY
        # =========================
        st.markdown("### 🧠 Clinical Interpretation")

        explanations = []
        for name, val in top_features[:3]:
            if val > 0:
                explanations.append(f"{name} is significantly increasing cardiac risk")
            else:
                explanations.append(f"{name} is helping reduce cardiac risk")

        st.info(" • " + "\n • ".join(explanations))

        # =========================
        # 📊 SHAP VISUALIZATION (DARK MODE FIX)
        # =========================
        st.markdown("### 📊 Feature Impact Visualization")

        plt.style.use("dark_background" if is_dark else "default")

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(plot_bg)

        shap.plots.waterfall(
            shap.Explanation(
                values=values_to_plot,
                base_values=base_value,
                data=input_scaled_df.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )

        st.pyplot(fig)

    except Exception as e:
        st.error("⚠️ Unable to generate advanced XAI insights.")
        st.write(str(e))

# 8. Export & Metadata
st.write("---")
col_footer1, col_footer2 = st.columns(2)
with col_footer1:
    report_text = f"Patient Cardiac Risk Report\nScore: {prob*100:.1f}%\nReliability: {'OOD' if is_ood else 'Normal'}"
    st.download_button("📥 Download Clinical Report", data=report_text, file_name="cardiac_analysis.txt")

with col_footer2:
    with st.expander("System Transparency"):
        st.caption("Algorithm: Calibrated Random Forest Classifier")
        st.caption("XAI: SHAP (Kernel/Tree Explainer)")
        st.caption("Training Source: UCI Cleveland Heart Disease Dataset")