import streamlit as st
import pandas as pd
import pickle
import xgboost
import os
import shap
import streamlit.components.v1 as components

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="GD Adverse Perinatal Outcome Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .result-box { padding: 20px; border-radius: 4px; border: 1px solid #dee2e6; margin-bottom: 20px; background-color: #fcfcfc; }
    .result-title { color: #495057; font-size: 15px; font-weight: 600; text-transform: uppercase; }
    .result-value { font-size: 32px; font-weight: 700; color: #212529; margin: 10px 0; }
    .status-high { border-left: 6px solid #dc3545; }
    .status-low { border-left: 6px solid #198754; }
    .status-mod { border-left: 6px solid #ffc107; }
    </style>
    """, unsafe_allow_html=True)

def st_shap(plot, height=None):
    # SHAP force plots require the Javascript library to be loaded
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height if height else 150)

st.markdown("### Clinical Prediction System for Adverse Perinatal Outcome")
st.markdown("GD Patient Risk Assessment | XGBoost-SHAP Integration")
st.markdown("---")

# ==========================================
# 2. Model Loading
# ==========================================
@st.cache_resource
def load_model():
    model_filename = 'gd_outcome_model.pkl' 
    if not os.path.exists(model_filename):
        st.error(f"Error: '{model_filename}' not found.")
        st.stop()
    with open(model_filename, 'rb') as file:
        return pickle.load(file)

model = load_model()

# ==========================================
# 3. Sidebar: Clinical Parameters
# ==========================================
with st.sidebar:
    st.markdown("#### Patient Clinical Profile")
    with st.form("clinical_input_form"):
        bmi = st.number_input("BMI (kg/m²)", 15.0, 50.0, 24.5, 0.1)
        fpg = st.number_input("Fasting Plasma Glucose (mmol/L)", 2.0, 20.0, 5.1, 0.1)
        tg = st.number_input("Triglycerides (TG) (mmol/L)", 0.1, 15.0, 1.7, 0.01)
        a1_b = st.number_input("ApoA1/ApoB Ratio (A1/B)", 0.1, 5.0, 1.2, 0.01)
        d_dimer = st.number_input("D-dimer (mg/L)", 0.0, 10.0, 0.5, 0.01)
        submitted = st.form_submit_button("Predict Outcome Risk")

# ==========================================
# 4. Main Interface
# ==========================================
if submitted:
    # UPDATED: Keys now match ['BMI', 'FPG', 'TG', 'A1.B', 'D-dimer']
    input_data = {
        'BMI': bmi,
        'FPG': fpg,
        'TG': tg,
        'A1.B': a1_b,
        'D-dimer': d_dimer
    }
    df_input = pd.DataFrame([input_data])

    try:
        prediction_probs = model.predict_proba(df_input)[0]
        risk_prob = float(prediction_probs[1])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if risk_prob > 0.6:
                status_class, status_text, advice = "status-high", "High Risk Profile", "Intensive monitoring recommended."
            elif risk_prob < 0.25:
                status_class, status_text, advice = "status-low", "Low Risk Profile", "Standard management suggested."
            else:
                status_class, status_text, advice = "status-mod", "Moderate Risk Profile", "Increased surveillance suggested."

            st.markdown(f"""
            <div class="result-box {status_class}">
                <div class="result-title">{status_text}</div>
                <div class="result-value">{risk_prob*100:.1f}%</div>
                <div class="result-desc">Probability of Adverse Perinatal Outcome</div>
                <hr style="margin: 15px 0; border: 0; border-top: 1px solid #eee;">
                <div class="result-desc"><strong>Guidance:</strong> {advice}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Risk Score", f"{risk_prob:.2f}")
            m2.metric("Relative Risk", "Elevated" if risk_prob > 0.4 else "Stable")
            
            with st.expander("Feature Input Log"):
                st.table(df_input.T.rename(columns={0: 'Value'}))

        st.markdown("#### Patient-Specific Contribution (SHAP Interpretation)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        
        # Link="logit" displays probabilities instead of log-odds
        st_shap(shap.force_plot(explainer.expected_value, shap_values[0], df_input.iloc[0], link="logit"), height=140)

    except Exception as e:
        st.error(f"Calculation Error: {str(e)}")
else:
    st.info("Enter clinical measurements in the sidebar and click 'Predict' to begin.")