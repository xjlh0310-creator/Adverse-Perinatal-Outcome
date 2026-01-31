import streamlit as st
import pandas as pd
import pickle
import xgboost
import os
import shap
import streamlit.components.v1 as components

# ==========================================
# 1. Page Configuration (Medical Professional)
# ==========================================
st.set_page_config(
    page_title="GD Adverse Perinatal Outcome Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clinical, high-contrast UI
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    .main {
        background-color: #ffffff;
    }
    .result-box {
        padding: 20px;
        border-radius: 4px;
        border: 1px solid #dee2e6;
        margin-bottom: 20px;
        background-color: #fcfcfc;
    }
    .result-title {
        color: #495057;
        font-size: 15px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .result-value {
        font-size: 32px;
        font-weight: 700;
        color: #212529;
        margin: 10px 0;
    }
    .result-desc {
        color: #6c757d;
        font-size: 14px;
    }
    .status-high { border-left: 6px solid #dc3545; }
    .status-low { border-left: 6px solid #198754; }
    .status-mod { border-left: 6px solid #ffc107; }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def st_shap(plot, height=None):
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
    # Update this filename to your actual model file path
    model_filename = 'gd_outcome_model.pkl' 
    if not os.path.exists(model_filename):
        st.error(f"Critical Error: Model file '{model_filename}' not found.")
        st.stop()
    try:
        with open(model_filename, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        st.stop()

model = load_model()

# ==========================================
# 3. Sidebar: Clinical Parameters
# ==========================================
with st.sidebar:
    st.markdown("#### Patient Clinical Profile")
    st.info("Input maternal data from gestational diagnosis (GD) records.")
    
    with st.form("clinical_input_form"):
        bmi = st.number_input("BMI (kg/m²)", 15.0, 50.0, 24.5, 0.1)
        fpg = st.number_input("Fasting Plasma Glucose (mmol/L)", 2.0, 20.0, 5.1, 0.1)
        tg = st.number_input("Triglycerides (TG) (mmol/L)", 0.1, 15.0, 1.7, 0.01)
        a1_b = st.number_input("ApoA1/ApoB Ratio (A1/B)", 0.1, 5.0, 1.2, 0.01)
        d_dimer = st.number_input("D-dimer (mg/L FEU)", 0.0, 10.0, 0.5, 0.01)
        
        st.markdown("---")
        submitted = st.form_submit_button("Predict Outcome Risk")

# ==========================================
# 4. Main Interface & Logic
# ==========================================
if submitted:
    # Feature names must match exactly what the XGBoost model was trained on
    input_data = {
        'BMI': bmi,
        'FPG': fpg,
        'TG': tg,
        'A1/B': a1_b,
        'D_dimer': d_dimer
    }
    df_input = pd.DataFrame([input_data])

    try:
        # Prediction
        prediction_probs = model.predict_proba(df_input)[0]
        risk_prob = float(prediction_probs[1])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Classification Logic
            if risk_prob > 0.6:
                status_class, status_text = "status-high", "High Risk Profile"
                advice = "Intensive perinatal monitoring and glycemic control adjustment recommended."
            elif risk_prob < 0.25:
                status_class, status_text = "status-low", "Low Risk Profile"
                advice = "Standard gestational diabetes management and routine follow-up."
            else:
                status_class, status_text = "status-mod", "Moderate Risk Profile"
                advice = "Increased surveillance of fetal growth and maternal metabolic markers."

            st.markdown(f"""
            <div class="result-box {status_class}">
                <div class="result-title">{status_text}</div>
                <div class="result-value">{risk_prob*100:.1f}%</div>
                <div class="result-desc">Calculated Probability of Adverse Perinatal Outcome</div>
                <hr style="margin: 15px 0; border: 0; border-top: 1px solid #eee;">
                <div class="result-desc"><strong>Clinical Guidance:</strong> {advice}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Risk Score", f"{risk_prob:.2f}")
            m2.metric("Relative Risk", "Elevated" if risk_prob > 0.4 else "Stable")
            
            with st.expander("Feature Input Log"):
                st.table(df_input.T.rename(columns={0: 'Value'}))

        # 5. Interpretability Section
        st.markdown("#### Patient-Specific Contribution (SHAP Interpretation)")
        st.caption("The visualization below explains the risk drivers for THIS specific patient. Red features increase risk; Blue features decrease it.")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        
        st_shap(shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            df_input.iloc[0], 
            link="logit"
        ), height=140)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.info("Ensure the model features (BMI, FPG, TG, A1/B, D-dimer) match the training dataset column names.")

else:
    st.info("Please provide the patient's clinical measurements in the sidebar to generate a risk assessment.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999; font-size: 12px;'>"
    "Clinical Decision Support Tool - Research Prototype Only. Not for final diagnostic use."
    "</div>", 
    unsafe_allow_html=True
)