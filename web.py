import streamlit as st
import pandas as pd
import pickle
import xgboost
import os

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="IVF/ICSI Miscarriage Risk Prediction",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact layout
st.markdown("""
    <style>
    /* Main background color */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Common Card Style (Compact) */
    .result-card {
        background-color: #ffffff;
        padding: 15px;       /* Reduced padding */
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-top: 10px;    /* Reduced margin */
        margin-bottom: 10px;
    }

    /* Color variations */
    .result-card-high-risk {
        border-left: 5px solid #FF5252;
    }
    .result-card-low-risk {
        border-left: 5px solid #4CAF50;
    }
    .result-card-moderate {
        border-left: 5px solid #FFC107;
    }

    /* Compact Typography inside cards */
    .compact-h3 {
        margin-top: 0 !important;
        margin-bottom: 5px !important;
        font-size: 1.1rem !important;
    }
    .compact-h1 {
        color: #333; 
        margin: 0 !important;
        font-size: 2.2rem !important;
        line-height: 1.2 !important;
    }
    .compact-p {
        color: #666; 
        margin: 0 !important; 
        font-size: 0.9rem !important;
    }
    .compact-hr {
        margin-top: 10px !important;
        margin-bottom: 10px !important;
    }
    
    /* Adjust Streamlit's default vertical spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚠️ IVF/ICSI Early Miscarriage Risk Prediction")
st.markdown("**XGBoost-based Clinical Decision Support System**")
st.markdown("---")

# ==========================================
# 2. Model Loading
# ==========================================
@st.cache_resource
def load_model():
    model_filename = 'xgb_model.pkl'
    
    if not os.path.exists(model_filename):
        alt_path = "2.训练集构建模型/xgb_model.pkl"
        if os.path.exists(alt_path):
            model_filename = alt_path
        else:
            st.error(f"❌ Critical Error: Model file `{model_filename}` not found. Please upload it to the GitHub repository.")
            st.stop()
        
    try:
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_model()

# ==========================================
# 3. Sidebar: Patient Data Entry
# ==========================================
with st.sidebar:
    st.header("📝 Patient Clinical Data")
    
    with st.form("input_form"):
        # 1. Female_age
        female_age = st.number_input(
            "1. Female Age (years)", 
            min_value=20.0, max_value=55.0, value=32.0, step=1.0
        )
        
        # 2. BMI
        bmi = st.number_input(
            "2. BMI (kg/m²)", 
            min_value=10.0, max_value=50.0, value=22.5, step=0.1,
            help="Body Mass Index"
        )
        
        # 3. PLT
        plt_val = st.number_input(
            "3. Platelet Count (10⁹/L)", 
            min_value=10.0, max_value=600.0, value=250.0, step=1.0,
            help="PLT"
        )
        
        # 4. FSH
        fsh = st.number_input(
            "4. Basal FSH (IU/L)", 
            min_value=0.0, max_value=100.0, value=7.5, step=0.1,
            help="Follicle-Stimulating Hormone"
        )
        
        # 5. TSH
        tsh = st.number_input(
            "5. TSH (mIU/L)", 
            min_value=0.0, max_value=50.0, value=2.0, step=0.01,
            help="Thyroid Stimulating Hormone"
        )
        
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Calculate Risk", use_container_width=True)

# ==========================================
# 4. Main Interface: Prediction Logic
# ==========================================
if submitted:
    # Construct DataFrame
    input_data = {
        'Female_age': female_age,
        'BMI': bmi,
        'PLT': plt_val,
        'FSH': fsh,
        'TSH': tsh
    }
    df_input = pd.DataFrame([input_data])

    # Compact Expander: Collapsed by default to save space
    with st.expander("📋 Click to view input data summary", expanded=False):
        st.dataframe(df_input, use_container_width=True)

    try:
        # Prediction
        prediction_probs = model.predict_proba(df_input)[0]
        risk_prob = float(prediction_probs[1]) 
        
        # Layout for results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📊 Analysis Result")
            
            # Progress Bar (Compact label)
            st.caption("Estimated Probability of Early Miscarriage")
            st.progress(risk_prob)
            
            # Logic for status
            if risk_prob > 0.5:
                card_style = "result-card result-card-high-risk"
                icon = "⚠️"
                status = "High Risk"
                advice = "High risk detected. Close monitoring recommended."
            elif risk_prob < 0.2:
                card_style = "result-card result-card-low-risk"
                icon = "✅"
                status = "Low Risk"
                advice = "Low risk detected. Routine care suggested."
                st.balloons()
            else:
                card_style = "result-card result-card-moderate"
                icon = "⚖️"
                status = "Moderate Risk"
                advice = "Intermediate risk. Clinical judgment required."

            # Compact HTML Card
            st.markdown(f"""
            <div class="{card_style}">
                <h3 class="compact-h3">{icon} Prediction: {status}</h3>
                <h1 class="compact-h1">{risk_prob*100:.2f}%</h1>
                <p class="compact-p">Probability of Early Miscarriage</p>
                <hr class="compact-hr">
                <p class="compact-p"><strong>💡 Suggestion:</strong> {advice}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📈 Metrics")
            st.metric(label="Miscarriage Prob.", value=f"{risk_prob:.2%}", delta_color="inverse")
            st.metric(label="Live Birth Prob.", value=f"{1-risk_prob:.2%}")

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("👈 Enter data in sidebar and click 'Calculate Risk'.")

# Compact footer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** For research use only. Not for medical diagnosis.")