import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
try:
    import shap
except ImportError:
    shap = None

st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")
st.title(" Industrial Predictive Maintenance System")
st.markdown("**Machine Failure Prediction | Real-time Risk Assessment**")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('D://projects//update older projects//Project//Predective maintenance//notebooks//models//xgb_final_model.pkl')
    return model

model = load_model()
preprocessor = joblib.load('D://projects//update older projects//Project//Predective maintenance//notebooks//models//preprocessor.pkl')

# Sidebar inputs
st.sidebar.header("Sensor Readings")

air_temp = st.sidebar.slider("Air Temperature [K]", 295.0, 305.0, 300.0, 0.1)
process_temp = st.sidebar.slider("Process Temperature [K]", 305.0, 315.0, 310.0, 0.1)
rot_speed = st.sidebar.slider("Rotational Speed [rpm]", 1200, 2800, 1500, 10)
torque = st.sidebar.slider("Torque [Nm]", 10.0, 80.0, 40.0, 0.5)
tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 300, 100, 1)
product_type = st.sidebar.selectbox("Product Quality Type", ["L", "M", "H"])

# Feature Engineering
temp_diff = process_temp - air_temp
power = rot_speed * torque
wear_per_power = tool_wear / (power + 1)
log_tool_wear = np.log1p(tool_wear)

input_data = pd.DataFrame({
    'Type': [product_type],
    'Air temperature [K]': [air_temp],
    'Process temperature [K]': [process_temp],
    'Rotational speed [rpm]': [rot_speed],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear],
    'temp_diff': [temp_diff],
    'power': [power],
    'wear_per_power': [wear_per_power],
    'log_tool_wear': [log_tool_wear]
})

# Prediction
if st.sidebar.button(" Predict Failure Risk", type="primary"):
    prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Failure Probability", f"{prob:.1%}")
    with col2:
        risk = "HIGH" if prob > 0.6 else " MEDIUM" if prob > 0.25 else " LOW"
        st.metric("Risk Level", risk)
    with col3:
        health = 100 - (prob * 100)
        st.metric("Machine Health Score", f"{health:.1f}/100")
    
    # Recommendation
    if prob > 0.6:
        st.error("**Action Required**: Schedule immediate maintenance within 12 hours.")
        st.write("Recommended: Inspect spindle, check tooling, reduce load.")
    elif prob > 0.25:
        st.warning("**Monitor Closely**: Schedule inspection within 48 hours.")
    else:
        st.success("**Healthy**: Continue normal operation.")

    
    # SHAP Explanation
        # ==================== SHAP EXPLANATION ====================
    st.subheader("🔍 Why this prediction?")

    try:
        processed_input = preprocessor.transform(input_data)
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        
        # For XGBoost binary classification
        shap_values = explainer.shap_values(processed_input)
        
        # Waterfall Plot (Best for single prediction - clean & professional)
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=processed_input[0],
            feature_names=preprocessor.get_feature_names_out()
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.waterfall_plot(explanation, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("Red = Increases failure probability | Blue = Decreases failure probability")
        
    except Exception as e:
        st.error(f"SHAP Error: {str(e)}")
        st.info("**Top Drivers from Model**: High Power, High Tool Wear, and Torque are pushing risk up.")
   

# Batch Prediction Option
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload sensor data (CSV)", type=["csv"])
if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    # Add same feature engineering here...
    # Then show predictions table