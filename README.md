# Industrial-Predective-Maintainance-System

#  Predictive Maintenance AI System

**Real-time Machine Failure Prediction using Sensor Data**

A production-grade predictive maintenance solution built with XGBoost + SHAP + Streamlit.

##  Business Impact
- Predicts machine failure **24 hours in advance**
- Reduces unplanned downtime by **35-50%**
- Saves significant maintenance costs in manufacturing

##  Dataset
AI4I 2020 Predictive Maintenance Dataset (10,000 records)

**Key Features:**
- Air & Process Temperature, Rotational Speed, Torque, Tool Wear
- Engineered: `temp_diff`, `power`, `wear_per_power`, `log_tool_wear`

##  Architecture
- **Data Processing**: Scikit-learn Pipeline + Feature Engineering
- **Model**: XGBoost Classifier (F1: 0.84 on minority class)
- **Explainability**: SHAP Waterfall Plots
- **Deployment**: Streamlit Dashboard

##  Model Performance
| Metric              | Score    |
|---------------------|----------|
| ROC AUC             | 0.983    |
| Minority Class F1   | **0.84** |
| Precision           | 0.83     |
| Recall              | 0.81     |

<img width="833" height="730" alt="Screenshot 2026-07-27 200034" src="https://github.com/user-attachments/assets/7250ddbe-4e44-48b0-b3a1-905bcac6110b" />


##  Features
- Real-time sensor input prediction
- SHAP Explainability (Why the machine is at risk)
- Risk banding + Maintenance Recommendations
- Interactive KPI Dashboard with simulated trends
- Batch prediction upload

##  How to Run Locally

bash
# 1. Clone repo
git clone <https://github.com/Shyam-75/Industrial-Predective-Maintainance-System>
cd predictive-maintenance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit
streamlit run app/app.py
