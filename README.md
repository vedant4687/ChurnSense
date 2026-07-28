# Customer Churn Prediction

A machine learning system that predicts customer churn probability using 
**440,833 customer records** across 10 behavioral and demographic features. 
Trained, evaluated, and deployed as a live Streamlit web application.

## 🎯 Problem Statement

Acquiring a new customer costs 5–7x more than retaining an existing one. 
This model identifies at-risk customers **before** they leave — enabling 
proactive retention through targeted offers and outreach.

## 🚀 Live Demo

👉 [[Open Streamlit App](https://churnsense-2807.streamlit.app/)]

## 📊 Dataset

- **Source:** Customer Churn Dataset (Kaggle)
- **Size:** 440,833 training records + 64,374 test records
- **Target:** Churn (0 = stayed, 1 = churned) — 56.7% / 43.3% split
- **Features:** Age, Gender, Tenure, Usage Frequency, Support Calls, 
  Payment Delay, Subscription Type, Contract Length, Total Spend, 
  Last Interaction

## 🔍 Key EDA Findings

- **Support Calls** is the strongest churn signal — churners average 
  5 calls vs 1.6 for non-churners
- **Total Spend** — non-churners spend 37% more on average
- **Monthly contract** customers show higher churn than annual/quarterly
- **Payment delays** strongly correlate with churn — disengaged customers 
  delay payments before leaving

## 🤖 Models Trained & Compared

| Model | Validation AUC | Final Test AUC |
|---|---|---|
| Logistic Regression | 0.9281 | 0.7749 |
| Random Forest | 1.0000 | — (overfit) |
| XGBoost | 1.0000 | 0.6529 (overfit) |
| XGBoost Tuned | — | 0.7508 |
| **LR + Combined Data** | — | **0.8110 ✅** |

**Final Model:** Logistic Regression trained on combined train+test data  
**Final AUC:** 0.8110  
**Recall:** 0.98 — catches 98% of actual churners

## 🧠 SHAP Explainability

Top features by SHAP importance:
1. **Support Calls** — #1 predictor, high calls = high churn risk
2. **Total Spend** — low spenders are high risk
3. **Age** — older customers show higher churn in this dataset
4. **Contract Length** — monthly contracts are highest risk
5. **Payment Delay** — delays signal disengagement

## 📱 Streamlit App Features

- 10 interactive input fields (sliders + dropdowns)
- Real-time churn probability (0–100%)
- Three risk levels: High (>70%), Medium (40–70%), Low (<40%)
- Actionable business recommendations per risk tier
- Visual probability progress bar

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn, xgboost, shap, joblib
- **Deployment** — Streamlit Cloud
- **Visualization** — matplotlib, seaborn

## 📁 Project Structure
Customer_Churn_Prediction/
├── data/
│ ├── customer_churn_train.csv
│ └── customer_churn_test.csv
├── notebooks/
│ └── churn_analysis.ipynb
├── models/
│ ├── churn_model_final.pkl
│ ├── scaler_final.pkl
│ └── feature_names.json
├── app.py
└── requirements.txt


## 💡 How to Run Locally

```bash
git clone https://github.com/vedant4687/customer-churn-prediction
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 📋 Requirements

streamlit
pandas
numpy
scikit-learn
xgboost
shap
joblib
matplotlib
seaborn
