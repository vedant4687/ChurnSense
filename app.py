import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model and scaler
model = joblib.load('models/churn_model_final.pkl')
scaler = joblib.load('models/scaler_final.pkl')

with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to churn based on their profile.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 35)
    tenure = st.slider("Tenure (months)", 1, 60, 24)
    usage_frequency = st.slider("Usage Frequency", 1, 30, 15)
    support_calls = st.slider("Support Calls", 0, 10, 2)
    payment_delay = st.slider("Payment Delay (days)", 0, 30, 5)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract_length = st.selectbox("Contract Length", ["Annual", "Monthly", "Quarterly"])
    total_spend = st.number_input("Total Spend ($)", 100, 1000, 500)
    last_interaction = st.slider("Days Since Last Interaction", 1, 30, 10)

st.divider()

if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    # Encode inputs
    gender_enc = 0 if gender == "Female" else 1
    sub_enc = {"Basic": 0, "Premium": 1, "Standard": 2}[subscription_type]
    contract_enc = {"Annual": 0, "Monthly": 1, "Quarterly": 2}[contract_length]

    # Create input dataframe
    input_data = pd.DataFrame([[
        age, gender_enc, tenure, usage_frequency,
        support_calls, payment_delay, sub_enc,
        contract_enc, total_spend, last_interaction
    ]], columns=feature_names)

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    # Display result
    st.divider()
    st.markdown(f"### Churn Probability: {prob*100:.1f}%")
    st.progress(float(prob))

    if prob >= 0.7:
        st.error("⚠️ HIGH CHURN RISK")
        st.markdown("**Recommended Actions:**")
        st.markdown("- Offer a loyalty discount or upgrade")
        st.markdown("- Assign a dedicated customer success manager")
        st.markdown("- Follow up on unresolved support issues")
    elif prob >= 0.4:
        st.warning("⚡ MEDIUM CHURN RISK")
        st.markdown("**Recommended Actions:**")
        st.markdown("- Send a satisfaction survey")
        st.markdown("- Offer a small incentive to stay")
    else:
        st.success("✅ LOW CHURN RISK")
        st.markdown("**Customer is likely to stay. Consider:**")
        st.markdown("- Upsell to a higher subscription tier")
        st.markdown("- Enroll in loyalty rewards program")

    # Risk gauge
    st.divider()
    st.markdown(f"**Churn Probability: {prob*100:.1f}%**")
    st.progress(float(prob))