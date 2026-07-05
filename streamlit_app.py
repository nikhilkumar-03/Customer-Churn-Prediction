import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'churn_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
model_columns = joblib.load(os.path.join(BASE_DIR, 'model_columns.pkl'))

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn.")

# Blank/neutral starting values
empty_customer = {
    'gender': 'Female', 'SeniorCitizen': 'No', 'Partner': 'No', 'Dependents': 'No',
    'tenure': 0, 'PhoneService': 'No', 'MultipleLines': 'No',
    'InternetService': 'No', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
    'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'No',
    'PaymentMethod': 'Mailed check', 'MonthlyCharges': 0.0, 'TotalCharges': 0.0
}

# Sample data, only loaded on button click
sample_customer = {
    'gender': 'Female', 'SeniorCitizen': 'No', 'Partner': 'Yes', 'Dependents': 'No',
    'tenure': 5, 'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': 85.5, 'TotalCharges': 425.0
}

if 'form_data' not in st.session_state:
    st.session_state.form_data = empty_customer.copy()

if st.button(" Load Sample Customer"):
    st.session_state.form_data = sample_customer.copy()
    st.rerun()

data = st.session_state.form_data

st.subheader("Customer Profile")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ['Female', 'Male'], index=['Female', 'Male'].index(data['gender']))
    partner = st.selectbox("Partner", ['Yes', 'No'], index=['Yes', 'No'].index(data['Partner']))
with col2:
    senior = st.selectbox("Senior Citizen", ['No', 'Yes'], index=['No', 'Yes'].index(data['SeniorCitizen']))
    dependents = st.selectbox("Dependents", ['Yes', 'No'], index=['Yes', 'No'].index(data['Dependents']))

st.subheader("Account Details")
col1, col2 = st.columns(2)
with col1:
    tenure = st.number_input("Tenure (months)", 0, 100, data['tenure'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'],
                             index=['Month-to-month', 'One year', 'Two year'].index(data['Contract']))
with col2:
    paperless = st.selectbox("Paperless Billing", ['Yes', 'No'], index=['Yes', 'No'].index(data['PaperlessBilling']))
    payment_method = st.selectbox("Payment Method",
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        index=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'].index(data['PaymentMethod']))

st.subheader(" Services")
col1, col2 = st.columns(2)
with col1:
    phone_service = st.selectbox("Phone Service", ['Yes', 'No'], index=['Yes', 'No'].index(data['PhoneService']))
    multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No'], index=['Yes', 'No'].index(data['MultipleLines']))
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'],
                                     index=['DSL', 'Fiber optic', 'No'].index(data['InternetService']))
    online_security = st.selectbox("Online Security", ['Yes', 'No'], index=['Yes', 'No'].index(data['OnlineSecurity']))
with col2:
    online_backup = st.selectbox("Online Backup", ['Yes', 'No'], index=['Yes', 'No'].index(data['OnlineBackup']))
    device_protection = st.selectbox("Device Protection", ['Yes', 'No'], index=['Yes', 'No'].index(data['DeviceProtection']))
    tech_support = st.selectbox("Tech Support", ['Yes', 'No'], index=['Yes', 'No'].index(data['TechSupport']))
    streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No'], index=['Yes', 'No'].index(data['StreamingTV']))

streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No'], index=['Yes', 'No'].index(data['StreamingMovies']))

st.subheader("Billing")
col1, col2 = st.columns(2)
with col1:
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, float(data['MonthlyCharges']))
with col2:
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(data['TotalCharges']))

if st.button("🔮 Predict Churn", type="primary"):
    input_dict = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_dict])

    binary_map = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}
    for col in input_df.columns:
        val = input_df[col].iloc[0]
        if isinstance(val, str) and val in binary_map:
            input_df[col] = input_df[col].map(binary_map)

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    numeric_cols = list(scaler.feature_names_in_)
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"**High Risk of Churn** — Probability: {probability*100:.1f}%")
    else:
        st.success(f"**Likely to Stay** — Churn Probability: {probability*100:.1f}%")

    st.progress(float(probability))