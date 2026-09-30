# churn_app.py

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("randomforest_churn_model.pkl")
CHURN_LABEL = "Attrited Customer"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()


st.title("💳 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn based on their profile & activity.")


with st.form("input_form"):
    st.subheader("Enter Customer Details")

    gender = st.selectbox("Gender", ["M", "F"])
    education = st.selectbox("Education Level", [
        "Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate", "Unknown"
    ])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unknown"])
    income = st.selectbox("Income Category", [
        "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"
    ])
    card_category = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])

    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=40)
    dependent_count = st.slider("Dependent Count", 0, 10, 2)
    total_relationship_count = st.slider("Total Relationship Count", 1, 10, 3)
    months_inactive = st.slider("Months Inactive (12 months)", 0, 12, 2)
    contacts_count = st.slider("Contacts Count (12 months)", 0, 10, 2)
    credit_limit = st.number_input("Credit Limit", min_value=500, max_value=100000, value=5000)
    revolving_bal = st.number_input("Total Revolving Balance", min_value=0, max_value=50000, value=1500)
    amt_chng = st.number_input("Total Amount Change (Q4/Q1)", min_value=0.0, max_value=5.0, value=1.2)
    trans_ct = st.number_input("Total Transaction Count", min_value=0, max_value=200, value=45)
    ct_chng = st.number_input("Total Count Change (Q4/Q1)", min_value=0.0, max_value=5.0, value=1.1)
    utilization = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.3)

    submitted = st.form_submit_button("Predict Churn")


if submitted:
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Education_Level": education,
        "Marital_Status": marital,
        "Income_Category": income,
        "Card_Category": card_category,
        "Customer_Age": customer_age,
        "Dependent_count": dependent_count,
        "Total_Relationship_Count": total_relationship_count,
        "Months_Inactive_12_mon": months_inactive,
        "Contacts_Count_12_mon": contacts_count,
        "Credit_Limit": credit_limit,
        "Total_Revolving_Bal": revolving_bal,
        "Total_Amt_Chng_Q4_Q1": amt_chng,
        "Total_Trans_Ct": trans_ct,
        "Total_Ct_Chng_Q4_Q1": ct_chng,
        "Avg_Utilization_Ratio": utilization
    }])

    class_labels = list(model.classes_)
    churn_index = class_labels.index(CHURN_LABEL)
    churn_proba = model.predict_proba(input_data)[0, churn_index]
    churn_class = model.predict(input_data)[0]
    is_churn = churn_class == CHURN_LABEL

    st.subheader("📊 Prediction Result")
    st.write(f"**Churn Probability:** {churn_proba:.2%}")
    st.write(f"**Predicted Class:** {'Churn' if is_churn else 'Not Churn'}")

    if is_churn:
        st.error("⚠️ This customer is at **HIGH risk of churn**.")
    else:
        st.success("✅ This customer is **not likely to churn**.")
