# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 08:46:04 2024

@author: khanu
"""

import streamlit as st
import pandas as pd
import joblib
import os

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def get_absolute_path(relative_path):
    return os.path.join(os.path.dirname(__file__), relative_path)

model = joblib.load(get_absolute_path("C:/Users/khanu/Documents/classexcelr/Project/email_marketing/xgb_model.pkl"))
scaler = joblib.load(get_absolute_path("C:/Users/khanu/Documents/classexcelr/Project/email_marketing/scaler.pkl"))

# Streamlit app title
st.set_page_config(
    page_title=" Email Marketing Campaign", layout="centered")
st.title("📈  Email Marketing Campaign")
st.write("""
### This application predicts whether a customer opened a previous email based on input features.
""")

# Sidebar for user inputs
st.header("Input Features")

# Collecting inputs for all features
customer_age = st.number_input("Customer Age", min_value=18, max_value=75, value=30)
emails_opened = st.number_input("Emails Opened", min_value=0, max_value=20, value=5)
emails_clicked = st.number_input("Emails Clicked", min_value=0, max_value=10, value=2)
purchase_history = st.number_input("Purchase History ($)", min_value=0.0, value=1000.0)
time_spent_on_website = st.number_input("Time Spent on Website (hrs)", min_value=0.0, max_value=12.0, value=5.0, step=0.1)
days_since_last_open = st.number_input("Days Since Last Open", min_value=0, max_value=100, value=30)
customer_engagement_score = st.number_input("Customer Engagement Score", min_value=0.0, max_value=150.0, value=50.0)
#clicked_previous_emails = st.sidebar.selectbox("Clicked Previous Emails", [0, 1], index=0)
#device_type = st.sidebar.selectbox("Device Type", [0, 1], index=0)
clicked_previous_emails = 1 if st.selectbox("Clicked Previous Emails", ["No", "Yes"], index=0) == "Yes" else 0
device_type = 1 if st.selectbox("Device Type", ["Desktop", "Mobile"], index=0) == "Mobile" else 0


# Create a DataFrame for the input features
input_data = pd.DataFrame({
    "Customer_Age": [customer_age],
    "Emails_Opened": [emails_opened],
    "Emails_Clicked": [emails_clicked],
    "Purchase_History": [purchase_history],
    "Time_Spent_On_Website": [time_spent_on_website],
    "Days_Since_Last_Open": [days_since_last_open],
    "Customer_Engagement_Score": [customer_engagement_score],
    "Clicked_Previous_Emails": [clicked_previous_emails],
    "Device_Type": [device_type]
})



# Scale the input data
scaled_data = scaler.transform(input_data)

# Predict button
if st.button("🔮 Predict"):
    prediction = model.predict(scaled_data)[0]
    result = "Opened Previous Email" if prediction == 1 else "Did Not Open Previous Email"
    st.success(f"Prediction: {result}")

st.markdown("---")
st.write("Developed by Umme Umara")

