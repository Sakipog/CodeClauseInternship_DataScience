
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

# Title of the app
st.title("Heart Disease Risk Prediction")

# Input fields for user health metrics
st.header("Enter Your Health Metrics")

age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=200)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)

# Convert gender to binary
gender_binary = 1 if gender == "Male" else 0

# Button to predict the risk
if st.button("Predict Risk"):
    # Prepare input data
    input_data = np.array([[age, gender_binary, cholesterol, blood_pressure]])
    
    # Make prediction
    prediction = model.predict(input_data)
    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
    
    # Display the result
    st.subheader("Prediction Result")
    st.write(f"Risk Level: **{risk_level}**")
