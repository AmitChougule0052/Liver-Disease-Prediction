
import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('liver_model.pkl', 'rb'))

st.title("Liver Disease Prediction")

# Collect user input
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ("Male", "Female"))
total_bilirubin = st.number_input("Total Bilirubin", value=1.0)
direct_bilirubin = st.number_input("Direct Bilirubin", value=0.3)
alkaline_phosphotase = st.number_input("Alkaline Phosphotase", value=200)
alamine_aminotransferase = st.number_input("Alamine Aminotransferase", value=30)
aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", value=30)
total_proteins = st.number_input("Total Proteins", value=6.5)
albumin = st.number_input("Albumin", value=3.5)
albumin_globulin_ratio = st.number_input("Albumin/Globulin Ratio", value=1.0)

# Convert gender to binary
gender_binary = 1 if gender == "Male" else 0

# Predict
if st.button("Predict"):
    features = np.array([[age, gender_binary, total_bilirubin, direct_bilirubin,
                          alkaline_phosphotase, alamine_aminotransferase,
                          aspartate_aminotransferase, total_proteins, albumin,
                          albumin_globulin_ratio]])
    prediction = model.predict(features)
    result = "Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease"
    st.success(result)
