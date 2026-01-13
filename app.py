import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Logistic Regression Deployment")

st.write("Enter values to predict the output")

feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)

input_data = np.array([[feature1, feature2]])

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction result
