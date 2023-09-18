# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

# Load your trained model
# Replace 'model.pkl' with the actual path to your model file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create a Streamlit web app
st.title("Water Quality Detection")

# Create input elements for user
st.header("Input Parameters")

# Input elements for user
ph = st.slider("pH (0 to 14)", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.slider("Hardness (mg/L)", min_value=0, max_value=500, value=100)
solids = st.slider("Total Dissolved Solids (ppm)", min_value=0, max_value=5000, value=1000)
chloramines = st.slider("Chloramines (ppm)", min_value=0.0, max_value=10.0, value=2.0)
sulfate = st.slider("Sulfate (mg/L)", min_value=0, max_value=500, value=150)
conductivity = st.slider("Conductivity (μS/cm)", min_value=0, max_value=2000, value=500)
organic_carbon = st.slider("Organic Carbon (ppm)", min_value=0, max_value=50, value=10)
trihalomethanes = st.slider("Trihalomethanes (μg/L)", min_value=0, max_value=200, value=50)
turbidity = st.slider("Turbidity (NTU)", min_value=0.0, max_value=10.0, value=5.0)

# Create a button to perform the prediction
if st.button("Predict"):
    # Prepare input data for prediction (use the input values)
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])

    # Make a prediction using your model
    prediction = model.predict(input_data)

    # Display the result
    if prediction == 1:
        st.success("Water Quality: Safe ")
    else:
        st.error("Water Quality: Not Safe")


