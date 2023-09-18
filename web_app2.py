# water_quality_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress the deprecation warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load your trained model
# Replace 'model.pkl' with the actual path to your model file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create a Streamlit web app
st.title("Water Quality Detection")

# Create input elements for user
st.header("Upload CSV Data")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.subheader("Uploaded Data")
    st.write(df)

    # Data Visualization Section
    st.header("Data Visualization")

    # Example: Bar chart of Hardness values
    st.subheader("Hardness Values Bar Chart")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Potability', y='Hardness', data=df)
    st.pyplot()

    # Add more visualizations as needed for other features
    # Example: Pair plot for multiple features
    st.subheader("Pair Plot for Multiple Features")
    sns.pairplot(df, hue='Potability')
    st.pyplot()

    # Example: Box plot for Total Dissolved Solids (TDS)
    st.subheader("Box Plot for Total Dissolved Solids (TDS)")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Potability', y='Solids', data=df)
    st.pyplot()

    # Example: Histogram of Chloramines
    st.subheader("Histogram of Chloramines")
    plt.figure(figsize=(8, 6))
    sns.histplot(df, x='Chloramines', hue='Potability', bins=20, kde=True)
    st.pyplot()

    # Example: Correlation Heatmap
    st.subheader("Correlation Heatmap")
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot()


    # Input elements for user
    st.header("Input Parameters")

    # Example input elements (customize as per your model's requirements)
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
            st.success("Water Quality: Safe")
        else:
            st.error("Water Quality: Not Safe")
