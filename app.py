import streamlit as st
import pickle
import numpy as np

# Page title
st.title("🏠 House Price Predictor")

# Load saved model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# User inputs
area = st.number_input("Enter Area (sq.ft)", min_value=300, value=1500)

rooms = st.number_input("Enter Number of Rooms", min_value=1, value=3)

# Predict button
if st.button("Predict Price"):

    # Create input data
    data = np.array([[area, rooms]])

    # Predict price
    price = model.predict(data)

    # Show result
    st.success(f"Predicted House Price: ₹{price[0]:,.2f} Lakhs")