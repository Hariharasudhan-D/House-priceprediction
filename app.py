import streamlit as st
import pickle
import numpy as np

# Set page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .title {
        font-family: 'Inter', sans-serif;
        color: #1e3a8a;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #4b5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .result-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 25px;
    }
    .result-val {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .footer {
        text-align: center;
        color: #9ca3af;
        margin-top: 50px;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please run 'model.py' first to train and save the model.")
    st.stop()

# Header Section
st.markdown("<h1 class='title'>🏠 House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Get instant, AI-powered property valuations based on area and rooms</p>", unsafe_allow_html=True)

# Main UI Card
st.markdown("<div class='card'>", unsafe_allow_html=True)

# Inputs
st.subheader("Property Specifications")
col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (in sq. ft)", 
        min_value=300, 
        max_value=10000, 
        value=1500, 
        step=50,
        help="Total built-up area of the house in square feet"
    )

with col2:
    rooms = st.number_input(
        "Number of Rooms", 
        min_value=1, 
        max_value=10, 
        value=3, 
        step=1,
        help="Total number of bedrooms/rooms in the house"
    )

# Prediction Button
if st.button("Predict Price", use_container_width=True):
    # Predict using model (expects 2D array: [[Area, Rooms]])
    prediction = model.predict(np.array([[area, rooms]]))
    predicted_price = prediction[0]
    
    # Format and display result
    st.markdown(f"""
        <div class='result-container'>
            <p style='margin-bottom: 5px; font-size: 1.1rem; opacity: 0.9;'>Estimated Property Value</p>
            <p class='result-val'>${predicted_price:,.2f}k</p>
            <p style='margin-top: 5px; font-size: 0.85rem; opacity: 0.8;'>Based on Linear Regression Model</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Details Card
with st.expander("ℹ️ About the Model"):
    st.markdown("""
        - **Algorithm**: Linear Regression
        - **Features used**: Area (sq. ft), Rooms
        - **Model Accuracy (R² Score)**: ~98.5%
        - Trained on historical property pricing datasets.
    """)

# Footer
st.markdown("<p class='footer'>Developed for Hariharasudhan's Resume Portfolio</p>", unsafe_allow_html=True)
