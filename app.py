import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Algerian Forest Fire Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR UNIQUE STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF4B4B, #FF8F00);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #FF8F00, #FF4B4B);
        box-shadow: 0 6px 8px rgba(255,75,75,0.4);
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL & DATA CACHE ---
@st.cache_resource
def load_assets():
    model_path = "model/model.pkl"
    scaler_path = "model/scaler.pkl"
    
    model = pickle.load(open(model_path, "rb")) if os.path.exists(model_path) else None
    scaler = pickle.load(open(scaler_path, "rb")) if os.path.exists(scaler_path) else None
    return model, scaler

model, scaler = load_assets()

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/color/96/fire-element.png", width=80)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Section", ["🔥 Fire Risk Predictor", "📊 Exploratory Data Analysis"])

st.sidebar.markdown("---")
st.sidebar.info("This app evaluates meteorological data to predict forest fire susceptibility in Algerian regions.")

# --- APP MODE 1: PREDICTOR ---
if app_mode == "🔥 Fire Risk Predictor":
    st.title("Algerian Forest Fire Risk Estimator")
    st.markdown("Provide the weather conditions below to evaluate the fire risk.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temperature = st.number_input("Temperature (°C)", min_value=20.0, max_value=50.0, value=30.0)
            rh = st.number_input("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=50.0)
            
        with col2:
            ws = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=30.0, value=15.0)
            rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=20.0, value=0.0)
            
        with col3:
            ffmc = st.number_input("FFMC Index", min_value=10.0, max_value=100.0, value=85.0)
            dmc = st.number_input("DMC Index", min_value=1.0, max_value=200.0, value=25.0)
            dc = st.number_input("DC Index", min_value=5.0, max_value=900.0, value=40.0)
            isi = st.number_input("ISI Index", min_value=0.0, max_value=60.0, value=6.0)

        submit_button = st.form_submit_button(label="Predict Fire Risk")

    if submit_button:
        if model is None or scaler is None:
            st.error("Model or Scaler files not found! Please make sure model.pkl and scaler.pkl are placed in the 'model/' directory.")
        else:
            input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, dc, isi]])
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)
            
            st.markdown("### Prediction Results")
            if prediction[0] > 0.5:
                st.error("🚨 **High Risk Alert:** Conditions indicate a high likelihood of forest fire hazard!")
            else:
                st.success("✅ **Low Risk:** Conditions are relatively safe.")

# --- APP MODE 2: EDA ---
elif app_mode == "📊 Exploratory Data Analysis":
    st.title("Exploratory Data Analysis Dashboard")
    st.markdown("Visualizing correlations between meteorological conditions and forest fires.")
    
    data_path = "Algerian_forest_fires_dataset_CLEANED.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.subheader("Dataset Overview")
        st.dataframe(df.head())
    else:
        st.warning("Dataset CSV file not found in root directory.")