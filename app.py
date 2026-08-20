import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# OpenWeatherMap API Key
API_KEY = "c78bc03c5ef520708d5d810783404823"

# 1. Page Configuration
st.set_page_config(page_title="Crop & Fertilizer Recommendation", layout="wide", page_icon="🌾")

# 2. Data Load and Model Training
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv("Crop_recommendation.csv")
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False

model, model_ready = train_model()

# 3. Web App User Interface (UI) Design
st.title("🌾 Crop Yield & Fertilizer Recommendation System")
st.write("Enter the soil and weather conditions, and AI will tell you which crop to grow!")
st.divider()

# --- LIVE WEATHER FEATURE ---
st.subheader("🌦️ Get Live Weather via City")

city = st.selectbox(
    "Select your village/city name for live weather:", 
    ["Pune", "Mumbai", "Nashik", "Nagpur", "Aurangabad", "Kolhapur", "Solapur", "Other (Type Below)"]
)

if city == "Other (Type Below)":
    city = st.text_input("Enter City Name:", "Pune")

# Backup Weather Database (Used if the Live API is down or newly created)
weather_backup = {
    "pune": {"temp": 25.4, "humidity": 78.0},
    "mumbai": {"temp": 28.5, "humidity": 85.0},
    "nashik": {"temp": 24.8, "humidity": 80.0},
    "nagpur": {"temp": 29.1, "humidity": 72.0},
    "aurangabad": {"temp": 26.5, "humidity": 75.0},
    "kolhapur": {"temp": 25.1, "humidity": 82.0},
    "solapur": {"temp": 28.0, "humidity": 68.0}
}

default_temp = 26.0
default_humidity = 75.0

if city:
    url = f"http://openweathermap.org{city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=3).json()
        if response.get("cod") == 200:
            default_temp = float(response["main"]["temp"])
            default_humidity = float(response["main"]["humidity"])
            st.success(f"📍 Live API weather for {city} loaded successfully!")
        else:
            city_lower = city.lower()
            if city_lower in weather_backup:
                default_temp = weather_backup[city_lower]["temp"]
                default_humidity = weather_backup[city_backup]["humidity"]
                st.info(f"ℹ️ Smart Backup: {city} weather loaded from internal database.")
    except Exception as weather_error:
        city_lower = city.lower()
        if city_lower in weather_backup:
            default_temp = weather_backup[city_lower]["temp"]
            default_humidity = weather_backup[city_lower]["humidity"]
            st.info(f"ℹ️ Smart Backup: {city} weather loaded from internal database.")

st.divider()

# --- INPUT FIELDS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Soil Components")
    n = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=40)
    p = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=40)
    k = st.number_input("Potassium (K)", min_value=0, max_value=300, value=40)
    ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5)

with col2:
    st.subheader("☁️ Weather Conditions")
    temp = st.number_input("Temperature in °C", min_value=0.0, max_value=50.0, value=default_temp, key="temp_input")
    humidity = st.number_input("Humidity %", min_value=0.0, max_value=100.0, value=default_humidity, key="humidity_input")
    rainfall = st.number_input("Rainfall in mm", min_value=0.0, max_value=500.0, value=150.0)

st.divider()

# 4. Prediction & Fertilizer Recommendation
if st.button("🌾 Check Results", type="primary"):
    if model_ready:
        user_data = [[n, p, k, temp, humidity, ph, rainfall]]
        prediction = model.predict(user_data)
        st.balloons()                 
        
        st.success(f"### 🎉 The absolute best crop for your soil is: **{prediction[0].upper()}**")
        
        st.subheader("💡 Tailored Chemical Fertilizer Advice:")
        advice = []
        if n < 40:
            advice.append("⚠️ **Low Nitrogen Content:** Please supplement your field using **Urea** or grow nitrogen-fixing legume crops.")
        elif n > 120:
            advice.append("✅ **High Nitrogen Content:** Stop adding chemical nitrogen fertilizers to avoid damaging plant roots.")
            
        if p < 40:
            advice.append("⚠️ **Low Phosphorus Content:** Apply **DAP (Di-Ammonium Phosphate)** or Single Super Phosphate (SSP) to improve root growth.")
            
        if k < 40:
            advice.append("⚠️ **Low Potassium Content:** Add **MOP (Muriate of Potash)** to improve overall crop immunity and yield quality.")
            
        if ph < 6.0:
            advice.append("⚠️ **Acidic Soil Warning:** Spread **Lime (Chuna)** across the field to normalize and increase the pH scale.")
        elif ph > 7.5:
            advice.append("⚠️ **Alkaline Soil Warning:** Mix **Gypsum** into the field soil to lower the high alkaline levels.")
            
        if not advice:
            st.info("👍 Perfect soil structure! Your field chemical properties are extremely well-balanced for farming.")
        else:
            for item in advice:
                st.write(item)
    else:
        st.warning("The AI model is not ready. Please check your data files or terminal logs.")
