import streamlit as st
import pandas as pd
import requests
import plotly.express as px  # Advanced data visualization engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# OpenWeatherMap API Key
API_KEY = "c78bc03c5ef520708d5d810783404823"

# 1. Page Configuration
st.set_page_config(page_title="Agri-Smart Enterprise AI", layout="wide", page_icon="🚜")

# 2. Data Load & Model Training
@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv("Crop_recommendation.csv")
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model, df, True
    except Exception as e:
        return None, None, False

model, df, model_ready = load_and_train()

# 3. Sidebar Multi-Page Navigation System
st.sidebar.title("🚜 Navigation Menu")
page = st.sidebar.radio("Go to Page:", ["🤖 AI Recommendation Engine", "📊 Market & Soil Data Analytics", "📖 Crop Requirements Guide"])

# ==============================================================================
# PAGE 1: AI RECOMMENDATION ENGINE
# ==============================================================================
if page == "🤖 AI Recommendation Engine":
    st.title("🌾 Smart Crop Yield & Fertilizer Recommendation System")
    st.write("Enter the soil and weather conditions, and AI will tell you which crop to grow!")
    st.divider()

    st.subheader("🌦️ Get Live Weather via City")
    city = st.selectbox(
        "Select your village/city name for live weather:", 
        ["Pune", "Mumbai", "Nashik", "Nagpur", "Aurangabad", "Kolhapur", "Solapur", "Other (Type Below)"]
    )

    if city == "Other (Type Below)":
        city = st.text_input("Enter City Name:", "Pune")

    # Backup Weather Database
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
                    default_humidity = weather_backup[city_lower]["humidity"]
                    st.info(f"ℹ️ Smart Backup: {city} weather loaded from internal database.")
        except Exception:
            city_lower = city.lower()
            if city_lower in weather_backup:
                default_temp = weather_backup[city_lower]["temp"]
                default_humidity = weather_backup[city_lower]["humidity"]
                st.info(f"ℹ️ Smart Backup: {city} weather loaded from internal database.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧪 Soil Components")
        n = st.number_input("Nitrogen (N Content)", min_value=0, max_value=200, value=40)
        p = st.number_input("Phosphorus (P Content)", min_value=0, max_value=200, value=40)
        k = st.number_input("Potassium (K Content)", min_value=0, max_value=300, value=40)
        ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5)

    with col2:
        st.subheader("☁️ Weather Conditions")
        temp = st.number_input("Temperature in °C", min_value=0.0, max_value=50.0, value=default_temp, key="temp_input")
        humidity = st.number_input("Humidity %", min_value=0.0, max_value=100.0, value=default_humidity, key="humidity_input")
        rainfall = st.number_input("Rainfall in mm", min_value=0.0, max_value=500.0, value=150.0)

    st.divider()

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
                advice.append("⚠️ **Low Potassium Content:** Add **MOP (Muriate of Potash)** to increase pest resilience.")
                
            if ph < 6.0:
                advice.append("⚠️ **Acidic Soil Warning:** Spread **Lime (Chuna)** across the field to normalize and increase the pH scale.")
            elif ph > 7.5:
                advice.append("⚠️ **Alkaline Soil Warning:** Mix **Gypsum** into the field soil to lower the high alkaline levels.")
                
            if not advice:
                st.info("👍 Perfect soil structure! Your field chemical properties are extremely well-balanced for farming.")
            else:
                for item in advice:
                    st.write(item)

# ==============================================================================
# PAGE 2: MARKET & SOIL DATA ANALYTICS
# ==============================================================================
elif page == "📊 Market & Soil Data Analytics":
    st.title("📊 Big Data Farming Analytics Dashboard")
    st.write("Explore chemical charts built straight out of your machine learning dataset.")
    st.divider()
    
    if model_ready:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌱 Nutrient Distribution Across Crops")
            feature = st.selectbox("Select Element to Graph:", ["N", "P", "K", "rainfall"])
            fig1 = px.box(df, x="label", y=feature, title=f"Required levels of {feature} across multiple target crops", color="label")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader(" Environmental Relationships")
            fig2 = px.scatter(df, x="temperature", y="humidity", color="label", title="Climatic Clustering of Crops based on Temperature vs Humidity")
            st.plotly_chart(fig2, use_container_width=True)

# ==============================================================================
# PAGE 3: CROP REQUIREMENTS GUIDE
# ==============================================================================
elif page == "📖 Crop Requirements Guide":
    st.title("📖 Master Crop Encyclopedia")
    st.write("Select any crop to retrieve its scientifically backed standard threshold metrics.")
    st.divider()
    
    if model_ready:
        crop_list = sorted(df['label'].unique())
        selected_crop = st.selectbox("Choose a Crop to Study:", crop_list)
        
        crop_data = df[df['label'] == selected_crop]
        
        st.subheader(f" Ideal Ecosystem Metrics for Growing {selected_crop.upper()}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Nitrogen (N)", f"{round(crop_data['N'].mean(), 1)} mg/kg")
        c2.metric("Avg Phosphorus (P)", f"{round(crop_data['P'].mean(), 1)} mg/kg")
        c3.metric("Avg Potassium (K)", f"{round(crop_data['K'].mean(), 1)} mg/kg")
        c4.metric("Optimal Soil pH", f"{round(crop_data['ph'].mean(), 2)}")
        
        st.markdown("---")
        st.write(f"This specific dataset contains **{len(crop_data)} field records** to back up this baseline calculation.")
