import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# OpenWeatherMap API Key
API_KEY = "c78bc03c5ef520708d5d810783404823"

# 1. Page Configuration
st.set_page_config(page_title="Crop & Fertilizer Recommendation", layout="wide")

# 2. Data Load ani Model Train karu
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

# 3. Web App chi Design (UI)
st.title("🌾 Crop Yield & Fertilizer Recommendation System")
st.write("Matiche ani Hamanache praman taka, tumhala konte pik ghyayche te AI sangel!")
st.divider()

# --- LIVE WEATHER FEATURE ---
st.subheader("🌦️ Get Live Weather via City")

city = st.selectbox(
    "Tumchya gavatil/shahratil chalu haman sathi nav nivda:", 
    ["Pune", "Mumbai", "Nashik", "Nagpur", "Aurangabad", "Kolhapur", "Solapur", "Other (Type Below)"]
)

if city == "Other (Type Below)":
    city = st.text_input("Enter City Name:", "Pune")

# बॅकअप हवामान डेटा
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
            st.success(f"📍 {city} che Live API haman यशस्वीरित्या लोड झाले!")
        else:
            city_lower = city.lower()
            if city_lower in weather_backup:
                default_temp = weather_backup[city_lower]["temp"]
                default_humidity = weather_backup[city_lower]["humidity"]
                st.info(f"ℹ️ Smart Backup: {city} चे हवामान डेटाबेसमधून ऑटो-फिल केले आहे.")
    except Exception as weather_error:
        city_lower = city.lower()
        if city_lower in weather_backup:
            default_temp = weather_backup[city_lower]["temp"]
            default_humidity = weather_backup[city_lower]["humidity"]
            st.info(f"ℹ️ Smart Backup: {city} चे हवामान डेटाबेसमधून ऑटो-फिल केले आहे.")

st.divider()

# --- INPUT FIELDS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Mati (Soil Components)")
    n = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=40)
    p = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=40)
    k = st.number_input("Potassium (K)", min_value=0, max_value=300, value=40)
    ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5)

with col2:
    st.subheader("☁️ Haman (Weather)")
    temp = st.number_input("Taapman (Temperature in °C)", min_value=0.0, max_value=50.0, value=default_temp, key="temp_input")
    humidity = st.number_input("Drauvata (Humidity %)", min_value=0.0, max_value=100.0, value=default_humidity, key="humidity_input")
    rainfall = st.number_input("Paus (Rainfall in mm)", min_value=0.0, max_value=500.0, value=150.0)

st.divider()

# --- इथून तुमचा विचारलेला कोड सुरू होतो (फाईलचा शेवटचा भाग) ---
# 4. Prediction & Fertilizer Recommendation
if st.button("🌾 Check Results", type="primary"):
    if model_ready:
        user_data = [[n, p, k, temp, humidity, ph, rainfall]]
        prediction = model.predict(user_data)
        st.balloons()                 
        
        # बरोबर केलेला कोड: prediction[0].upper()
        st.success(f"### 🎉 Tumchya jaminisathi sarvyat uttam pik ahe: **{prediction[0].upper()}**")
        
        st.subheader("💡 Matichya pramananusar khatanchi shifarash (Fertilizer Advice):")
        advice = []
        if n < 40:
            advice.append("⚠️ **Nitrogen che praman kami ahe:** Jaminith Nitrogen vadhavnyasathi **Urea** kiva hiryavaliche khat vapara.")
        elif n > 120:
            advice.append("✅ **Nitrogen che praman jast ahe:** Urfa khat takणे tळा, jaminicha baddal thik karnyasathi danyavargiya pike ghya.")
            
        if p < 40:
            advice.append("⚠️ **Phosphorus che praman kami ahe:** **DAP (Di-Ammonium Phosphate)** kiva Single Super Phosphate (SSP) cha vapor kara.")
            
        if k < 40:
            advice.append("⚠️ **Potassium che praman kami ahe:** Pikaंची javad vadhavnyasathi **MOP (Muriate of Potash)** khat taka.")
            
        if ph < 6.0:
            advice.append("⚠️ **Jameen Aamla (Acidic) ahe:** Maticha pH sudharnyasathi **Chuna (Lime)** vapara.")
        elif ph > 7.5:
            advice.append("⚠️ **Jameen Khaari (Alkaline) ahe:** Maticha pH kammi karnyasathi **Gypsum** cha vapor kara.")
            
        if not advice:
            st.info("👍 Tumchya matitil ghadak vyavasthit ahet, jameen shetisathi uttam ahe!")
        else:
            for item in advice:
                st.write(item)
    else:
        st.warning("Model tayar nahi ahe, please error check kara.")
