import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

# OpenWeatherMap API Key
API_KEY = "c78bc03c5ef520708d5d810783404823"

# 1. Page Configuration
st.set_page_config(page_title="Crop Yield Recommendation", layout="wide")

# 2. Data Load ani Model Train karu
@st.cache_resource
def train_model():
    df = pd.read_csv("Crop_recommendation.csv")
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

try:
    model = train_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"Error: {e}")

# 3. Web App chi Design (UI)
st.title("🌾 Crop Yield & Fertilizer Recommendation System")
st.write("Matiche ani Hamanache praman taka, tumhala konte pik ghyayche te AI sangel ani khatanchi mahiti bhetel!")

st.divider()

# --- LIVE WEATHER FEATURE (STEP 1) ---
st.subheader("🌦️ Get Live Weather via City")
city = st.text_input("Tumchya gavatil/shahratil chalu haman sathi nav taka (e.g., Pune):", "")

default_temp = 25.0
default_humidity = 60.0

if city:
    url = f"http://openweathermap.org{city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url).json()
        if response.get("cod") == 200:
            default_temp = float(response["main"]["temp"])
            default_humidity = float(response["main"]["humidity"])
            st.success(f"📍 {city} che live haman sapadle! Taapman ani Drauvata khali auto-fill zale ahet.")
        else:
            st.error("Shahrace nav sapadle nahi. Please spelling check kara.")
    except Exception as weather_error:
        st.error("Haman cha data anyat adchan yet ahe.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Matiche Praman (Soil)")
    n = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    p = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    k = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    ph = st.number_input("Matiche pH level", min_value=0.0, max_value=14.0, value=6.5)

with col2:
    st.subheader("☁️ Haman (Weather)")
    temp = st.number_input("Taapman (Temperature in °C)", min_value=0.0, max_value=50.0, value=default_temp)
    humidity = st.number_input("Drauvata (Humidity %)", min_value=0.0, max_value=100.0, value=default_humidity)
    rainfall = st.number_input("Paus (Rainfall in mm)", min_value=0.0, max_value=500.0, value=100.0)

st.divider()

# 4. Prediction & Fertilizer Recommendation (STEP 2)
if st.button("🌾 Check Results", type="primary"):
    if model_ready:
        user_data = [[n, p, k, temp, humidity, ph, rainfall]]
        prediction = model.predict(user_data)
        st.balloons() 
        
        # पीक दाखवणे
        st.success(f"### 🎉 Tumchya jaminisathi sarvyat uttam pik ahe: **{prediction[0].upper()}**")
        
        # --- खतांची नवीन लॉजिक सिस्टीम ---
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
