import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# OpenWeatherMap API Key
API_KEY = "9acd0f1436e8c1375fcd7fd749c1b5cb"

# 1. Page Configuration
st.set_page_config(page_title="Crop & Fertilizer Recommendation", layout="wide")

# 2. Data Load ani Model Train karu
@st.cache_resource
def train_model():
    try:
        # Crop_recommendation.csv फाईल वाचणे
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
st.write("Matiche ani Hamanache praman taka, tumhala konte pik ghyayche te AI sangel ani khatanchi mahiti bhetel!")
st.divider()

# --- LIVE WEATHER FEATURE ---
st.subheader("🌦️ Get Live Weather via City")
city = st.text_input("Tumchya gavatil/shahratil chalu haman sathi nav taka (e.g., Pune):", "")

# डिफॉल्ट व्हॅल्यूज
default_temp = 25.0
default_humidity = 60.0

if city:
    # तुमच्या कोडमधील URL ची चूक इथे दुरुस्त केली आहे
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
    # थेट लाईव्ह हवामानाचे तापमान आणि दमटपणा इथे ऑटो-लोढ होईल
    temp = st.number_input("Taapman (Temperature in °C)", min_value=0.0, max_value=50.0, value=default_temp, key="temp_input")
    humidity = st.number_input("Drauvata (Humidity %)", min_value=0.0, max_value=100.0, value=default_humidity, key="humidity_input")
    rainfall = st.number_input("Paus (Rainfall in mm)", min_value=0.0, max_value=500.0, value=100.0)

st.divider()

# 4. Prediction & Fertilizer Recommendation
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

