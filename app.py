import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(page_title="Crop Yield Recommendation", layout="wide")

# 2. Data Load ani Model Train karu
@st.cache_resource
def train_model():
    # हा कोड आपोआप तुमच्या 'Crop_recommendation.csv' फाईलला वाचेल!
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
st.title("🌾 Crop Yield Recommendation System")
st.write("Matiche ani Hamanache praman taka, tumhala konte pik ghyayche te AI sangel!")

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
    temp = st.number_input("Taapman (Temperature in °C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Drauvata (Humidity %)", min_value=0.0, max_value=100.0, value=60.0)
    rainfall = st.number_input("Paus (Rainfall in mm)", min_value=0.0, max_value=500.0, value=100.0)

st.divider()

# 4. Prediction Button
if st.button("🌾 Check Best Crop", type="primary"):
    if model_ready:
        user_data = [[n, p, k, temp, humidity, ph, rainfall]]
        prediction = model.predict(user_data)
        st.balloons() 
        st.success(f"### 🎉 Tumchya jaminisathi sarvyat uttam pik ahe: **{prediction[0].upper()}**")
    else:
        st.warning("Model tayar nahi ahe, please error check kara.")
