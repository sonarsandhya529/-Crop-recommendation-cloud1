import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------------------------
# प्रोग्रॅम कॉन्फिगरेशन आणि कॉन्स्टंट्स (Enterprise Configuration)
# ------------------------------------------------------------------------------
API_KEY = "c78bc03c5ef520708d5d810783404823"
WEATHER_API_ENDPOINT = "http://openweathermap.org"

# पेज सेटिंग्ज (Professional Wide Layout)
st.set_page_config(
    page_title="Agri-Smart Enterprise Management System",
    layout="wide",
    page_icon="🚜",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# मशीन लर्निंग मॉडेल ट्रेनिंग इंजिन (Core Data Processing Engine)
# ------------------------------------------------------------------------------
@st.cache_resource
def initialize_predictive_engine():
    """ही सिस्टीम डेटासेट लोड करते आणि रँडम फॉरेस्ट अल्गोरिदम वापरून मॉडेल ट्रेन करते."""
    try:
        # डेटासेट लोड करणे
        df = pd.read_csv("Crop_recommendation.csv")
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        X = df[features]
        y = df['label']
        
        # ८०% ट्रेनिंग आणि २०% टेस्टिंग डेटा स्प्लिट
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # रँडम फॉरेस्ट क्लासिफायर मॉडेल तयार करणे
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # इंटरनेट बंद असल्यास डेटासेटमधील सरासरी व्हॅल्यूज मॅप करणे (Failsafe Mechanism)
        fallback_metrics = df.groupby('label').mean().to_dict(orient='index')
        global_averages = {col: float(df[col].mean()) for col in features}
        
        return model, df, fallback_metrics, global_averages, True
    except Exception as e:
        st.error(f"Critical System Core Deployment Failure: {str(e)}")
        return None, pd.DataFrame(), {}, {}, False

model, master_df, dynamic_fallback_db, global_macro_averages, engine_status = initialize_predictive_engine()

# ------------------------------------------------------------------------------
# डाव्या बाजूचा मुख्य नेव्हिगेशन मेनू (Sidebar Multi-Page Layout)
# ------------------------------------------------------------------------------
st.sidebar.title("🚜 Core Modules")
selected_module = st.sidebar.radio(
    "Select Operating Environment:",
    [
        "🤖 AI Recommendation Engine",
        "📊 Applied Agrochemical Analytics",
        "📖 Agronomic Taxonomy Encyclopedia",
        "💰 Operational Financial Estimator"
    ]
)

# डिफॉल्ट हवामान सेट करणे (बॅकअप म्हणून)
ambient_temperature = global_macro_averages.get('temperature', 25.0)
ambient_humidity = global_macro_averages.get('humidity', 70.0)

# ==============================================================================
# मॉड्यूल १: एआय पीक शिफारस इंजिन (AI RECOMMENDATION ENGINE)
# ==============================================================================
if selected_module == "🤖 AI Recommendation Engine":
    st.title("🌾 Precision Agriculture Crop & Fertilizer Optimization Engine")
    st.write("Production Framework: Real-time multivariate optimization engine running on a Random Forest classification model topology.")
    st.divider()

    st.subheader("🌦️ Ambient Climate Integration Pipeline")
    target_geography = st.selectbox(
        "Select Regional Telemetry Node Location:",
        ["Pune", "Mumbai", "Nashik", "Nagpur", "Aurangabad", "Kolhapur", "Solapur", "Manual / Override Node Location"]
    )

    if target_geography == "Manual / Override Node Location":
        target_geography = st.text_input("Enter Target City Name Specification:", "Pune")

    if target_geography:
        query_parameters = {"q": target_geography, "appid": API_KEY, "units": "metric"}
        try:
            # थेट इंटरनेटवरून लाईव्ह हवामान डेटा मिळवणे
            telemetry_payload = requests.get(WEATHER_API_ENDPOINT, params=query_parameters, timeout=3).json()
            if str(telemetry_payload.get("cod")) == "200":
                ambient_temperature = float(telemetry_payload["main"]["temp"])
                ambient_humidity = float(telemetry_payload["main"]["humidity"])
                st.success(f"✅ Real-time telemetry pipeline established. Node: [{target_geography}] values auto-injected successfully.")
            else:
                target_key = target_geography.lower()
                if target_key in dynamic_fallback_db:
                    ambient_temperature = dynamic_fallback_db[target_key].get('temperature', ambient_temperature)
                    ambient_humidity = dynamic_fallback_db[target_key].get('humidity', ambient_humidity)
                st.info(f"ℹ️ Operational Safety Default: Falling back to statistical baselines for node validation.")
        except Exception:
            st.info(f"ℹ️ Network Timeout Fail-safe: Resolving inputs via system microclimatic baselines.")

    st.divider()

    # स्क्रीनचे दोन उभ्या कॉलम्समध्ये विभाजन
    input_column_left, input_column_right = st.columns(2)
    with input_column_left:
        st.subheader("🧪 Soil Matrix Chemical Diagnostics")
        soil_nitrogen = st.number_input("Elemental Nitrogen Content (N - mg/kg)", min_value=0, max_value=250, value=50)
        soil_phosphorus = st.number_input("Available Phosphorus Content (P - mg/kg)", min_value=0, max_value=250, value=50)
        soil_potassium = st.number_input("Exchangeable Potassium Content (K - mg/kg)", min_value=0, max_value=350, value=50)
        soil_ph_level = st.number_input("Soil Active Acidity (pH Logarithmic Scale)", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    with input_column_right:
        st.subheader("☁️ Macroclimate Atmospheric Boundaries")
        input_temp = st.number_input("Mean Ambient Temperature (°C)", min_value=0.0, max_value=60.0, value=ambient_temperature, step=0.1, key="production_temp_node")
        input_humidity = st.number_input("Relative Air Humidity Percentage (%)", min_value=0.0, max_value=100.0, value=ambient_humidity, step=0.1, key="production_humid_node")
        estimated_rainfall = st.number_input("Cumulative Seasonal Precipitative Runoff (Rainfall in mm)", min_value=0.0, max_value=600.0, value=150.0, step=10.0)

    st.divider()

    if st.button("🌾 Compute Diagnostics & Execute Predictions", type="primary"):
        if engine_status and model is not None:
            multivariate_vector = [[soil_nitrogen, soil_phosphorus, soil_potassium, input_temp, input_humidity, soil_ph_level, estimated_rainfall]]
            deterministic_prediction = model.predict(multivariate_vector)
            st.balloons()                 
            
            st.success(f"### 🎯 Optimal Crop Classification Target Resolved: **{deterministic_prediction[0].upper()}**")
            
            # --- पाणी व्यवस्थापन प्रणाली (Irrigation Sizing Logic) ---
            st.subheader("💧 Hydraulic Fluid Sizing & Irrigation Architecture Recommendation:")
            if estimated_rainfall < 100.0:
                st.warning("⚠️ **High Hydrological Stress Detected:** Sub-surface Drip Irrigation (SDI) layout required to maximize distribution efficiency indices.")
            elif estimated_rainfall > 250.0:
                st.info("🌧️ **High Volumetric Runoff Context:** Design deep superficial open drainage ditches to mitigate anaerobic root zone conditions.")
            else:
                st.info("✅ **Equilibrium Hydro-Cycle:** Standard overhead mechanized solid-set sprinkler loops are topologically viable.")

            # --- खतांचे वैज्ञानिक नियोजन (Agrochemical Rectification) ---
            st.subheader("💡 Tailored Chemical Fertilizer Matrix Rectification Sizing:")
            diagnostic_logs = []
            if soil_nitrogen < 40:
                diagnostic_logs.append("⚠️ **Primary Nutrient Deficiency (N):** Apply amide-based **Urea formulations** or introduce rotational leguminous green manures.")
            elif soil_nitrogen > 120:
                diagnostic_logs.append("✅ **Nitrogen Toxic Threshold:** Immediately halt organic manure loading cycles to secure structural crop equilibrium.")
                
            if soil_phosphorus < 40:
                diagnostic_logs.append("⚠️ **Secondary Deficiency (P):** Treat with concentrated **DAP (Di-Ammonium Phosphate)** or water-soluble Single Super Phosphate (SSP) lines.")
                
            if soil_potassium < 40:
                diagnostic_logs.append("⚠️ **Potassium Inadequacy Framework:** Apply granulate **MOP (Muriate of Potash)** protocols to build cellular cell-wall elasticity.")
                
            if soil_ph_level < 6.0:
                diagnostic_logs.append("⚠️ **Acidic Substrate Saturation:** Apply processed agricultural **Lime (Calcium Carbonate)** profiles to raise operational base saturation.")
            elif soil_ph_level > 7.5:
                diagnostic_logs.append("⚠️ **Alkaline Substrate Satiation:** Incorporate mineral **Gypsum (Calcium Sulfate)** into plowing depths to break down high sodium compounds.")
                
            if not diagnostic_logs:
                st.info("👍 Homeostatic chemical equilibrium achieved. Soil substrate metrics comply with strict farming structural specifications.")
            else:
                for log_item in diagnostic_logs:
                    st.write(log_item)
        else:
            st.error("System Runtime Invalidation: The downstream core execution model failed validation passes.")

# ==============================================================================
# मॉड्यूल २: डेटा ॲनालिटिक्स डॅशबोर्ड (APPLIED AGROCHEMICAL ANALYTICS)
# ==============================================================================
