# ---------------- AQI Forecasting Web App ----------------

# Section 1 Imported important libraries
import streamlit as st  # type: ignore
import numpy as np
import pandas as pd
import joblib
import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
import requests
import time
import plotly.graph_objects as go  # type: ignore

# Section 2 Configuration of page and title
st.set_page_config(page_title="AQI Forecast App", layout="wide")

# Section 3 Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
    color: #111111 !important;
}

p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6 {
    color: #111111 !important;
}

.stApp { background-color: #f0f4f8; }

.card {
    background: white; border-radius: 12px;
    padding: 20px 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 8px;
}

.big-aqi { font-size: 3.5rem; font-weight: 700; line-height: 1; }

.label {
    font-size: 0.75rem; color: #888; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;
}

.value { font-size: 1.6rem; font-weight: 600; color: #111111; }
.unit  { font-size: 0.8rem; color: #aaa; }

.badge {
    display: inline-block; padding: 6px 18px; border-radius: 20px;
    font-size: 0.85rem; font-weight: 600; margin-top: 10px;
}

.stButton > button {
    background-color: #4361ee; color: white; border: none;
    border-radius: 8px; padding: 10px 28px;
    font-family: 'Poppins', sans-serif; font-weight: 500; font-size: 0.9rem;
}
.stButton > button:hover { background-color: #3a0ca3; }

[data-testid="stSidebar"] {
    background-color: #ffffff; border-right: 1px solid #e9ecef;
    color: #111111 !important;
}
[data-testid="stSidebar"] * { color: #111111 !important; }

/* dropdown visibility fix */
div[data-baseweb="select"] > div { background-color: #ffffff !important; color: #111111 !important; }
div[data-baseweb="select"] span  { color: #111111 !important; }
div[data-baseweb="popover"] *    { background-color: #ffffff !important; color: #111111 !important; }
div[data-baseweb="menu"]         { background-color: #ffffff !important; }
div[data-baseweb="menu"] li      { color: #111111 !important; background-color: #ffffff !important; }
div[data-baseweb="menu"] li:hover{ background-color: #f0f4f8 !important; color: #111111 !important; }
</style>
""", unsafe_allow_html=True)

# Section 4 AQI category + health advice helpers
def get_aqi_category(aqi):
    if aqi <= 50:  return "Good",            "#2ecc71", "#e8f8f2"
    if aqi <= 100: return "Moderate",         "#f39c12", "#fef9e7"
    if aqi <= 150: return "Sensitive Groups", "#e67e22", "#fef0e6"
    if aqi <= 200: return "Unhealthy",        "#e74c3c", "#fdedec"
    if aqi <= 300: return "Very Unhealthy",   "#8e44ad", "#f5eef8"
    return               "Hazardous",         "#7b241c", "#f9ebea"

def health_advice(aqi):
    if aqi <= 50:  return "Air is clean! Great day for outdoor activities."
    if aqi <= 100: return "Air is okay. Sensitive people should be cautious."
    if aqi <= 150: return "Sensitive groups should reduce outdoor time."
    if aqi <= 200: return "Unhealthy for everyone. Limit outdoor activities."
    if aqi <= 300: return "Very unhealthy. Stay indoors if possible."
    return               "Hazardous! Avoid going outside completely."

# Section 5 AQI formula + tier/city PM2.5 caps
def calculate_aqi(pm25):
    if pm25 <= 30:  return (50 / 30) * pm25
    if pm25 <= 60:  return ((100-51)  / (60-31))  * (pm25-31)  + 51
    if pm25 <= 90:  return ((200-101) / (90-61))  * (pm25-61)  + 101
    if pm25 <= 120: return ((300-201) / (120-91)) * (pm25-91)  + 201
    if pm25 <= 250: return ((400-301) / (250-121))* (pm25-121) + 301
    return 500

# tier-based caps — same tiers as collect_training.py and train.py
TIER_PM25_CAPS = {
    1: 150,   # Very polluted
    2: 100,   # Moderately polluted
    3:  70,   # Less polluted
    4:  40,   # Clean / hills
}

# city-level overrides for specific known cases
CITY_PM25_CAPS = {
    "Shimla": 35, "Manali": 30, "Leh": 25, "Gangtok": 30, "Aizawl": 30,
    "Solan": 35,  "Mandi": 40,  "Dharamshala": 35,
    "Srinagar": 45, "Anantnag": 40, "Baramulla": 40, "Jammu": 55,
    "Haridwar": 75, "Roorkee": 75, "Dehradun": 70,
    "London": 60, "Tokyo": 35, "New York": 25, "Paris": 30, "Sydney": 20,
    "Bangalore": 60, "Chennai": 65, "Kochi": 50, "Pune": 70, "Hyderabad": 70,
}

EXPECTED_FEATURES = [
    "pm2_5","pm10","no2","so2","o3","co",
    "temp_c","humidity","windspeed_kph","pressure_mb",
]

# Section 5a State → cities mapping with coordinates
# used for nearest-neighbour fallback and grouped dropdown
STATE_CITIES = {
    "Delhi":              {"tier": 1, "cities": [("Delhi", 28.6139, 77.2090)]},
    "Uttar Pradesh":      {"tier": 1, "cities": [
        ("Lucknow",26.8467,80.9462),("Kanpur",26.4499,80.3319),("Agra",27.1767,78.0081),
        ("Varanasi",25.3176,82.9739),("Meerut",28.9845,77.7064),("Ghaziabad",28.6692,77.4538),
        ("Allahabad",25.4358,81.8463),("Bareilly",28.3670,79.4304),("Moradabad",28.8386,78.7733),
        ("Gorakhpur",26.7606,83.3732),("Mathura",27.4924,77.6737)]},
    "Punjab":             {"tier": 1, "cities": [
        ("Amritsar",31.6340,74.8723),("Ludhiana",30.9010,75.8573),
        ("Jalandhar",31.3260,75.5762),("Patiala",30.3398,76.3869)]},
    "Haryana":            {"tier": 1, "cities": [
        ("Gurugram",28.4595,77.0266),("Faridabad",28.4089,77.3178),
        ("Ambala",30.3782,76.7767),("Hisar",29.1492,75.7217)]},
    "Bihar":              {"tier": 1, "cities": [
        ("Patna",25.5941,85.1376),("Gaya",24.7955,85.0002),
        ("Muzaffarpur",26.1209,85.3647),("Bhagalpur",25.2425,86.9842)]},
    "Chandigarh":         {"tier": 2, "cities": [("Chandigarh",30.7333,76.7794)]},
    "Rajasthan":          {"tier": 2, "cities": [
        ("Jaipur",26.9124,75.7873),("Jodhpur",26.2389,73.0243),("Udaipur",24.5854,73.7125),
        ("Kota",25.2138,75.8648),("Ajmer",26.4499,74.6399),("Bikaner",28.0229,73.3119)]},
    "Madhya Pradesh":     {"tier": 2, "cities": [
        ("Bhopal",23.2599,77.4126),("Indore",22.7196,75.8577),("Gwalior",26.2183,78.1828),
        ("Jabalpur",23.1815,79.9864),("Ujjain",23.1765,75.7885)]},
    "Chhattisgarh":       {"tier": 2, "cities": [("Raipur",21.2514,81.6296)]},
    "Gujarat":            {"tier": 2, "cities": [
        ("Ahmedabad",23.0225,72.5714),("Surat",21.1702,72.8311),("Rajkot",22.3039,70.8022),
        ("Vadodara",22.3072,73.1812),("Bhavnagar",21.7645,72.1519),
        ("Gandhinagar",23.2156,72.6369),("Jamnagar",22.4707,70.0577)]},
    "Maharashtra":        {"tier": 2, "cities": [
        ("Mumbai",19.0760,72.8777),("Pune",18.5204,73.8567),("Nashik",19.9975,73.7898),
        ("Aurangabad",19.8762,75.3433),("Nagpur",21.1458,79.0882),("Solapur",17.6599,75.9064),
        ("Kolhapur",16.7050,74.2433),("Thane",19.2183,72.9781)]},
    "West Bengal":        {"tier": 2, "cities": [
        ("Kolkata",22.5726,88.3639),("Howrah",22.5958,88.2636),("Siliguri",26.7271,88.3953),
        ("Durgapur",23.5204,87.3119),("Asansol",23.6739,86.9524)]},
    "Jharkhand":          {"tier": 2, "cities": [
        ("Ranchi",23.3441,85.3096),("Jamshedpur",22.8046,86.2029),
        ("Dhanbad",23.7957,86.4304),("Bokaro",23.6693,86.1511)]},
    "Odisha":             {"tier": 2, "cities": [
        ("Bhubaneswar",20.2961,85.8245),("Cuttack",20.4625,85.8830),
        ("Rourkela",22.2604,84.8536),("Sambalpur",21.4669,83.9756)]},
    "Telangana":          {"tier": 2, "cities": [
        ("Hyderabad",17.3850,78.4867),("Warangal",17.9784,79.5941),("Nizamabad",18.6725,78.0941)]},
    "Andhra Pradesh":     {"tier": 2, "cities": [
        ("Visakhapatnam",17.6868,83.2185),("Vijayawada",16.5062,80.6480),
        ("Tirupati",13.6288,79.4192),("Guntur",16.3067,80.4365),("Nellore",14.4426,79.9865)]},
    "Karnataka":          {"tier": 3, "cities": [
        ("Bangalore",12.9716,77.5946),("Mysore",12.2958,76.6394),("Mangalore",12.9141,74.8560),
        ("Hubli",15.3647,75.1240),("Belgaum",15.8497,74.4977)]},
    "Tamil Nadu":         {"tier": 3, "cities": [
        ("Chennai",13.0827,80.2707),("Coimbatore",11.0168,76.9558),("Madurai",9.9252,78.1198),
        ("Tiruchirappalli",10.7905,78.7047),("Salem",11.6643,78.1460),("Tirunelveli",8.7139,77.7567)]},
    "Kerala":             {"tier": 3, "cities": [
        ("Kochi",9.9312,76.2673),("Thiruvananthapuram",8.5241,76.9366),
        ("Kozhikode",11.2588,75.7804),("Thrissur",10.5276,76.2144),("Kollam",8.8932,76.6141)]},
    "Goa":                {"tier": 3, "cities": [("Panaji",15.4909,73.8278),("Margao",15.2993,73.9862)]},
    "Uttarakhand":        {"tier": 3, "cities": [
        ("Dehradun",30.3165,78.0322),("Haridwar",29.9457,78.1642),("Roorkee",29.8543,77.8880)]},
    "Himachal Pradesh":   {"tier": 4, "cities": [
        ("Shimla",31.1048,77.1734),("Manali",32.2396,77.1887),("Dharamshala",32.2190,76.3234),
        ("Solan",30.9045,77.0967),("Mandi",31.7080,76.9318)]},
    "Jammu and Kashmir":  {"tier": 4, "cities": [
        ("Srinagar",34.0837,74.7973),("Jammu",32.7266,74.8570),
        ("Anantnag",33.7311,75.1487),("Baramulla",34.1980,74.3636)]},
    "Ladakh":             {"tier": 4, "cities": [("Leh",34.1526,77.5770)]},
    "Northeast":          {"tier": 4, "cities": [
        ("Guwahati",26.1445,91.7362),("Shillong",25.5788,91.8933),("Imphal",24.8170,93.9368),
        ("Agartala",23.8315,91.2868),("Dibrugarh",27.4728,94.9120),("Silchar",24.8333,92.7789),
        ("Gangtok",27.3389,88.6065),("Aizawl",23.7271,92.7176)]},
}

# build flat lookup: city_name -> (lat, lon, state, tier)
ALL_TRAINED_CITIES = {}
for state, info in STATE_CITIES.items():
    for city, lat, lon in info["cities"]:
        ALL_TRAINED_CITIES[city] = {"lat": lat, "lon": lon, "state": state, "tier": info["tier"]}

# Section 5b Nearest-neighbour fallback
# given a lat/lon, finds the closest trained city using Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def find_nearest_trained_city(lat, lon):
    best_city, best_dist = None, float("inf")
    for city, info in ALL_TRAINED_CITIES.items():
        d = haversine(lat, lon, info["lat"], info["lon"])
        if d < best_dist:
            best_dist = d
            best_city = city
    return best_city, best_dist, ALL_TRAINED_CITIES[best_city]

# Section 5c Geocoding helper
def get_coordinates(city_name):
    api_key = st.secrets["OPENWEATHER_API_KEY"]
    r = requests.get(
        f"https://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}",
        timeout=10)
    data = r.json()
    if not data:
        raise ValueError(f"Couldn't find '{city_name}'. Please check spelling.")
    return data[0]["lat"], data[0]["lon"], data[0].get("name", city_name)

# Section 6 Load all models
@st.cache_resource
def load_models():
    lstm_model = load_model("models/aqi_lstm_model.h5", compile=False)
    lr_model   = joblib.load("models/linear_model.pkl")
    rf_model   = joblib.load("models/random_forest_model.pkl")
    scaler     = joblib.load("models/scaler.pkl")
    metrics    = joblib.load("models/model_metrics.pkl")
    meta       = joblib.load("models/scaler_meta.pkl")
    return lstm_model, lr_model, rf_model, scaler, metrics, meta

lstm_model, lr_model, rf_model, scaler, metrics, meta = load_models()
N_COLS  = meta["N_COLS"]
AQI_COL = meta["AQI_COL"]

def inverse_aqi(scaled_values):
    dummy = np.zeros((len(scaled_values), N_COLS))
    dummy[:, AQI_COL] = scaled_values
    return scaler.inverse_transform(dummy)[:, AQI_COL]

# Section 7 Sidebar
with st.sidebar:
    st.markdown("## About This App")
    st.markdown("""
    This app predicts Air Quality Index (AQI) using machine learning trained on real pollution data.

    **Models used:**
    - LSTM (Deep Learning)
    - Random Forest
    - Linear Regression

    **How it works:**
    1. Select a city from the dropdown
    2. App fetches last 24 hours of air quality data
    3. LSTM predicts next-hour AQI
    4. 24-hour forecast generated

    ---
    **AQI Scale:**
    """)
    for rng, label, color in [
        ("0–50",   "Good",            "#2ecc71"),
        ("51–100", "Moderate",        "#f39c12"),
        ("101–150","Sensitive Groups","#e67e22"),
        ("151–200","Unhealthy",       "#e74c3c"),
        ("201–300","Very Unhealthy",  "#8e44ad"),
        ("301+",   "Hazardous",       "#7b241c"),
    ]:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:10px;padding:4px 0;font-size:0.82rem;'>
            <div style='width:12px;height:12px;border-radius:3px;background:{color};'></div>
            <span style='color:#555;'>{rng}</span>
            <span style='margin-left:auto;color:{color};font-weight:600;'>{label}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem;color:#aaa;'>Data: OpenWeatherMap API</div>", unsafe_allow_html=True)

# Section 8 Header
st.markdown("# AQI Forecast Dashboard")
st.markdown("<p style='color:#666;margin-top:-12px;'>Predicting air quality using LSTM deep learning + real-time data</p>", unsafe_allow_html=True)
st.markdown("---")

# Section 9 Model performance cards
st.markdown("### How well do the models perform?")
metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
col1, col2, col3, col4 = st.columns(4)
icons  = {"Persistence": "", "Linear Regression": "", "Random Forest": "", "LSTM": ""}
colors = {"Persistence": "#aaaaaa", "Linear Regression": "#4cc9f0", "Random Forest": "#4361ee", "LSTM": "#7209b7"}

for col, (_, row) in zip([col1, col2, col3, col4], metrics_df.iterrows()):
    clr  = colors.get(row["Model"], "#4361ee")
    icon = icons.get(row["Model"], "")
    col.markdown(f"""
    <div class='card' style='border-top:4px solid {clr};'>
        <div style='font-size:1rem;font-weight:600;color:#1a1a2e;margin-bottom:12px;'>{icon} {row["Model"]}</div>
        <div style='display:flex;gap:20px;'>
            <div><div class='label'>MAE</div><div style='font-size:1.3rem;font-weight:600;color:{clr};'>{row["MAE"]:.2f}</div></div>
            <div><div class='label'>RMSE</div><div style='font-size:1.3rem;font-weight:600;color:{clr};'>{row["RMSE"]:.2f}</div></div>
            <div><div class='label'>R²</div><div style='font-size:1.3rem;font-weight:600;color:{clr};'>{row["R2"]:.3f}</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

fig_bar = go.Figure(go.Bar(
    x=metrics_df["Model"], y=metrics_df["RMSE"],
    marker_color=[colors.get(m, "#4361ee") for m in metrics_df["Model"]],
    text=[f"{v:.2f}" for v in metrics_df["RMSE"]], textposition="outside",
))
fig_bar.update_layout(
    title=dict(text="RMSE Comparison (lower = better)", font=dict(color="#111111")),
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Poppins, sans-serif", size=12, color="#111111"),
    yaxis=dict(gridcolor="#f0f0f0", tickfont=dict(color="#111111")),
    xaxis=dict(linecolor="#eee", tickfont=dict(color="#111111")),
    height=300, margin=dict(t=40,b=20,l=20,r=20), showlegend=False,
)
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown("---")

# Section 10 Grouped city dropdown by state
st.markdown("### Get Live AQI Prediction")
st.markdown("<p style='color:#666;font-size:0.9rem;'>Cities grouped by state — or type any city manually</p>", unsafe_allow_html=True)

# build dropdown options: state headers + cities underneath
DROPDOWN_OPTIONS = ["-- Type a city manually --"]
for state, info in STATE_CITIES.items():
    DROPDOWN_OPTIONS.append(f"── {state} ──")          # state header (non-selectable visually)
    for city, _, _ in info["cities"]:
        DROPDOWN_OPTIONS.append(city)

c1, c2 = st.columns([2, 1])
with c1:
    selected = st.selectbox("Select City", DROPDOWN_OPTIONS)
    # if user picks a state header treat it as manual
    if selected.startswith("──") or selected == "-- Type a city manually --":
        city_input = st.text_input("Type city name", placeholder="e.g. Kishtwar, Noida, Hapur ...")
    else:
        city_input = selected

with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Predict AQI", use_container_width=True)

# Section 11 Fetch last 24 hours of data
def fetch_last_24_hours(lat, lon, city_tier=None):
    api_key = st.secrets["OPENWEATHER_API_KEY"]
    end = int(time.time()); start = end - 24*3600

    r = requests.get(
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start}&end={end}&appid={api_key}",
        timeout=30)
    r.raise_for_status()
    data = r.json()

    if "list" not in data or not data["list"]:
        raise ValueError("No pollution data from API.")

    pollution_list = data["list"]
    while len(pollution_list) < 24:
        pollution_list = pollution_list * 2
    pollution_list = pollution_list[-24:]

    w = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric",
        timeout=30).json()

    city      = w.get("name", "Unknown")
    temp      = w["main"]["temp"]
    humidity  = w["main"]["humidity"]
    pressure  = w["main"]["pressure"]
    windspeed = w["wind"]["speed"] * 3.6

    rows = []
    for item in pollution_list:
        pm25 = item["components"]["pm2_5"]
        rows.append({
            "pm2_5": pm25, "pm10": item["components"]["pm10"],
            "no2": item["components"]["no2"], "so2": item["components"]["so2"],
            "o3": item["components"]["o3"],   "co":  item["components"]["co"],
            "temp_c": temp, "humidity": humidity,
            "windspeed_kph": windspeed, "pressure_mb": pressure,
            "aqi_index": calculate_aqi(pm25),
        })

    live_df = pd.DataFrame(rows)[EXPECTED_FEATURES + ["aqi_index"]]

    # determine PM2.5 cap — city override first, then tier, then global
    pm25_cap = 250  # global hard ceiling
    for cap_city, cap_val in CITY_PM25_CAPS.items():
        if cap_city.lower() in city.lower() or city.lower() in cap_city.lower():
            pm25_cap = cap_val
            break
    else:
        # no city override — use tier cap if available
        if city_tier is not None:
            pm25_cap = TIER_PM25_CAPS.get(city_tier, 150)

    live_df["pm2_5"]     = live_df["pm2_5"].clip(upper=pm25_cap)
    live_df["aqi_index"] = live_df["pm2_5"].apply(calculate_aqi)

    return live_df, city, {"temp": temp, "humidity": humidity, "windspeed": windspeed, "pressure": pressure}

# Section 12 24-hour rolling forecast
def forecast_next_24_hours(model, scaler, live_df):
    predictions = []; window = live_df.copy()
    for _ in range(24):
        scaled = scaler.transform(window)
        pred_s = model.predict(np.expand_dims(scaled, axis=0), verbose=0)[0, 0]
        pred_r = float(np.clip(inverse_aqi(np.array([pred_s]))[0], 0, 500))
        predictions.append(pred_r)
        new_row = window.iloc[-1].copy(); new_row["aqi_index"] = pred_r
        window  = pd.concat([window.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
    return predictions

# Section 13 Run prediction
if run:
    if not city_input.strip():
        st.warning("Please select a city or type a city name.")
        st.stop()
    try:
        with st.spinner(f"Searching for {city_input}..."):
            lat, lon, resolved_city = get_coordinates(city_input.strip())
        st.caption(f"Found: {resolved_city} ({lat:.4f}, {lon:.4f})")

        # check if city is in trained list
        is_trained = any(
            t.lower() in city_input.lower() or city_input.lower() in t.lower()
            for t in ALL_TRAINED_CITIES.keys()
        )

        # find nearest trained city for tier lookup + warning message
        nearest_city, dist_km, nearest_info = find_nearest_trained_city(lat, lon)
        city_tier = nearest_info["tier"]

        if not is_trained:
            st.warning(
                f"⚠️ **{resolved_city}** is not in the training data. "
                f"Showing estimate based on nearest trained city: **{nearest_city}** "
                f"({dist_km:.0f} km away, {nearest_info['state']}). "
                f"Tier {city_tier} PM2.5 cap applied."
            )

        with st.spinner("Fetching air quality data..."):
            live_df, city, wx = fetch_last_24_hours(lat, lon, city_tier=city_tier)

        st.markdown(f"### Results for **{city}**")
        if not is_trained:
            st.caption(f"Prediction model reference: {nearest_city} ({nearest_info['state']})")

        # Section 13a weather cards
        st.markdown("#### Current Weather")
        wc1, wc2, wc3, wc4 = st.columns(4)
        for col, label, val, unit in [
            (wc1, "Temperature",  f"{wx['temp']:.1f}",      "°C"),
            (wc2, "Humidity",     f"{wx['humidity']}",       "%"),
            (wc3, "Wind Speed",   f"{wx['windspeed']:.1f}", "km/h"),
            (wc4, "Pressure",     f"{wx['pressure']}",       "hPa"),
        ]:
            col.markdown(f"""
            <div class='card' style='text-align:center;'>
                <div class='label'>{label}</div>
                <div class='value'>{val}</div>
                <div class='unit'>{unit}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 13b LSTM prediction
        scaled_data = scaler.transform(live_df)
        pred_scaled = lstm_model.predict(np.expand_dims(scaled_data, axis=0), verbose=0)[0, 0]
        lstm_pred   = float(np.clip(inverse_aqi(np.array([pred_scaled]))[0], 0, 500))
        cat, clr, bg = get_aqi_category(lstm_pred)

        st.markdown("#### Predicted AQI (Next Hour)")
        r1, r2 = st.columns([1, 1.5])

        with r1:
            st.markdown(f"""
            <div class='card' style='text-align:center;background:{bg};border:2px solid {clr};'>
                <div class='label'>LSTM Prediction</div>
                <div class='big-aqi' style='color:{clr};'>{lstm_pred:.0f}</div>
                <div class='badge' style='background:{clr};color:white;'>{cat}</div>
                <div style='margin-top:14px;font-size:0.88rem;color:#555;line-height:1.5;'>
                    {health_advice(lstm_pred)}
                </div>
            </div>""", unsafe_allow_html=True)

        with r2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=lstm_pred,
                number=dict(font=dict(color=clr, size=40, family="Poppins")),
                gauge=dict(
                    axis=dict(range=[0, 500]),
                    bar=dict(color=clr, thickness=0.3),
                    bgcolor="#f8f9fa",
                    steps=[
                        dict(range=[0,50],   color="#e8f8f2"),
                        dict(range=[50,100], color="#fef9e7"),
                        dict(range=[100,150],color="#fef0e6"),
                        dict(range=[150,200],color="#fdedec"),
                        dict(range=[200,300],color="#f5eef8"),
                        dict(range=[300,500],color="#f9ebea"),
                    ],
                    threshold=dict(line=dict(color=clr, width=3), thickness=0.8, value=lstm_pred),
                ),
            ))
            fig_gauge.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                font=dict(family="Poppins, sans-serif", color="#111111"),
                height=280, margin=dict(l=20,r=20,t=20,b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 13c pollutant breakdown
        st.markdown("#### Pollution Breakdown")
        poll_vals = {
            "PM2.5": live_df["pm2_5"].iloc[-1],
            "PM10":  live_df["pm10"].iloc[-1],
            "NO2":   live_df["no2"].iloc[-1],
            "SO2":   live_df["so2"].iloc[-1],
            "O3":    live_df["o3"].iloc[-1],
            "CO":    live_df["co"].iloc[-1] / 10,
        }
        fig_poll = go.Figure(go.Bar(
            x=list(poll_vals.keys()), y=list(poll_vals.values()),
            marker_color=["#4361ee","#4cc9f0","#7209b7","#f77f00","#e63946","#2ec4b6"],
            text=[f"{v:.1f}" for v in poll_vals.values()], textposition="outside",
        ))
        fig_poll.update_layout(
            title=dict(text="Pollutant Concentrations (CO ÷10 for scale)", font=dict(color="#111111")),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Poppins, sans-serif", size=12, color="#111111"),
            yaxis=dict(gridcolor="#f0f0f0", title=dict(text="µg/m³", font=dict(color="#111111")), tickfont=dict(color="#111111")),
            xaxis=dict(linecolor="#eee", tickfont=dict(color="#111111")),
            height=300, margin=dict(t=40,b=20,l=20,r=20), showlegend=False,
        )
        st.plotly_chart(fig_poll, use_container_width=True)

        dominant = max(poll_vals, key=poll_vals.get)
        insights = {
            "PM2.5": "Fine dust particles — likely from traffic or burning.",
            "PM10":  "Coarse dust — possibly road dust or construction.",
            "NO2":   "Vehicle exhaust is the likely main source.",
            "SO2":   "Could be from factories or power plants nearby.",
            "O3":    "Ground-level ozone — forms in sunlight + pollution.",
            "CO":    "Incomplete burning from vehicles or generators.",
        }
        st.info(f"**Dominant pollutant: {dominant}** — {insights.get(dominant, '')}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 13d 24h historical trend
        st.markdown("#### AQI — Last 24 Hours")
        hours    = [f"-{23-i}h" for i in range(24)]
        aqi_vals = live_df["aqi_index"].values

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hours, y=aqi_vals, mode="lines+markers",
            line=dict(color="#4361ee", width=2.5), marker=dict(size=5, color="#4361ee"),
            fill="tozeroy", fillcolor="rgba(67,97,238,0.08)",
        ))
        fig_hist.add_hline(y=100, line=dict(color="#f39c12", width=1.5, dash="dot"),
                           annotation_text="Moderate", annotation_font=dict(color="#f39c12"))
        fig_hist.add_hline(y=200, line=dict(color="#e74c3c", width=1.5, dash="dot"),
                           annotation_text="Unhealthy", annotation_font=dict(color="#e74c3c"))
        fig_hist.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Poppins, sans-serif", size=12, color="#111111"),
            yaxis=dict(gridcolor="#f0f0f0", title=dict(text="AQI", font=dict(color="#111111")), tickfont=dict(color="#111111")),
            xaxis=dict(linecolor="#eee", tickfont=dict(color="#111111")),
            height=300, margin=dict(t=20,b=20,l=20,r=20), showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Note: Historical variation may appear flat on the free API plan.")

        # Section 13e 24h forecast
        st.markdown("#### 24-Hour AQI Forecast")
        with st.spinner("Generating forecast..."):
            forecast = forecast_next_24_hours(lstm_model, scaler, live_df)

        f_hours  = [f"+{i+1}h" for i in range(24)]
        f_colors = [get_aqi_category(v)[1] for v in forecast]

        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(
            x=f_hours, y=forecast, mode="lines+markers",
            line=dict(color="#7209b7", width=2.5),
            marker=dict(color=f_colors, size=9, line=dict(color="white", width=1.5)),
            fill="tozeroy", fillcolor="rgba(114,9,183,0.07)",
        ))
        fig_fore.add_hline(y=100, line=dict(color="#f39c12", width=1.5, dash="dot"))
        fig_fore.add_hline(y=200, line=dict(color="#e74c3c", width=1.5, dash="dot"))
        fig_fore.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Poppins, sans-serif", size=12, color="#111111"),
            yaxis=dict(gridcolor="#f0f0f0", title=dict(text="AQI", font=dict(color="#111111")), tickfont=dict(color="#111111")),
            xaxis=dict(linecolor="#eee", tickfont=dict(color="#111111")),
            height=300, margin=dict(t=20,b=20,l=20,r=20), showlegend=False,
        )
        st.plotly_chart(fig_fore, use_container_width=True)

        # Section 13f forecast table
        with st.expander("See full forecast table"):
            fcols = st.columns(6)
            for i, (h, v) in enumerate(zip(f_hours, forecast)):
                cat_label, c, bg_c = get_aqi_category(v)
                fcols[i%6].markdown(f"""
                <div style='background:{bg_c};border:1px solid {c}55;border-radius:8px;
                            padding:10px 6px;text-align:center;margin-bottom:8px;'>
                    <div style='font-size:0.7rem;color:#888;'>{h}</div>
                    <div style='font-size:1.1rem;font-weight:700;color:{c};'>{v:.0f}</div>
                    <div style='font-size:0.6rem;color:{c};'>{cat_label}</div>
                </div>""", unsafe_allow_html=True)

        # Section 13g map
        st.markdown("#### Location")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.caption("Make sure your API key is valid and the city name is correct.")

# Footer
st.markdown("---")
st.markdown("### How This Works")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **Data Collection**

    Air quality data collected from OpenWeatherMap across 115 cities in India, grouped by state into 4 pollution tiers. Features include PM2.5, PM10, NO2, SO2, O3, CO and weather variables.
    """)
with col2:
    st.markdown("""
    **Model Training**

    Three models trained: Linear Regression and Random Forest as baselines, plus an LSTM deep learning model that learns from 24-hour sequences of pollution data per city.
    """)
with col3:
    st.markdown("""
    **Live Prediction**

    App fetches last 24 hours of real data, applies tier-based PM2.5 caps for accuracy, and passes it through the LSTM to predict next-hour AQI. Unknown cities use nearest trained city as reference.
    """)

st.markdown("<br><div style='text-align:center;color:#aaa;font-size:0.8rem;'>Data from OpenWeatherMap API</div>", unsafe_allow_html=True)