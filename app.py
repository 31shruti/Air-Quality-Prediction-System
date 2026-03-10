# ---------------- AQI Forecasting Web App ----------------

# Section 1 Imported important libraries 
import streamlit as st  # type: ignore
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
import requests
import time
import plotly.graph_objects as go  # type: ignore

# Section 2 Configuration of page and title 
st.set_page_config(
    page_title="AQI Forecast App",
    layout="wide"
)

# Section 3 used custom css for app ; used poppins font and white card 
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

.stApp {
    background-color: #f0f4f8;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 8px;
}

.big-aqi {
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
}

.label {
    font-size: 0.75rem;
    color: #888;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #111111;
}

.unit {
    font-size: 0.8rem;
    color: #aaa;
}

.badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 10px;
}

.stButton > button {
    background-color: #4361ee;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    transition: background 0.2s;
}

.stButton > button:hover {
    background-color: #3a0ca3;
}

.stNumberInput input {
    border-radius: 8px;
    border: 1.5px solid #dee2e6;
    font-family: 'Poppins', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e9ecef;
    color: #111111 !important;
}

[data-testid="stSidebar"] * {
    color: #111111 !important;
}

/* fix for dropdown selectbox - options were dark text on dark background */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #111111 !important;
}

div[data-baseweb="select"] span {
    color: #111111 !important;
}

div[data-baseweb="popover"] * {
    background-color: #ffffff !important;
    color: #111111 !important;
}

div[data-baseweb="menu"] {
    background-color: #ffffff !important;
}

div[data-baseweb="menu"] li {
    color: #111111 !important;
    background-color: #ffffff !important;
}

div[data-baseweb="menu"] li:hover {
    background-color: #f0f4f8 !important;
    color: #111111 !important;
}
</style>
""", unsafe_allow_html=True)

# Section 4 used helper function for aqi category and health advice
# these functions took an aqi number and return color, label, and advice
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

# Section 5 aqi formula and city PM2.5 caps
# these are same formulas as used in train.py so predictions can be consistent
def calculate_aqi(pm25):
    if pm25 <= 30:  return (50 / 30) * pm25
    if pm25 <= 60:  return ((100-51)  / (60-31))  * (pm25-31)  + 51
    if pm25 <= 90:  return ((200-101) / (90-61))  * (pm25-61)  + 101
    if pm25 <= 120: return ((300-201) / (120-91)) * (pm25-91)  + 201
    if pm25 <= 250: return ((400-301) / (250-121))* (pm25-121) + 301
    return 500

# per city caps to stop api spike value from giving wrong aqi
# London changed from 30 to 60 — old value was too aggressive and caused it to show AQI of 3
CITY_PM25_CAPS = {
    "Shimla": 35, "Manali": 30, "Leh": 25, "Gangtok": 30, "Aizawl": 30,
    "London": 60, "Tokyo": 35, "New York": 25, "Paris": 30, "Sydney": 20,
    "Bangalore": 60, "Chennai": 65, "Kochi": 50, "Pune": 70, "Hyderabad": 70,
}

EXPECTED_FEATURES = [
    "pm2_5","pm10","no2","so2","o3","co",
    "temp_c","humidity","windspeed_kph","pressure_mb",
]

# Section 5a converted city name to Lat Lon using geocoding API
# added this so there would be no need to type Lon and Lat manually
def get_coordinates(city_name):
    api_key = st.secrets["OPENWEATHER_API_KEY"]
    r = requests.get(
        f"https://api.openweathermap.org/geo/1.0/direct"
        f"?q={city_name}&limit=1&appid={api_key}",
        timeout=10
    )
    data = r.json()
    if not data:
        raise ValueError(f"Couldnt find '{city_name}'. Please check spelling.")
    lat  = data[0]["lat"]
    lon  = data[0]["lon"]
    name = data[0].get("name", city_name)
    return lat, lon, name


# Section 6 loaded all trained models and scaler
# used cache_resources so models load only once and stay in memory
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

# helper to convert scaled predictions back to real aqi value
def inverse_aqi(scaled_values):
    dummy = np.zeros((len(scaled_values), N_COLS))
    dummy[:, AQI_COL] = scaled_values
    return scaler.inverse_transform(dummy)[:, AQI_COL]

# Section 7 this is side-bar with about info and aqi scale legend
with st.sidebar:
    st.markdown("## About This App")
    st.markdown("""
    This app predicts Air Quality Index (AQI) using machine learning models trained on real pollution data fetched from api.

    **Models used:**
    - LSTM (Deep Learning)
    - Random Forest
    - Linear Regression

    **How it works:**
    1. Select a city from the dropdown or type manually
    2. App fetches last 24 hours of air quality data
    3. LSTM model predicts next hour AQI
    4. You also get a 24-hour forecast

    ---
    **AQI Scale:**
    """)

    scale = [
        ("0–50",   "Good",           "#2ecc71"),
        ("51–100", "Moderate",       "#f39c12"),
        ("101–150","Sensitive Groups","#e67e22"),
        ("151–200","Unhealthy",      "#e74c3c"),
        ("201–300","Very Unhealthy", "#8e44ad"),
        ("301+",   "Hazardous",      "#7b241c"),
    ]
    for rng, label, color in scale:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:10px;padding:4px 0;font-size:0.82rem;'>
            <div style='width:12px;height:12px;border-radius:3px;background:{color};'></div>
            <span style='color:#555;'>{rng}</span>
            <span style='margin-left:auto;color:{color};font-weight:600;'>{label}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem;color:#aaa;'>Data: OpenWeatherMap API</div>", unsafe_allow_html=True)

# Section 8 set app header and title
st.markdown("# AQI Forecast Dashboard")
st.markdown("<p style='color:#666;margin-top:-12px;'>Predicting air quality using LSTM deep learning + real-time data</p>", unsafe_allow_html=True)
st.markdown("---")

# Section 9 added model performance cards and RMSE chart
st.markdown("### How well do the models perform?")

metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})

col1, col2, col3, col4 = st.columns(4)
icons  = {"Persistence": " ","Linear Regression": "1", "Random Forest": "2", "LSTM": "3"}
colors = {"Persistence": "#aaaaaa", "Linear Regression": "#4cc9f0", "Random Forest": "#4361ee", "LSTM": "#7209b7"}

for col, (_, row) in zip([col1, col2, col3, col4], metrics_df.iterrows()):
    clr  = colors.get(row["Model"], "#4361ee")
    icon = icons.get(row["Model"], "")
    col.markdown(f"""
    <div class='card' style='border-top: 4px solid {clr};'>
        <div style='font-size:1rem;font-weight:600;color:#1a1a2e;margin-bottom:12px;'>{icon} {row["Model"]}</div>
        <div style='display:flex;gap:20px;'>
            <div>
                <div class='label'>MAE</div>
                <div style='font-size:1.3rem;font-weight:600;color:{clr};'>{row["MAE"]:.2f}</div>
            </div>
            <div>
                <div class='label'>RMSE</div>
                <div style='font-size:1.3rem;font-weight:600;color:{clr};'>{row["RMSE"]:.2f}</div>
            </div>
            <div>
                <div class='label'>R²</div>
                <div style='font-size:1.3rem;font-weight:600;color:{clr};'>{row["R2"]:.3f}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

# used RMSE bar-chart to visually compare all models
fig_bar = go.Figure(go.Bar(
    x=metrics_df["Model"],
    y=metrics_df["RMSE"],
    marker_color=[colors.get(m, "#4361ee") for m in metrics_df["Model"]],
    text=[f"{v:.2f}" for v in metrics_df["RMSE"]],
    textposition="outside",
))
fig_bar.update_layout(
    title=dict(text="RMSE Comparison (lower = better)", font=dict(color="#111111")),
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Poppins, sans-serif", size=12, color="#111111"),
    yaxis=dict(gridcolor="#f0f0f0", tickfont=dict(color="#111111")),
    xaxis=dict(linecolor="#eee", tickfont=dict(color="#111111")),
    height=300, margin=dict(t=40,b=20,l=20,r=20),
    showlegend=False,
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# Section 10 City dropdown + manual type option
# dropdown has all 115 trained cities, user can also type manually if city not in list
st.markdown("### Get Live AQI Prediction")
st.markdown("<p style='color:#666;font-size:0.9rem;'>Select a city from the dropdown or type manually</p>", unsafe_allow_html=True)

CITY_LIST = [
    "-- Type a city manually --",
    # North India
    "Delhi", "Chandigarh", "Jaipur", "Lucknow", "Kanpur", "Agra",
    "Varanasi", "Amritsar", "Meerut", "Ghaziabad", "Allahabad",
    "Bareilly", "Moradabad", "Gorakhpur", "Mathura",
    # Punjab & Haryana
    "Ludhiana", "Jalandhar", "Patiala", "Gurugram", "Faridabad",
    "Ambala", "Hisar",
    # Rajasthan
    "Jodhpur", "Udaipur", "Kota", "Ajmer", "Bikaner",
    # Uttarakhand
    "Dehradun", "Haridwar", "Roorkee",
    # Himachal Pradesh
    "Shimla", "Manali", "Dharamshala", "Solan", "Mandi",
    # Jammu & Kashmir & Ladakh
    "Srinagar", "Jammu", "Anantnag", "Baramulla", "Leh",
    # Gujarat
    "Ahmedabad", "Surat", "Rajkot", "Vadodara", "Bhavnagar",
    "Gandhinagar", "Jamnagar",
    # Maharashtra
    "Mumbai", "Pune", "Nashik", "Aurangabad", "Nagpur",
    "Solapur", "Kolhapur", "Thane",
    # Goa
    "Panaji", "Margao",
    # Karnataka
    "Bangalore", "Mysore", "Mangalore", "Hubli", "Belgaum",
    # Tamil Nadu
    "Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli",
    # Andhra Pradesh
    "Visakhapatnam", "Vijayawada", "Tirupati", "Guntur", "Nellore",
    # Telangana
    "Hyderabad", "Warangal", "Nizamabad",
    # Kerala
    "Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur", "Kollam",
    # West Bengal
    "Kolkata", "Howrah", "Siliguri", "Durgapur", "Asansol",
    # Odisha
    "Bhubaneswar", "Cuttack", "Rourkela", "Sambalpur",
    # Bihar
    "Patna", "Gaya", "Muzaffarpur", "Bhagalpur",
    # Jharkhand
    "Ranchi", "Jamshedpur", "Dhanbad", "Bokaro",
    # Central India
    "Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain", "Raipur",
    # Northeast India
    "Guwahati", "Shillong", "Imphal", "Agartala", "Dibrugarh", "Silchar",
    # Hill stations
    "Gangtok", "Aizawl",
]

c1, c2 = st.columns([2, 1])
with c1:
    selected = st.selectbox("Select City", CITY_LIST)
    # show text box only if user picks the manual option
    if selected == "-- Type a city manually --":
        city_input = st.text_input(
            "Type city name",
            placeholder="e.g. Kishtwar, Hapur, Noida ..."
        )
    else:
        city_input = selected

with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Predict AQI", use_container_width=True)


# Section 11 used function to fetch last 24 hours of pollution and weather data 
# here the OpenWeatherMAp api has been called and returned a clean dataframe with 24 rows 
def fetch_last_24_hours(lat, lon):
    api_key = st.secrets["OPENWEATHER_API_KEY"]
    end = int(time.time()); start = end - 24*3600

    r = requests.get(
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start}&end={end}&appid={api_key}",
        timeout=30)
    r.raise_for_status()
    data = r.json()

    if "list" not in data or not data["list"]:
        raise ValueError("No pollution data from API. Check your coordinates.")

    pollution_list = data["list"]
    while len(pollution_list) < 24:
        pollution_list = pollution_list * 2
    pollution_list = pollution_list[-24:]

    # fetched current weather for temperature, humidity, wind and pressure
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
            "pm2_5": pm25,          "pm10": item["components"]["pm10"],
            "no2":   item["components"]["no2"], "so2": item["components"]["so2"],
            "o3":    item["components"]["o3"],  "co":  item["components"]["co"],
            "temp_c": temp, "humidity": humidity,
            "windspeed_kph": windspeed, "pressure_mb": pressure,
            "aqi_index": calculate_aqi(pm25),
        })

    live_df = pd.DataFrame(rows)[EXPECTED_FEATURES + ["aqi_index"]]

    # using partial match so "New Delhi" correctly matches "Delhi" in the dict
    # old code used exact match so caps were never applied for many cities
    pm25_cap = 150
    for cap_city, cap_val in CITY_PM25_CAPS.items():
        if cap_city.lower() in city.lower() or city.lower() in cap_city.lower():
            pm25_cap = cap_val
            break

    # hard global cap — no Indian city realistically has PM2.5 above 250
    # this stops small towns like Kishtwar from getting garbage API values
    live_df["pm2_5"] = live_df["pm2_5"].clip(upper=min(pm25_cap, 250))
    live_df["aqi_index"] = live_df["pm2_5"].apply(calculate_aqi)

    return live_df, city, {"temp": temp, "humidity": humidity, "windspeed": windspeed, "pressure": pressure}

# Section 12 function to generate 24 hrs rolling forecast
# feeded lstm prediction back into the window step by step for 24 hrs
def forecast_next_24_hours(model, scaler, live_df):
    predictions = []; window = live_df.copy()
    for _ in range(24):
        scaled  = scaler.transform(window)
        pred_s  = model.predict(np.expand_dims(scaled, axis=0), verbose=0)[0, 0]
        pred_r  = float(np.clip(inverse_aqi(np.array([pred_s]))[0], 0, 500))
        predictions.append(pred_r)
        new_row = window.iloc[-1].copy(); new_row["aqi_index"] = pred_r
        window  = pd.concat([window.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
    return predictions

# Section 13 run prediction when button is clicked
if run:
    # first checks if user actually typed something or selected manual but left it empty
    if not city_input.strip():
        st.warning("Please select a city or type a city name first.")
        st.stop()
    try:
        # step 1 - get coordinates from city name
        with st.spinner(f"Searching for {city_input}..."):
            lat, lon, resolved_city = get_coordinates(city_input.strip())

        st.caption(f"Found: {resolved_city} ({lat:.4f}, {lon:.4f})")

        # step 2 - fetch pollution data for those coordinates
        with st.spinner("Fetching air quality data..."):
            live_df, city, wx = fetch_last_24_hours(lat, lon)

        st.markdown(f"### Results for **{city}**")

        # Section 13a showing current weather cards
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

        # Section 13b run lstm model and show predicted aqi
        scaled_data = scaler.transform(live_df)
        pred_scaled = lstm_model.predict(np.expand_dims(scaled_data, axis=0), verbose=0)[0, 0]
        lstm_pred   = float(np.clip(inverse_aqi(np.array([pred_scaled]))[0], 0, 500))

        # check if searched city is in our trained city list
        # if not, show a warning so user knows prediction is an estimate
        TRAINED_CITIES = [
            "Delhi", "Chandigarh", "Jaipur", "Lucknow", "Kanpur", "Agra",
            "Varanasi", "Amritsar", "Meerut", "Ghaziabad", "Allahabad",
            "Bareilly", "Moradabad", "Gorakhpur", "Mathura", "Ludhiana",
            "Jalandhar", "Patiala", "Gurugram", "Faridabad", "Ambala", "Hisar",
            "Jodhpur", "Udaipur", "Kota", "Ajmer", "Bikaner", "Dehradun",
            "Haridwar", "Roorkee", "Shimla", "Manali", "Dharamshala", "Solan",
            "Mandi", "Srinagar", "Jammu", "Anantnag", "Baramulla", "Leh",
            "Ahmedabad", "Surat", "Rajkot", "Vadodara", "Bhavnagar",
            "Gandhinagar", "Jamnagar", "Mumbai", "Pune", "Nashik", "Aurangabad",
            "Nagpur", "Solapur", "Kolhapur", "Thane", "Panaji", "Margao",
            "Bangalore", "Mysore", "Mangalore", "Hubli", "Belgaum", "Chennai",
            "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli",
            "Visakhapatnam", "Vijayawada", "Tirupati", "Guntur", "Nellore",
            "Hyderabad", "Warangal", "Nizamabad", "Kochi", "Thiruvananthapuram",
            "Kozhikode", "Thrissur", "Kollam", "Kolkata", "Howrah", "Siliguri",
            "Durgapur", "Asansol", "Bhubaneswar", "Cuttack", "Rourkela",
            "Sambalpur", "Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Ranchi",
            "Jamshedpur", "Dhanbad", "Bokaro", "Bhopal", "Indore", "Gwalior",
            "Jabalpur", "Ujjain", "Raipur", "Guwahati", "Shillong", "Imphal",
            "Agartala", "Dibrugarh", "Silchar", "Gangtok", "Aizawl",
        ]
        # check if city is in trained list using partial match
        is_trained = any(
            t.lower() in city.lower() or city.lower() in t.lower()
            for t in TRAINED_CITIES
        )
        if not is_trained:
            st.warning(
                f"⚠️ **{city}** is not in the training data. "
                f"Prediction is an estimate based on similar pollution patterns. "
                f"Accuracy may be lower than for trained cities."
            )

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
                        dict(range=[0,   50],  color="#e8f8f2"),
                        dict(range=[50,  100], color="#fef9e7"),
                        dict(range=[100, 150], color="#fef0e6"),
                        dict(range=[150, 200], color="#fdedec"),
                        dict(range=[200, 300], color="#f5eef8"),
                        dict(range=[300, 500], color="#f9ebea"),
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

        # Section 13c showing pollutant breakdown bar chart
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
            x=list(poll_vals.keys()),
            y=list(poll_vals.values()),
            marker_color=["#4361ee","#4cc9f0","#7209b7","#f77f00","#e63946","#2ec4b6"],
            text=[f"{v:.1f}" for v in poll_vals.values()],
            textposition="outside",
        ))
        fig_poll.update_layout(
            title=dict(text="Pollutant Concentrations (CO ÷10 for scale)", font=dict(color="#111111")),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Poppins, sans-serif", size=12, color="#111111"),
            yaxis=dict(gridcolor="#f0f0f0", title=dict(text="µg/m³", font=dict(color="#111111")), tickfont=dict(color="#111111")),
            xaxis=dict(linecolor="#eee", tickfont=dict(color="#111111")),
            height=300, margin=dict(t=40,b=20,l=20,r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_poll, use_container_width=True)

        # Dominant pollutant info
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

        # Section 13d showing last 24 hrs historical aqi trend
        st.markdown("#### AQI — Last 24 Hours")
        hours    = [f"-{23-i}h" for i in range(24)]
        aqi_vals = live_df["aqi_index"].values

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hours, y=aqi_vals,
            mode="lines+markers",
            line=dict(color="#4361ee", width=2.5),
            marker=dict(size=5, color="#4361ee"),
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
            height=300, margin=dict(t=20,b=20,l=20,r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Note: Historical variation may appear flat on the free API plan. A paid OpenWeatherMap plan provides real 24-hour historical data.")

        # Section 13e generated and showed 24 hour forecast chart
        st.markdown("#### 24-Hour AQI Forecast")
        st.caption("Predicted using LSTM model with rolling 24-hour window")

        with st.spinner("Generating forecast..."):
            forecast = forecast_next_24_hours(lstm_model, scaler, live_df)

        f_hours  = [f"+{i+1}h" for i in range(24)]
        f_colors = [get_aqi_category(v)[1] for v in forecast]

        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(
            x=f_hours, y=forecast,
            mode="lines+markers",
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
            height=300, margin=dict(t=20,b=20,l=20,r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_fore, use_container_width=True)

        # Section 13f for expandable forecast table with all 24 hrs
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

        # Section 13g showed location on map
        st.markdown("#### Location")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.caption("Make sure your API key is valid and the city name is correct.")

# this is footer section showing how this works explanation
st.markdown("---")
st.markdown("### How This Works")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **Data Collection**
    
    Air quality data collected from OpenWeatherMap across 115 cities across India. Features include PM2.5, PM10, NO2, SO2, O3, CO and weather variables.
    """)
with col2:
    st.markdown("""
    **Model Training**
    
    Three models trained: Linear Regression and Random Forest as baselines, plus an LSTM deep learning model that learns from 24-hour sequences of pollution data.
    """)
with col3:
    st.markdown("""
    **Live Prediction**
    
    App fetches last 24 hours of real data for your location, scales it using the same scaler from training, and passes it through the LSTM to predict next-hour AQI.
    """)

st.markdown("<br><div style='text-align:center;color:#aaa;font-size:0.8rem;'> Data from OpenWeatherMap API</div>", unsafe_allow_html=True)