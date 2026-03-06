# ---------------- AQI Forecasting Web App ----------------

import streamlit as st  # type: ignore
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
import requests
import time
import plotly.graph_objects as go  # type: ignore

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Forecast App",
    page_icon="🌿",
    layout="wide"
)

# ── Simple clean CSS ──────────────────────────────────────────────────────
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
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────
def get_aqi_category(aqi):
    if aqi <= 50:  return "Good",            "#2ecc71", "#e8f8f2"
    if aqi <= 100: return "Moderate",         "#f39c12", "#fef9e7"
    if aqi <= 150: return "Sensitive Groups", "#e67e22", "#fef0e6"
    if aqi <= 200: return "Unhealthy",        "#e74c3c", "#fdedec"
    if aqi <= 300: return "Very Unhealthy",   "#8e44ad", "#f5eef8"
    return               "Hazardous",         "#7b241c", "#f9ebea"

def health_advice(aqi):
    if aqi <= 50:  return "😊 Air is clean! Great day for outdoor activities."
    if aqi <= 100: return "😐 Air is okay. Sensitive people should be cautious."
    if aqi <= 150: return "😷 Sensitive groups should reduce outdoor time."
    if aqi <= 200: return "⚠️ Unhealthy for everyone. Limit outdoor activities."
    if aqi <= 300: return "🚨 Very unhealthy. Stay indoors if possible."
    return               "☠️ Hazardous! Avoid going outside completely."

def calculate_aqi(pm25):
    if pm25 <= 30:  return (50 / 30) * pm25
    if pm25 <= 60:  return ((100-51)  / (60-31))  * (pm25-31)  + 51
    if pm25 <= 90:  return ((200-101) / (90-61))  * (pm25-61)  + 101
    if pm25 <= 120: return ((300-201) / (120-91)) * (pm25-91)  + 201
    if pm25 <= 250: return ((400-301) / (250-121))* (pm25-121) + 301
    return 500

CITY_PM25_CAPS = {
    "Shimla": 35, "Manali": 30, "Leh": 25, "Gangtok": 30, "Aizawl": 30,
    "London": 30, "Tokyo": 35, "New York": 25, "Paris": 30, "Sydney": 20,
    "Bangalore": 60, "Chennai": 65, "Kochi": 50, "Pune": 70, "Hyderabad": 70,
}

EXPECTED_FEATURES = [
    "pm2_5","pm10","no2","so2","o3","co",
    "temp_c","humidity","windspeed_kph","pressure_mb",
]

# ── Load Models ───────────────────────────────────────────────────────────
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

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 About This App")
    st.markdown("""
    This app predicts Air Quality Index (AQI) using machine learning models trained on real pollution data.

    **Models used:**
    - 🧠 LSTM (Deep Learning)
    - 🌳 Random Forest
    - 📈 Linear Regression

    **How it works:**
    1. Enter your location coordinates
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
    st.markdown("<div style='font-size:0.75rem;color:#aaa;'>Made by a CS student 👨‍💻<br>Data: OpenWeatherMap API</div>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("# 🌍 AQI Forecast Dashboard")
st.markdown("<p style='color:#666;margin-top:-12px;'>Predicting air quality using LSTM deep learning + real-time data</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Model Performance ─────────────────────────────────────────────────────
st.markdown("### 📊 How well do the models perform?")

metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})

col1, col2, col3 = st.columns(3)
icons = {"Linear Regression": "📈", "Random Forest": "🌳", "LSTM": "🧠"}
colors = {"Linear Regression": "#4cc9f0", "Random Forest": "#4361ee", "LSTM": "#7209b7"}

for col, (_, row) in zip([col1, col2, col3], metrics_df.iterrows()):
    clr  = colors.get(row["Model"], "#4361ee")
    icon = icons.get(row["Model"], "🤖")
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

# Simple RMSE bar chart
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

# ── Live Prediction ───────────────────────────────────────────────────────
st.markdown("### 🔍 Get Live AQI Prediction")
st.markdown("<p style='color:#666;font-size:0.9rem;'>Enter coordinates of any city to fetch real-time data and predict AQI</p>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 1])
with c1: lat = st.number_input("Latitude",  value=28.6139, format="%.4f", help="e.g. 28.6139 for Delhi")
with c2: lon = st.number_input("Longitude", value=77.2090, format="%.4f", help="e.g. 77.2090 for Delhi")
with c3:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔍 Predict AQI", use_container_width=True)

# ── Fetch & predict ───────────────────────────────────────────────────────
def fetch_last_24_hours(lat, lon):
    api_key = st.secrets["OPENWEATHER_API_KEY"]

    r = requests.get(
      f"https://api.openweathermap.org/data/2.5/air_pollution"
      f"?lat={lat}&lon={lon}&appid={api_key}",
     timeout=30)
    r.raise_for_status()
    data = r.json()

    if "list" not in data or not data["list"]:
        raise ValueError("No pollution data from API. Check your coordinates.")

    pollution_list = data["list"] * 24

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
    pm25_cap = CITY_PM25_CAPS.get(city, 150)
    live_df["pm2_5"] = live_df["pm2_5"].clip(upper=pm25_cap)
    live_df["aqi_index"] = live_df["pm2_5"].apply(calculate_aqi)
    return live_df, city, {"temp": temp, "humidity": humidity, "windspeed": windspeed, "pressure": pressure}

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

if run:
    try:
        with st.spinner("Fetching data from OpenWeatherMap..."):
            live_df, city, wx = fetch_last_24_hours(lat, lon)

        # City name
        st.markdown(f"### 📍 Results for **{city}**")

        # Weather strip
        st.markdown("#### 🌤 Current Weather")
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

        # Predict
        scaled_data = scaler.transform(live_df)
        pred_scaled = lstm_model.predict(np.expand_dims(scaled_data, axis=0), verbose=0)[0, 0]
        lstm_pred   = float(np.clip(inverse_aqi(np.array([pred_scaled]))[0], 0, 500))

        cat, clr, bg = get_aqi_category(lstm_pred)

        # AQI result
        st.markdown("#### 🎯 Predicted AQI (Next Hour)")
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

        # Pollution breakdown
        st.markdown("#### 🧪 Pollution Breakdown")
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

        # Dominant pollutant
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

        # Historical trend
        st.markdown("#### 📈 AQI — Last 24 Hours")
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

        # 24h forecast
        st.markdown("#### 🔮 24-Hour AQI Forecast")
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

        # Forecast table
        with st.expander("📋 See full forecast table"):
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

        # Map
        st.markdown("#### 🗺 Location")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.caption("Make sure your coordinates are correct and your API key is valid.")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 How This Works")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **📥 Data Collection**
    
    Air quality data collected from OpenWeatherMap across 40+ cities in India and worldwide. Features include PM2.5, PM10, NO2, SO2, O3, CO and weather variables.
    """)
with col2:
    st.markdown("""
    **🧠 Model Training**
    
    Three models trained: Linear Regression and Random Forest as baselines, plus an LSTM deep learning model that learns from 24-hour sequences of pollution data.
    """)
with col3:
    st.markdown("""
    **📡 Live Prediction**
    
    App fetches last 24 hours of real data for your location, scales it using the same scaler from training, and passes it through the LSTM to predict next-hour AQI.
    """)

st.markdown("<br><div style='text-align:center;color:#aaa;font-size:0.8rem;'>Built as a final year ML project · Data from OpenWeatherMap API</div>", unsafe_allow_html=True)