# ---------------- AQI Forecasting Web App ----------------

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
import requests
import time

def fetch_last_24_hours(lat, lon):
    api_key = st.secrets["OPENWEATHER_API_KEY"]
    end = int(time.time())
    start = end - (24 * 60 * 60)

    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={api_key}"
    
    response = requests.get(url)
    data = response.json()

    values = [item["main"]["aqi"] for item in data["list"]]

    return values[-24:]

# ---------------- Page Config ----------------
st.set_page_config(page_title="AQI Forecast System", layout="centered")

st.title("🌍 AQI Forecasting System")
st.markdown("Predict next-hour AQI using **LSTM, Random Forest, and Linear Regression** models.")

# ---------------- AQI Category Function ----------------
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    lstm_model = load_model("aqi_lstm_model.h5", compile=False)
    lr_model = joblib.load("linear_model.pkl")
    rf_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.save")
    metrics = joblib.load("model_metrics.pkl")   # ← ADD THIS LINE
    return lstm_model, lr_model, rf_model, scaler, metrics

lstm_model, lr_model, rf_model, scaler, metrics = load_models()

# ---------------- Upload Section ----------------
st.subheader("Upload Last 24 Hours Data (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

EXPECTED_COLUMNS = [
    "aqi_index",
    "temp_c",
    "humidity",
    "windspeed_kph",
    "pm2_5",
    "pm10",
    "pressure_mb"
]

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Data:")
    st.dataframe(data.head())

    # Validate rows
    if len(data) != 24:
        st.error("⚠ Please upload exactly 24 rows (24 hours data).")
        st.stop()

    # Validate columns
    if list(data.columns) != EXPECTED_COLUMNS:
        st.error("⚠ CSV columns do not match expected format.")
        st.write("Expected columns:", EXPECTED_COLUMNS)
        st.stop()

    if st.button("🚀 Predict Next Hour AQI"):

        # ---------------- Preprocessing ----------------
        scaled_data = scaler.transform(data)

        lstm_input = np.expand_dims(scaled_data, axis=0)
        flat_input = scaled_data.reshape(1, -1)

        # ---------------- Predictions ----------------
        # LSTM
        lstm_pred_scaled = lstm_model.predict(lstm_input)
        dummy_lstm = np.zeros((1, 7))
        dummy_lstm[:, 0] = lstm_pred_scaled[:, 0]
        lstm_pred = scaler.inverse_transform(dummy_lstm)[0][0]

        # Linear Regression
        lr_pred_scaled = lr_model.predict(flat_input)
        dummy_lr = np.zeros((1, 7))
        dummy_lr[:, 0] = lr_pred_scaled
        lr_pred = scaler.inverse_transform(dummy_lr)[0][0]

        # Random Forest
        rf_pred_scaled = rf_model.predict(flat_input)
        dummy_rf = np.zeros((1, 7))
        dummy_rf[:, 0] = rf_pred_scaled
        rf_pred = scaler.inverse_transform(dummy_rf)[0][0]

        # ---------------- Results Section ----------------
        st.success("✅ Prediction Completed")

        results_df = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest", "LSTM"],
            "Predicted AQI": [
                round(lr_pred, 2),
                round(rf_pred, 2),
                round(lstm_pred, 2)
            ]
        })

        st.subheader("📊 Model Comparison")
        st.dataframe(results_df)

        # ---------------- Model Performance Section ----------------
        st.subheader("📉 Model Performance (Training RMSE)")

        rmse_df = pd.DataFrame({
        "Model": list(metrics.keys()),
        "RMSE": list(metrics.values())
         })

        st.dataframe(rmse_df)

        # Highlight LSTM as primary model
        category = get_aqi_category(lstm_pred)

        st.markdown(f"### 🌫 Final AQI (LSTM): **{round(lstm_pred,2)}**")
        st.info(f"Category: {category}")

        # Confidence band
        margin = 5
        lower = lstm_pred - margin
        upper = lstm_pred + margin
        st.write(f"Prediction Range: {round(lower,2)} - {round(upper,2)}")

        # ---------------- Visualization ----------------
        st.subheader("📈 Last 24 Hours Trend + Forecast")

        plt.figure(figsize=(8,5))

        hours = list(range(24))
        future_hour = 24

        plt.plot(hours, data["aqi_index"].values, marker='o', label="Past 24 Hours")

        plt.plot(
            [23, future_hour],
            [data["aqi_index"].values[-1], lstm_pred],
            linestyle='--',
            label="Forecast"
        )

        plt.scatter(future_hour, lstm_pred)
        plt.vlines(future_hour, lower, upper)
        plt.text(future_hour, lstm_pred, f"{round(lstm_pred,2)}")

        plt.xlabel("Hour")
        plt.ylabel("AQI")
        plt.xticks(range(0, 25))
        plt.legend()

        st.pyplot(plt)

        # ---------------- Bar Chart Comparison ----------------
        st.subheader("📊 Model Prediction Comparison")

        fig, ax = plt.subplots()
        ax.bar(results_df["Model"], results_df["Predicted AQI"])
        ax.set_ylabel("Predicted AQI")
        ax.set_title("Model Comparison")
        plt.xticks(rotation=20)

        st.pyplot(fig)