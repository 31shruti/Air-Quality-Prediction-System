# ---------------- AQI Forecasting Web App ----------------

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
import requests
import time
import plotly.graph_objects as go # type: ignore

def fetch_last_24_hours_full(lat, lon):
    api_key = st.secrets["OPENWEATHER_API_KEY"]
    end = int(time.time())
    start = end - (24 * 60 * 60)

    # Air pollution history
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={api_key}"
    
    pollution_response = requests.get(pollution_url).json()

    # Current weather (for temperature, humidity etc.)
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    weather_response = requests.get(weather_url).json()
    city = weather_response["name"]

    pollution_list = pollution_response["list"]
    if len(pollution_list) < 24:
        pollution_list = pollution_list * (24 // len(pollution_list) + 1)

    pollution_list = pollution_list[-24:]

    aqi = [item["main"]["aqi"] for item in pollution_list][-24:]
    pm25 = [item["components"]["pm2_5"] for item in pollution_list][-24:]
    pm10 = [item["components"]["pm10"] for item in pollution_list][-24:]
    no2 = [item["components"]["no2"] for item in pollution_list][-24:]
    co = [item["components"]["co"] for item in pollution_list][-24:]
    o3 = [item["components"]["o3"] for item in pollution_list][-24:]
    so2 = [item["components"]["so2"] for item in pollution_list][-24:]

    temp = weather_response["main"]["temp"]
    humidity = weather_response["main"]["humidity"]
    pressure = weather_response["main"]["pressure"]
    windspeed = weather_response["wind"]["speed"] * 3.6

    # Repeat weather values for 24 rows
    live_df = pd.DataFrame({
        "aqi_index": aqi,
        "temp_c": [temp]*24,
        "humidity": [humidity]*24,
        "windspeed_kph": [windspeed]*24,
        "pm2_5": pm25,
        "pm10": pm10,
        "pressure_mb": [pressure]*24
     })
    
    extra_pollutants = {
        "NO2": no2[-1],
        "CO": co[-1],
        "O3": o3[-1],
        "SO2": so2[-1]
    }

    return live_df, city, extra_pollutants

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AI AQI Forecast Dashboard",
    layout="wide"
)

st.sidebar.title("Project Info")

st.sidebar.info("""
AI-Based Air Quality Forecast System

Models Used:
• LSTM Deep Learning  
• Random Forest  
• Linear Regression  

Features:
• CSV-based prediction  
• Real-time AQI forecast  
• Model comparison  
• Trend visualization
""")

st.markdown("""
# 🌍 AI-Based AQI Forecasting Dashboard
### Real-Time Environmental Prediction using Machine Learning
""")

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
    
# ---------------- Multi-step Forecast ----------------

def forecast_next_24_hours(model, scaler, data):

    predictions = []
    temp_data = data.copy()

    for i in range(24):

        scaled = scaler.transform(temp_data)
        lstm_input = np.expand_dims(scaled, axis=0)

        pred_scaled = model.predict(lstm_input)

        dummy = np.zeros((1,7))
        dummy[:,0] = pred_scaled[:,0]

        pred = scaler.inverse_transform(dummy)[0][0]

        predictions.append(pred)

        # slide window
        new_row = temp_data.iloc[-1].copy()
        new_row["aqi_index"] = pred

        temp_data = pd.concat(
            [temp_data.iloc[1:], pd.DataFrame([new_row])],
            ignore_index=True
        )

    return predictions       

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

# # ---------------- Upload Section ----------------
# st.subheader("Upload Last 24 Hours Data (CSV)")

# uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# EXPECTED_COLUMNS = [
#     "aqi_index",
#     "temp_c",
#     "humidity",
#     "windspeed_kph",
#     "pm2_5",
#     "pm10",
#     "pressure_mb"
# ]

# if uploaded_file is not None:

#     data = pd.read_csv(uploaded_file)

#     st.write("Preview of Uploaded Data:")
#     st.dataframe(data.head())

#     # Validate rows
#     if len(data) != 24:
#         st.error("⚠ Please upload exactly 24 rows (24 hours data).")
#         st.stop()

#     # Validate columns
#     if list(data.columns) != EXPECTED_COLUMNS:
#         st.error("⚠ CSV columns do not match expected format.")
#         st.write("Expected columns:", EXPECTED_COLUMNS)
#         st.stop()

#     if st.button("Predict Next Hour AQI"):

#         # ---------------- Preprocessing ----------------
#         scaled_data = scaler.transform(data)

#         lstm_input = np.expand_dims(scaled_data, axis=0)
#         flat_input = scaled_data.reshape(1, -1)

#         # ---------------- Predictions ----------------
#         # LSTM
#         lstm_pred_scaled = lstm_model.predict(lstm_input)
#         dummy_lstm = np.zeros((1, 7))
#         dummy_lstm[:, 0] = lstm_pred_scaled[:, 0]
#         lstm_pred = scaler.inverse_transform(dummy_lstm)[0][0]

#         # Linear Regression
#         lr_pred_scaled = lr_model.predict(flat_input)
#         dummy_lr = np.zeros((1, 7))
#         dummy_lr[:, 0] = lr_pred_scaled
#         lr_pred = scaler.inverse_transform(dummy_lr)[0][0]

#         # Random Forest
#         rf_pred_scaled = rf_model.predict(flat_input)
#         dummy_rf = np.zeros((1, 7))
#         dummy_rf[:, 0] = rf_pred_scaled
#         rf_pred = scaler.inverse_transform(dummy_rf)[0][0]

#         # ---------------- Results Section ----------------
#         st.success("✅ Prediction Completed")

#     if st.button("📅 Forecast Next 24 Hours"):

#         future_predictions = forecast_next_24_hours(
#             lstm_model,
#             scaler,
#             data
#         )

#         forecast_df = pd.DataFrame({
#             "Hour Ahead": list(range(1,25)),
#             "Predicted AQI": [round(x,2) for x in future_predictions]
#         })

#         st.subheader("24 Hour AQI Forecast")
#         st.dataframe(forecast_df)

#         results_df = pd.DataFrame({
#             "Model": ["Linear Regression", "Random Forest", "LSTM"],
#             "Predicted AQI": [
#                 round(lr_pred, 2),
#                 round(rf_pred, 2),
#                 round(lstm_pred, 2)
#             ]
#         })

#         # ---------------- Model Agreement / Uncertainty ----------------

#         predictions = [lr_pred, rf_pred, lstm_pred]

#         std_dev = np.std(predictions)

#         st.subheader("Prediction Confidence")

#         if std_dev < 10:
#             st.success("High Model Agreement (High Confidence)")
#         elif std_dev < 30:
#             st.warning("Moderate Agreement")
#         else:
#             st.error("Low Agreement (Higher Uncertainty)")

#         st.write(f"Prediction Standard Deviation: {round(std_dev,2)}")

#         st.subheader("📊 Model Comparison")
#         st.dataframe(results_df)

#         # ---------------- Model Performance Section ----------------
#         st.subheader("📉 Model Performance (Training RMSE)")

#         rmse_df = pd.DataFrame({
#         "Model": list(metrics.keys()),
#         "RMSE": list(metrics.values())
#          })

#         st.dataframe(rmse_df)

#         # Highlight LSTM as primary model
#         category = get_aqi_category(lstm_pred)

#         st.markdown(f"### 🌫 Final AQI (LSTM): **{round(lstm_pred,2)}**")
#         st.info(f"Category: {category}")

#         # Confidence band
#         margin = 5
#         lower = lstm_pred - margin
#         upper = lstm_pred + margin
#         st.write(f"Prediction Range: {round(lower,2)} - {round(upper,2)}")

#         # ---------------- Visualization ----------------
#         st.subheader("📈 Last 24 Hours Trend + Forecast")

#         plt.figure(figsize=(8,5))

#         hours = list(range(24))
#         future_hour = 24

#         plt.plot(hours, data["aqi_index"].values, marker='o', label="Past 24 Hours")

#         plt.plot(
#             [23, future_hour],
#             [data["aqi_index"].values[-1], lstm_pred],
#             linestyle='--',
#             label="Forecast"
#         )

#         plt.scatter(future_hour, lstm_pred)
#         plt.vlines(future_hour, lower, upper)
#         plt.text(future_hour, lstm_pred, f"{round(lstm_pred,2)}")

#         plt.xlabel("Hour")
#         plt.ylabel("AQI")
#         plt.xticks(range(0, 25))
#         plt.legend()

#         st.pyplot(plt)

#         # ---------------- Bar Chart Comparison ----------------
#         st.subheader("📊 Model Prediction Comparison")

#         fig, ax = plt.subplots()
#         ax.bar(results_df["Model"], results_df["Predicted AQI"])
#         ax.set_ylabel("Predicted AQI")
#         ax.set_title("Model Comparison")
#         plt.xticks(rotation=20)

#         st.pyplot(fig)
 
def health_advice(aqi):

    if aqi <= 50:
        return "Air quality is good. Enjoy outdoor activities."

    elif aqi <= 100:
        return "Air quality is moderate. Sensitive people should reduce prolonged outdoor exertion."

    elif aqi <= 200:
        return "Unhealthy for sensitive groups. Children and elderly should limit outdoor exposure."

    elif aqi <= 300:
        return "Unhealthy. Everyone should reduce prolonged outdoor activities."

    else:
        return "Very unhealthy. Avoid outdoor activities and wear protective masks." 

# ---------------- Real-Time Mode ----------------
st.markdown("---")
st.subheader("🌍 Real-Time AQI Forecast (Live API)")

lat = st.number_input("Latitude", value=28.6139)
lon = st.number_input("Longitude", value=77.2090)

if st.button("Predict Using Live API Data"):

    try:
        # Fetch full feature dataframe
        live_df, city, extra_pollutants = fetch_last_24_hours_full(lat, lon)
        st.subheader(f"Location: {city}")

        # Map showing selected location
        st.map(pd.DataFrame({
            "lat": [lat],
            "lon": [lon]
        }))

        st.subheader("Live Data Used for Prediction")
        st.dataframe(live_df.tail())
        st.write("PM2.5:", live_df["pm2_5"].iloc[-1])
        st.write("PM10:", live_df["pm10"].iloc[-1])
        st.write("Latest API Row:", live_df.tail(1))

        if len(live_df) < 24:
            st.error("Not enough data returned from API.")
            st.stop()

        # Convert AQI category to approximate numeric value
        live_df["aqi_index"] = live_df["aqi_index"].map({
            1: 25,
            2: 75,
            3: 125,
            4: 175,
            5: 225
        })

        # Scale features
        scaled_data = scaler.transform(live_df)

        # Reshape for LSTM (1 sample, 24 timesteps, 7 features)
        lstm_input = np.expand_dims(scaled_data, axis=0)

        # Predict
        lstm_pred_scaled = lstm_model.predict(lstm_input)

        # Inverse scaling
        dummy = np.zeros((1, 7))
        dummy[:, 0] = lstm_pred_scaled[:, 0]
        lstm_pred = scaler.inverse_transform(dummy)[0][0]

        # Clamp AQI range
        lstm_pred = max(10, min(lstm_pred, 350))

        st.write("Scaled Prediction:", lstm_pred_scaled)
        st.write("Final AQI:", lstm_pred)

        st.metric(
            label="Predicted Next Hour AQI",
            value=round(lstm_pred, 2)
        )

        # Health advice
        advice = health_advice(lstm_pred)

        st.subheader("Health Recommendation")
        st.info(advice)

        # Pollution components
        st.subheader("🧪 Pollution Components")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("PM2.5", round(live_df["pm2_5"].iloc[-1], 2))

        with col2:
            st.metric("PM10", round(live_df["pm10"].iloc[-1], 2))

        # Pollution distribution
        st.subheader("Pollution Distribution")

        pollution_data = {
            "PM2.5": live_df["pm2_5"].iloc[-1],
            "PM10": live_df["pm10"].iloc[-1],
            "NO2": extra_pollutants["NO2"],
            "CO": extra_pollutants["CO"],
            "O3": extra_pollutants["O3"],
            "SO2": extra_pollutants["SO2"]
        }

        fig2, ax2 = plt.subplots()
        ax2.bar(pollution_data.keys(), pollution_data.values())
        ax2.set_ylabel("Concentration")

        st.pyplot(fig2)

        st.subheader("Pollution Source Insight")

        dominant = max(pollution_data, key=pollution_data.get)

        if dominant == "PM2.5":
            st.warning("Fine particulate pollution is dominant. Likely caused by traffic, construction dust, or biomass burning.")

        elif dominant == "PM10":
            st.warning("Coarse particulate pollution detected. Possible sources include road dust and construction.")

        elif dominant == "NO2":
            st.warning("Nitrogen dioxide is high. This often indicates heavy vehicle emissions.")

        elif dominant == "SO2":
            st.warning("Sulfur dioxide detected. Possible industrial or power plant emissions.")

        elif dominant == "O3":
            st.warning("Ozone pollution detected. This forms during sunlight-driven chemical reactions in polluted air.")

        elif dominant == "CO":
            st.warning("Carbon monoxide levels elevated. This may indicate incomplete combustion from vehicles or generators.")

        # AQI Category
        category = get_aqi_category(lstm_pred)

        def get_color(aqi):
            if aqi <= 50:
                return "green"
            elif aqi <= 100:
                return "yellow"
            elif aqi <= 200:
                return "orange"
            elif aqi <= 300:
                return "red"
            else:
                return "purple"

        color = get_color(lstm_pred)

        # AQI Gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=lstm_pred,
                title={"text": "AQI Level"},
                gauge={
                    "axis": {"range": [0, 500]},
                    "steps": [
                        {"range": [0, 50], "color": "green"},
                        {"range": [50, 100], "color": "yellow"},
                        {"range": [100, 200], "color": "orange"},
                        {"range": [200, 300], "color": "red"},
                        {"range": [300, 500], "color": "purple"}
                    ]
                }
            )
        )

        st.plotly_chart(fig)

        # AQI Category Box
        st.markdown(
            f"""
            <div style="
                background-color:{color};
                padding:20px;
                border-radius:10px;
                text-align:center;
                font-size:20px;
                font-weight:bold;
                color:white;">
                AQI Category: {category}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Last 24 hour trend
        if "live_df" in locals():

            st.subheader("Last 24 Hour AQI Trend")

            fig, ax = plt.subplots()
            ax.plot(live_df["aqi_index"], marker="o")

            ax.set_xlabel("Time (Past 24 Hours)")
            ax.set_ylabel("AQI")

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error fetching API data: {e}")


# ---------------- Methodology Section ----------------

st.markdown("---")

st.subheader("System Methodology")

st.markdown("""
1 Collect historical air quality data  
2 Preprocess environmental features  
3 Train machine learning models  
4 Train LSTM deep learning model  
5 Deploy using Streamlit dashboard  
6 Integrate real-time environmental API
""")
 
st.caption("AI-Based Environmental Prediction System")