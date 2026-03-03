import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore

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
    
st.set_page_config(page_title="AQI Forecast System", layout="centered")

st.title("AQI Forecasting (Next Hour Prediction)")

st.write("Upload last 24 hours air quality data (CSV file).")

# Load model and scaler
model = load_model("aqi_lstm_model.h5", compile=False)
scaler = joblib.load("scaler.save")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)
    predicted_aqi = None
    lower_bound = None
    upper_bound = None

    st.write("Preview of uploaded data:")
    st.write(data.head())

    if len(data) != 24:
        st.error("Please upload exactly 24 rows (24 hours data).")

    else:

        if st.button("Predict Next Hour AQI"):

            scaled_data = scaler.transform(data)
            sequence = np.expand_dims(scaled_data, axis=0)
            prediction = model.predict(sequence)

            dummy = np.zeros((1, scaled_data.shape[1]))
            dummy[0, 0] = prediction[0][0]

            predicted_aqi = scaler.inverse_transform(dummy)[0][0]
            category = get_aqi_category(predicted_aqi)

            confidence_margin = 5
            lower_bound = predicted_aqi - confidence_margin
            upper_bound = predicted_aqi + confidence_margin

            st.success(f"Predicted Next Hour AQI: {round(predicted_aqi, 2)}")
            st.info(f"AQI Category: {category}")
            st.write(f"Prediction Range: {round(lower_bound,2)} - {round(upper_bound,2)}")

        # -------- GRAPH SECTION --------
        st.subheader("Last 24 Hours AQI Trend + Forecast")

        plt.figure()

        hours = list(range(24))
        future_hour = 24

        plt.plot(hours, data["aqi_index"].values, marker='o')

        if predicted_aqi is not None:

            plt.plot(
                [23, future_hour],
                [data["aqi_index"].values[-1], predicted_aqi],
                linestyle='--'
            )

            plt.scatter(future_hour, predicted_aqi)

            if lower_bound is not None and upper_bound is not None:
                plt.vlines(future_hour, lower_bound, upper_bound)

            plt.text(
                future_hour,
                predicted_aqi,
                f"{round(predicted_aqi, 2)}"
            )

        plt.xlabel("Hour")
        plt.ylabel("AQI")
        plt.xticks(range(0, 25))

        st.pyplot(plt)