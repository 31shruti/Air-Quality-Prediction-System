# Air Quality Prediction System
The LSTM (Long Short-Term Memory) (Deep Learning) based project that is a predictive system for identifying Air Quality.
# 🌍 Air Quality Forecasting Using LSTM  
A Time-Series Deep Learning Approach for Short-Term AQI Prediction  

---

## 📌 1. Problem Statement

Air Quality Index (AQI) forecasting is crucial for public health planning and environmental risk mitigation.  
Traditional statistical models often fail to capture nonlinear temporal dependencies in environmental data.  

This project proposes a deep learning-based time-series forecasting system using LSTM (Long Short-Term Memory) networks to predict next-hour AQI based on 24-hour historical environmental observations.

---

## 🎯 2. Motivation

- Air pollution levels fluctuate dynamically due to meteorological and anthropogenic factors.
- Short-term AQI forecasting enables early warnings.
- Sequential modeling is required due to temporal dependencies.
- Deep learning models like LSTM are well-suited for capturing long-term dependencies in time-series data.

This project explores the effectiveness of LSTM in short-term environmental forecasting.

---

## 🧠 3. Methodology

### Data Representation
The model uses 24-hour sequential input data consisting of:

- AQI Index
- Temperature (°C)
- Humidity (%)
- Wind Speed (kph)
- PM2.5
- PM10
- Atmospheric Pressure (mb)

### Preprocessing
- MinMax scaling applied to normalize input features.
- Data reshaped into 3D tensors for LSTM input:  
  `(samples, time_steps=24, features)`

### Model Architecture
- LSTM Layer
- Dense Output Layer
- Output: Next-hour AQI value

### Forecasting Strategy
Single-step ahead forecasting (t+1 prediction).

---

## 🏗 4. System Architecture

1. User uploads 24-hour environmental data (CSV).
2. Data is scaled using trained MinMaxScaler.
3. LSTM model predicts next-hour AQI.
4. Prediction is inverse transformed.
5. AQI category classification is applied.
6. Forecast visualization is generated with confidence band.

---

## 📊 5. Results

Example Output:

- Predicted Next Hour AQI: 39.15  
- AQI Category: Good  
- Prediction Range: 34.15 – 44.15  

The system visualizes:
- Historical AQI trend
- Forecast extension
- Simulated uncertainty interval

---

## 📈 6. Key Features

- 24-hour sequential modeling
- LSTM-based time-series forecasting
- Environmental parameter integration
- Forecast visualization
- AQI health categorization
- Confidence band simulation

---

## ⚠️ 7. Limitations

- Single-step forecasting only
- Confidence interval is heuristic-based
- No external event modeling (rain, traffic spikes, wildfire)
- No cross-city generalization validation

---

## 🚀 8. Future Scope

- Multi-step forecasting (6–12 hour horizon)
- Monte Carlo Dropout for uncertainty estimation
- Model comparison (LSTM vs GRU vs Random Forest)
- Real-time API integration
- Spatial AQI forecasting using GIS layers
- Model evaluation metrics (MAE, RMSE, R²)

---

## 🛠 9. Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Streamlit
- Matplotlib
- Scikit-learn

---

## ▶️ 10. How to Run

```bash
pip install -r requirements.txt
streamlit run app.py