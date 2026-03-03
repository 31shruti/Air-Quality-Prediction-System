#-----Data Preprocessing Utility Functions-----
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Function to load dataset from CSV file
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    return df

# Function to scale selected feature columns
def scale_data(df, feature_cols):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    return scaled, scaler

# Function to create time-series sequences for LSTM
def create_sequences(data, seq_length, forecast_horizon):
    X, y = [], []

    for i in range(len(data) - seq_length - forecast_horizon):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_horizon, 0])

    return np.array(X), np.array(y)