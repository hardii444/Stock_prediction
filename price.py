import streamlit as st
import yfinance as yf
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Load saved model and scalers
def load_model_and_scalers():
    try:
        model = load_model('stock_price_predictor.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('close_price_scaler.pkl', 'rb') as f:
            close_price_scaler = pickle.load(f)
        return model, scaler, close_price_scaler
    except Exception as e:
        st.error(f"Error loading model or scalers: {e}")
        return None, None, None

# Calculate technical indicators
def calculate_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    middle = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_middle'] = middle
    df['BB_upper'] = middle + 2 * std
    df['BB_lower'] = middle - 2 * std

    return df

# Prepare the sequences for prediction
def prepare_sequences(data, seq_length):
    features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 
                'BB_middle', 'BB_upper', 'BB_lower']
    X, y = [], []
    for i in range(len(data) - seq_length - 5):
        X.append(data[features].iloc[i:(i + seq_length)].values)
        y.append(data['Close'].iloc[i + seq_length:i + seq_length + 5].values)
    return np.array(X), np.array(y)

# Predict stock prices
def predict_stock_prices(symbol, seq_length=60):
    try:
        stock_data = yf.download(symbol, start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'), interval='1d')
        if stock_data.empty:
            raise ValueError("No data found for the given stock symbol.")

        stock_data = calculate_technical_indicators(stock_data).dropna()
        model, scaler, close_price_scaler = load_model_and_scalers()

        if model is None or scaler is None or close_price_scaler is None:
            raise ValueError("Model or scalers could not be loaded.")

        features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 
                    'BB_middle', 'BB_upper', 'BB_lower']
        data_scaled = stock_data[features].copy()
        data_scaled[features] = scaler.transform(data_scaled[features])

        X, _ = prepare_sequences(data_scaled, seq_length)
        last_sequence = X[-1:]
        prediction = model.predict(last_sequence)

        predicted_prices = close_price_scaler.inverse_transform(prediction[0].reshape(-1, 1))

        last_date = stock_data.index[-1]
        future_dates = [(last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(5)]

        predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': predicted_prices.flatten()
        })
        return stock_data, predictions

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Generate a predicted price chart
def plot_predicted_price_chart(predictions):
    fig = go.Figure(data=[go.Scatter(x=pd.to_datetime(predictions['Date']),
                                     y=predictions['Predicted Price'],
                                     mode='lines+markers',
                                     name='Predicted Price',
                                     line=dict(color='orange', width=2))])
    fig.update_layout(title="Predicted Stock Prices for Next 5 Days",
                      xaxis_title="Date",
                      yaxis_title="Price (INR)")
    st.plotly_chart(fig)

# Streamlit UI
st.title("Stock Price Prediction with Technical Analysis")
st.markdown("Enter a stock symbol to predict its next 5 days' closing prices and view technical indicators.")

symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS, AAPL, TSLA)", "RELIANCE.NS")

if st.button('Get Predictions and Recommendations'):
    stock_data, predictions = predict_stock_prices(symbol)
    if predictions is not None:
        st.write(f"Predicted closing prices for {symbol} for the next 5 days:")
        st.dataframe(predictions)
        plot_predicted_price_chart(predictions)
