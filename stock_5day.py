import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

def calculate_technical_indicators(df):
    df = df.copy()
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    middle = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_middle'] = middle
    df['BB_upper'] = middle + 2*std
    df['BB_lower'] = middle - 2*std
    
    return df

def fetch_stock_data(symbol, start_date='2020-01-01'):
    stock = yf.download(symbol, start=start_date, end=datetime.now().strftime('%Y-%m-%d'))
    stock = calculate_technical_indicators(stock)
    return stock.dropna()

def prepare_sequences(data, seq_length):
    features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 
                'BB_middle', 'BB_upper', 'BB_lower']
    
    X, y = [], []
    for i in range(len(data) - seq_length - 5):
        X.append(data[features].iloc[i:(i + seq_length)].values)
        y.append(data['Close'].iloc[i + seq_length:i + seq_length + 5].values)
    return np.array(X), np.array(y)

def create_model(seq_length, n_features):
    model = Sequential([
        LSTM(100, activation='relu', input_shape=(seq_length, n_features), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(5)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_stock_prices_and_save_model(symbol, seq_length=60):
    # Fetch data with technical indicators
    data = fetch_stock_data(symbol)
    
    # Scale all features
    scaler = MinMaxScaler()
    features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 
                'BB_middle', 'BB_upper', 'BB_lower']
    
    data_scaled = data[features].copy()
    data_scaled[features] = scaler.fit_transform(data[features])
    
    # Prepare sequences
    X, y = prepare_sequences(data_scaled, seq_length)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train model
    n_features = X.shape[2]
    model = create_model(seq_length, n_features)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    
    # Save the model
    from keras.losses import mean_squared_error

# Compile the model explicitly before saving (if not already compiled)
    model.compile(optimizer='adam', loss=mean_squared_error)

# Save the model
    model.save('stock_price_predictor.h5', save_format='h5')

    
    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Predict next 5 days
    last_sequence = X[-1:]
    prediction = model.predict(last_sequence)
    
    # Inverse transform predictions for Close price only
    close_price_scaler = MinMaxScaler()
    close_price_scaler.fit(data[['Close']])
    predicted_prices = close_price_scaler.inverse_transform(prediction[0].reshape(-1, 1))
    
    # Save the Close price scaler
    with open('close_price_scaler.pkl', 'wb') as f:
        pickle.dump(close_price_scaler, f)
    
    # Get dates for predictions
    last_date = datetime.now()
    future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                   for i in range(5)]
    
    predictions = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': predicted_prices.flatten()
    })
    
    return predictions

# Example usage
symbol = 'RELIANCE.NS'
predictions = predict_stock_prices_and_save_model(symbol)
print(f"\nPredicted prices for {symbol}:")
print(predictions)
