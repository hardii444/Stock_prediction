

# Stock Prediction App

This project is a simple stock prediction tool developed using **Streamlit** and **Python**. It predicts stock prices based on historical data using machine learning algorithms. The application visualizes stock trends and provides users with a prediction for future stock prices.



## Project Overview

This stock prediction app uses historical stock data (e.g., stock price, volume, open, close, high, low) and applies machine learning algorithms to predict the future price of a selected stock. The app allows users to visualize the stock data and observe predicted trends over time.

The app is built using **Streamlit** for the frontend and **Python** for the backend. The machine learning models used in the project include **Linear Regression**, **Random Forest**, or any model of choice that can handle time-series data.



## Usage

To run the Streamlit app, execute the following command in your terminal:

```bash
streamlit run price.py
```

This will start the Streamlit app in your default browser, where you can input the stock symbol and view the stock prediction results.

### How to Use the App:
1. Select a stock symbol (e.g., AAPL for Apple, TSLA for Tesla).
2. Choose a prediction range (e.g., 1 week, 1 month).
3. The app will fetch historical data and display the stock trends.
4. The model will predict future prices, and the prediction graph will be displayed.

## Features


- **Prediction**: Predict future stock prices using machine learning models.
- **Data Selection**: Choose a stock symbol.
- **Model Performance**: View metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) for model evaluation.

## Technologies Used

- **Python**: The primary language used for data analysis and model building.
- **Streamlit**: For building the interactive web interface.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning models and metrics.
- **Matplotlib**/**Plotly**: For visualizing the stock data and predictions.
- **yfinance**: For fetching historical stock data.

## Future Improvements
- Add portfolio management features for tracking multiple stocks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

