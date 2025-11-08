import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Fix for yfinance with pandas_datareader
pdr.get_data_yahoo = yf.download

# Load your trained model
model = load_model('keras_model.h5')

# App title
st.title('📈 Stock Price Prediction App')

# Date range
start = '2010-01-01'
end = '2019-12-31'

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker Symbol (e.g., AAPL, TSLA, INFY)', '')

if user_input:
    try:
        # Fetch stock data
        df = pdr.get_data_yahoo(user_input, start=start, end=end)

        # Display basic info
        st.subheader(f'{user_input} Data from 2010 - 2019')
        st.write(df.describe())

        # Closing price chart
        st.subheader('Closing Price vs Time')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Closing Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)

        # Prepare data for prediction
        data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):])

        scaler = MinMaxScaler(feature_range=(0,1))

        past_100_days = data_train.tail(100)
        final_df = pd.concat([past_100_days, data_test], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        # Splitting the data
        x_test = []
        y_test = []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        y_predicted = model.predict(x_test)
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plot predictions
        st.subheader('Predicted Price vs Original Price')
        fig2 = plt.figure(figsize=(12,6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error fetching data for {user_input}: {e}")
else:
    st.info("👆 Please enter a stock ticker symbol to begin.")
