import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st

start = '2015-01-01'
end = '2024-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

# Describing the data
st.subheader('Data from 2015-2024')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], 'b')
plt.title(f'{user_input} Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], 'b', label='Close Price')
plt.plot(ma100, 'orange', label='100-Day MA')
plt.title(f'{user_input} Closing Price with 100-Day MA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], 'b', label='Close Price')
plt.plot(ma100, 'orange', label='100-Day MA')
plt.plot(ma200, 'r', label='200-Day MA')
plt.title(f'{user_input} Closing Price with 200-Day MA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
st.pyplot(fig)

# Preparing the data for LSTM model
# We are using the closing price for prediction
# Splitting data into training and testing sets
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)


# Splitting the data into x_train and y_train
# Here, we take the last 100 days of data to predict the next day price
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i, 0])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Loading the model
model = load_model('keras_model.h5')

# Testing the model on existing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1 / scaler[0]  
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final Visualization
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time (days)')
plt.ylabel('Price (USD)')
plt.legend()
st.pyplot(fig2)