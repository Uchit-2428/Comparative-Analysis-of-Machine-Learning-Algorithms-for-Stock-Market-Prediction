import streamlit as st
import pandas as pd
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import logging

# Disable warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set logging level to suppress warnings
logging.basicConfig(level=logging.ERROR)

global combined_df, dfs
tickers = ['RELIANCE.NS', 'INFY.NS', 'LT.NS', 'ADANIENT.NS']
start_date = '2019-01-01'
end_date = datetime.now()
def download_and_preprocess_data(tickers, start_date, end_date):
    dfs = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        dfs[ticker] = df
    combined_df = pd.DataFrame()
    for ticker, df in dfs.items():
        combined_df[ticker] = df['Close']
    return combined_df, dfs


combined_df, dfs = download_and_preprocess_data(tickers, start_date, end_date)
# Streamlit App
def main():
    tickers = ['RELIANCE.NS', 'INFY.NS', 'LT.NS', 'ADANIENT.NS']
    start_date = '2019-01-01'
    end_date = datetime.now()
    

        
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(combined_df)
    time_step = 60  # Number of time steps
    X = []
    y = []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:i + time_step])
        y.append(scaled_data[i + time_step])
    X = np.array(X)
    y = np.array(y)

    # Split the data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    # Load the saved RNN model




    # Page title
    st.title("Stock Price Prediction")
    option1="Compare Algos for Stock X keeping average as Control"
    option2="Compare Time Period Of Prediction"
    options=[option1,option2] 
    option= st.selectbox("Select Option", options, key="algos")
    if option==option1:

        # Display 1row*4columns(Algos)
        model1=keras.models.load_model("rnn_model_80-20.h5")
        model2=keras.models.load_model("rnn_model_80-20.h5")
        model3=keras.models.load_model("gru_model_80-20.h5")
        model4=keras.models.load_model("lstm_model_80-20.h5")
        tickers = ['RELIANCE.NS', 'INFY.NS', 'LT.NS', 'ADANIENT.NS']
        data = yf.download(tickers, start='2023-02-20', end='2024-02-28')
        for i, ticker in enumerate(tickers):
            st.subheader(f"{ticker}")
            col1, col2, col3, col4 = st.columns(4)
            
            
            with col1:
                st.write(f"Evaluation metrics for **Moving Average**:")

                closing_prices = data['Close']

                # Calculate the moving average for each stock
                moving_avg = closing_prices.rolling(window=5).mean()

                # Predict the closing prices for the specified period for the current stock
                predicted_prices = moving_avg[ticker]['2023-02-28':'2024-02-28']

                # Load actual closing prices for the specified period for the current stock
                actual_closing_prices = closing_prices[ticker]['2023-02-28':'2024-02-28']

                # Calculate Algorithm the current stock
                mse = mean_squared_error(actual_closing_prices, predicted_prices)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual_closing_prices, predicted_prices)
                r2 = r2_score(actual_closing_prices, predicted_prices)

                # Print Eval Metrics
                
                st.write(f"MSE: {mse}")
                st.write(f"MAE: {mae}")
                st.write(f"R2 Score: {r2}")
                st.write()
                

                # Plot the graph for the current stock
                plt.figure(figsize=(10, 5))
                plt.plot(closing_prices.index, closing_prices[ticker], label='Actual Closing Prices', color='blue')
                plt.plot(predicted_prices.index, predicted_prices, label='Predicted Closing Prices', color='red')
                plt.title(f'{ticker} Actual vs Predicted Closing Prices')
                plt.xlabel('Date')
                plt.ylabel('Closing Price')
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.tight_layout()
                st.pyplot()
            with col2:
                st.write(f"Evaluation metrics for **RNN**:")
                predictions = model2.predict(X_test)
                predictions_original_scale = scaler.inverse_transform(predictions)
                y_test_original_scale = scaler.inverse_transform(y_test)
                evaluation_results = {}
                
                mse, mae, r2 = calculate_evaluation_metrics(y_test_original_scale[:, i], predictions_original_scale[:, i])         
                st.write(f"MSE: {mse}")
                st.write(f"MAE: {mae}")
                st.write(f"R2 Score: {r2}")
                st.write()

                date_list = dfs[ticker].index.tolist()[-len(y_test_original_scale):]
                true_values = y_test_original_scale[:, i]
                predicted_values = predictions_original_scale[:, i]
                plots=plot_results(date_list, true_values, predicted_values, ticker)
                st.pyplot(plots)
            with col3:
                st.write(f"Evaluation metrics for **GRU**:")
                predictions = model3.predict(X_test)
                predictions_original_scale = scaler.inverse_transform(predictions)
                y_test_original_scale = scaler.inverse_transform(y_test)
                evaluation_results = {}
                
                mse, mae, r2 = calculate_evaluation_metrics(y_test_original_scale[:, i], predictions_original_scale[:, i])         
                st.write(f"MSE: {mse}")
                st.write(f"MAE: {mae}")
                st.write(f"R2 Score: {r2}")
                st.write()

                date_list = dfs[ticker].index.tolist()[-len(y_test_original_scale):]
                true_values = y_test_original_scale[:, i]
                predicted_values = predictions_original_scale[:, i]
                plots=plot_results(date_list, true_values, predicted_values, ticker)
                st.pyplot(plots)
            with col4:
                st.write(f"Evaluation metrics for **LSTM**:")
                predictions = model4.predict(X_test)
                predictions_original_scale = scaler.inverse_transform(predictions)
                y_test_original_scale = scaler.inverse_transform(y_test)
                evaluation_results = {}
                
                mse, mae, r2 = calculate_evaluation_metrics(y_test_original_scale[:, i], predictions_original_scale[:, i])         
                st.write(f"MSE: {mse}")
                st.write(f"MAE: {mae}")
                st.write(f"R2 Score: {r2}")
                st.write()

                date_list = dfs[ticker].index.tolist()[-len(y_test_original_scale):]
                true_values = y_test_original_scale[:, i]
                predicted_values = predictions_original_scale[:, i]
                plots=plot_results(date_list, true_values, predicted_values, ticker)
                st.pyplot(plots)
                


    if option==option2:
        algos = ['Moving Average', 'RNN', 'GRU', 'LSTM']
        selected_algo = st.selectbox("Select Algo", algos, key="algo")

            #Display 4rows(Stocks) 2columns(3months & 1 year) graph for moving average
        if selected_algo=="Moving Average":
            col1,col2=st.columns(2)
            with col1:
                st.write("### 6 Months")
                tickers = ['RELIANCE.NS', 'INFY.NS', 'LT.NS', 'ADANIENT.NS']
                data = yf.download(tickers, start='2023-08-25', end='2024-02-28')
                closing_prices = data['Close']

                # Calculate the moving average for each stock
                moving_avg = closing_prices.rolling(window=5).mean()

                # Iterate over each stock ticker
                for ticker in tickers:
                    # Predict the closing prices for the specified period for the current stock
                    predicted_prices = moving_avg[ticker]['2023-09-01':'2024-02-28']

                    # Load actual closing prices for the specified period for the current stock
                    actual_closing_prices = closing_prices[ticker]['2023-09-01':'2024-02-28']

                    # Calculate evaluation metrics for the current stock
                    mse = mean_squared_error(actual_closing_prices, predicted_prices)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(actual_closing_prices, predicted_prices)
                    r2 = r2_score(actual_closing_prices, predicted_prices)

                    # Print Eval Metrics
                    st.write(f"Evaluation metrics for **{ticker}**:")
                    st.write(f"MSE: {mse}")
                    st.write(f"MAE: {mae}")
                    st.write(f"R2 Score: {r2}")
                    st.write()
                    

                    # Plot the graph for the current stock
                    plt.figure(figsize=(10, 5))
                    plt.plot(closing_prices.index, closing_prices[ticker], label='Actual Closing Prices', color='blue')
                    plt.plot(predicted_prices.index, predicted_prices, label='Predicted Closing Prices', color='red')
                    plt.title(f'{ticker} Actual vs Predicted Closing Prices')
                    plt.xlabel('Date')
                    plt.ylabel('Closing Price')
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.grid(True)
                    plt.tight_layout()
                    st.pyplot()
                with col2:
                    st.write("### 1 Year")
                    tickers = ['RELIANCE.NS', 'INFY.NS', 'LT.NS', 'ADANIENT.NS']
                    data = yf.download(tickers, start='2023-02-20', end='2024-02-28')
                    closing_prices = data['Close']

                    # Calculate the moving average for each stock
                    moving_avg = closing_prices.rolling(window=5).mean()

                    # Iterate over each stock ticker
                    for ticker in tickers:
                        # Predict the closing prices for the specified period for the current stock
                        predicted_prices = moving_avg[ticker]['2023-02-28':'2024-02-28']

                        # Load actual closing prices for the specified period for the current stock
                        actual_closing_prices = closing_prices[ticker]['2023-02-28':'2024-02-28']

                        # Calculate evaluation metrics for the current stock
                        mse = mean_squared_error(actual_closing_prices, predicted_prices)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(actual_closing_prices, predicted_prices)
                        r2 = r2_score(actual_closing_prices, predicted_prices)

                        # Print Eval Metrics
                        st.write(f"Evaluation metrics for **{ticker}**:")
                        st.write(f"MSE: {mse}")
                        st.write(f"MAE: {mae}")
                        st.write(f"R2 Score: {r2}")
                        st.write()
                        

                        # Plot the graph for the current stock
                        plt.figure(figsize=(10, 5))
                        plt.plot(closing_prices.index, closing_prices[ticker], label='Actual Closing Prices', color='blue')
                        plt.plot(predicted_prices.index, predicted_prices, label='Predicted Closing Prices', color='red')
                        plt.title(f'{ticker} Actual vs Predicted Closing Prices')
                        plt.xlabel('Date')
                        plt.ylabel('Closing Price')
                        plt.legend()
                        plt.xticks(rotation=45)
                        plt.grid(True)
                        plt.tight_layout()
                        st.pyplot()

        else:   
            for i, ticker in enumerate(tickers):
                col1, col2 = st.columns(2)
                # Plot in the first column
                with col1:
                    st.write("### 6 Months")
                    
                    loaded_model = keras.models.load_model(f"{selected_algo.lower()}_model_80-20.h5")

                    # Evaluate the model on the test set
                    predictions = loaded_model.predict(X_test)
                    predictions_original_scale = scaler.inverse_transform(predictions)
                    y_test_original_scale = scaler.inverse_transform(y_test)
                    evaluation_results = {}
                    
                    mse, mae, r2 = calculate_evaluation_metrics(y_test_original_scale[:, i], predictions_original_scale[:, i])         
                    st.write(f"Evaluation metrics for **{ticker}**:")
                    st.write(f"MSE: {mse}")
                    st.write(f"MAE: {mae}")
                    st.write(f"R2 Score: {r2}")
                    st.write()

                    date_list = dfs[ticker].index.tolist()[-len(y_test_original_scale):]
                    true_values = y_test_original_scale[:, i]
                    predicted_values = predictions_original_scale[:, i]
                    plots=plot_results(date_list, true_values, predicted_values, ticker)
                    st.pyplot(plots)

                # Plot in the second column
                with col2:
                    st.write("### 1 Year")
                    # Split the data into train and test sets
                    train_size = int(len(X) * 0.9)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    loaded_model = keras.models.load_model(f"{selected_algo.lower()}_model_90-10.h5")

                    # Evaluate the model on the test set
                    predictions = loaded_model.predict(X_test)
                    predictions_original_scale = scaler.inverse_transform(predictions)
                    y_test_original_scale = scaler.inverse_transform(y_test)
                    evaluation_results = {}
                    
                    mse, mae, r2 = calculate_evaluation_metrics(y_test_original_scale[:, i], predictions_original_scale[:, i])         
                    st.write(f"Evaluation metrics for **{ticker}** :")
                    st.write(f"MSE: {mse}")
                    st.write(f"MAE: {mae}")
                    st.write(f"R2 Score: {r2}")
                    st.write()

                    date_list = dfs[ticker].index.tolist()[-len(y_test_original_scale):]
                    true_values = y_test_original_scale[:, i]
                    predicted_values = predictions_original_scale[:, i]
                    plots=plot_results(date_list, true_values, predicted_values, ticker)
                    st.pyplot(plots)
            
# Function to calculate evaluation metrics
def calculate_evaluation_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2



def plot_results(date_list, true_values, predicted_values, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(date_list, true_values, label='True Close Price', color='blue')
    plt.plot(date_list, predicted_values, label='Predicted Close Price', color='red')
    plt.title(f'True vs Predicted Close Price for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    
def load_model(filename):
    # with open(filename, 'rb') as f:
    #     model = pickle.load(f)
    model=pd.read_pickle(filename)
    return model

if __name__ == "__main__":
    main()