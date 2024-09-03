import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Dictionary of model file paths
model_files = {
    'NVDA': 'NVIDIA_LSTM_5(1.38).h5',
    'AMZN': 'AMAZON_LSTM_5(2.90).h5',
    'AAPL': 'APPLE_LSTM_10(2.05).h5',
    'NFLX': 'NETFLIX_LSTM_10(12.04).h5',
    'META': 'META_LSTM_5(8.12).h5',
    'TSLA': 'TESLA_LSTM_30(8.98).h5',
    'IBM': 'IBM_LSTM_5(1.59).h5',
    'KO': 'COCACOLA_LSTM_10(0.48).h5',
    'MSFT': 'MICROSOFT_LSTM_10(3.84).h5',
    'GOOGL': 'GOOGLE_LSTM_5(2.33).h5'
}

# Define look-back periods for each stock
look_back_periods = {
    'NVDA': 5,
    'AMZN': 5,
    'AAPL': 10,
    'NFLX': 10,
    'META': 5,
    'TSLA': 30,
    'IBM': 5,
    'KO': 10,
    'MSFT': 10,
    'GOOGL': 5
}

# Company logos URLs
logo_urls = {
    'NVDA': 'https://1000logos.net/wp-content/uploads/2017/05/Color-NVIDIA-Logo.jpg',
    'AMZN': 'https://1000logos.net/wp-content/uploads/2016/10/Amazon-logo-meaning.jpg',
    'AAPL': 'https://companiesmarketcap.com/img/company-logos/64/AAPL.webp',
    'NFLX': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Netflix_icon.svg/2048px-Netflix_icon.svg.png',
    'META': 'https://1000logos.net/wp-content/uploads/2021/10/logo-Meta.png',
    'TSLA': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Tesla_logo.png/900px-Tesla_logo.png',
    'IBM': 'https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg',
    'KO': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Coca-Cola_logo.svg/1200px-Coca-Cola_logo.svg.png',
    'MSFT': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Microsoft_logo.svg/1024px-Microsoft_logo.svg.png',
    'GOOGL': 'https://w7.pngwing.com/pngs/63/1016/png-transparent-google-logo-google-logo-g-suite-chrome-text-logo-chrome.png'
}

# Load all trained LSTM models with error handling
models = {}
for stock, file in model_files.items():
    try:
        models[stock] = load_model(file)
        print(f"{stock} model loaded successfully.")
    except Exception as e:
        print(f"Error loading {stock} model: {e}")

# Function to get the stock data
def get_stock_data(ticker='NVDA'):
    data = yf.download(ticker, period='max')
    return data

# Function to generate a list of business days
def generate_business_days(start_date, num_days):
    """
    Generate a list of business days starting from start_date for num_days.
    """
    return pd.bdate_range(start=start_date, periods=num_days).tolist()

# Function to make predictions for business days
def predict_next_business_days(model, data, look_back=30, days=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    last_sequence = data_scaled[-look_back:]
    predictions = []

    for _ in range(days):
        X_input = np.reshape(last_sequence, (1, look_back, 1))
        prediction = model.predict(X_input)
        predictions.append(prediction[0, 0])
        
        # Update the sequence for the next prediction
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    
    # Inverse transform the predictions to the original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Streamlit app layout
st.markdown("<h1 style='text-align: center; font-size: 44px;'>Stock-Price-Predictor 📈📉💰</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)  # Add a gap between rows

# Initialize stock selection in session state
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None

# Display company logos in a grid with 4 per row and add gaps between rows
def display_logos():
    # First row of logos
    cols = st.columns(4)
    with cols[0]:
        if st.button("NVIDIA", key='NVDA_button'):
            st.session_state.selected_stock = 'NVDA'
        st.image(logo_urls['NVDA'], width=90)
    with cols[1]:
        if st.button("Amazon", key='AMZN_button'):
            st.session_state.selected_stock = 'AMZN'
        st.image(logo_urls['AMZN'], width=100)
    with cols[2]:
        if st.button("Apple", key='AAPL_button'):
            st.session_state.selected_stock = 'AAPL'
        st.image(logo_urls['AAPL'], width=50)
    with cols[3]:
        if st.button("Netflix", key='NFLX_button'):
            st.session_state.selected_stock = 'NFLX'
        st.image(logo_urls['NFLX'], width=50)
    
    st.markdown("<br>", unsafe_allow_html=True)  # Add a gap between rows

    # Second row of logos
    cols = st.columns(4)
    with cols[0]:
        if st.button("Meta", key='META_button'):
            st.session_state.selected_stock = 'META'
        st.image(logo_urls['META'], width=70)
    with cols[1]:
        if st.button("Tesla", key='TSLA_button'):
            st.session_state.selected_stock = 'TSLA'
        st.image(logo_urls['TSLA'], width=80)
    with cols[2]:
        if st.button("IBM", key='IBM_button'):
            st.session_state.selected_stock = 'IBM'
        st.image(logo_urls['IBM'], width=100)
    with cols[3]:
        if st.button("Coca-Cola", key='KO_button'):
            st.session_state.selected_stock = 'KO'
        st.image(logo_urls['KO'], width=120)
    
    st.markdown("<br>", unsafe_allow_html=True)  # Add a gap between rows

    # Third row of logos
    cols = st.columns(4)
    with cols[0]:
        if st.button("Microsoft", key='MSFT_button'):
            st.session_state.selected_stock = 'MSFT'
        st.image(logo_urls['MSFT'], width=55)
    with cols[1]:
        if st.button("Google", key='GOOGL_button'):
            st.session_state.selected_stock = 'GOOGL'
        st.image(logo_urls['GOOGL'], width=55)

display_logos()

# Get the selected stock from session state
stock = st.session_state.selected_stock

if stock is None:
    st.error("Please select a stock.")
else:
    # User input for number of business days to forecast
    num_days = st.slider("Select number of business days to forecast", min_value=1, max_value=30, value=5)

    # Display current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    st.write(f"Current Date: {current_date}")

    # Apply custom CSS to change button colors and add green outline when clicked
    st.markdown(
        """
        <style>
        div.stButton > button#forecast-button {
            background-color: green; /* Green color for the Forecast button */
            color: white;
        }
        div.stButton > button#forecast-button:focus,
        div.stButton > button#forecast-button:hover,
        div.stButton > button#forecast-button:active {
            color: white; /* Keep text white on hover, focus, and active states for the Forecast button */
            outline: 2px solid green; /* Green outline for the clicked button */
        }

        div.stButton > button:not(#forecast-button) {
            background-color: red; /* Red color for all other buttons */
            color: white;
        }
        div.stButton > button:not(#forecast-button):focus,
        div.stButton > button:not(#forecast-button):hover,
        div.stButton > button:not(#forecast-button):active {
            color: white; /* Keep text white on hover, focus, and active states for the other buttons */
            outline: 2px solid green; /* Green outline for the clicked button */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use unique key for the "Forecast" button
    if st.button(f'Predict Next {num_days} Days Stock Prices for {stock}', key='forecast-button'):
        # Load stock data
        stock_data = get_stock_data(stock)
        close_prices = stock_data['Close'].values.reshape(-1, 1)
        dates = stock_data.index

        # Display the historical data with increased height
        st.markdown(f"### Historical Data for {stock}")
        st.dataframe(stock_data, height=400, width = 1000)  # Increased height for better visibility

        # Get the appropriate model and look-back period
        model_lstm = models.get(stock)
        if model_lstm is None:
            st.error(f"No model found for {stock}")
        else:
            look_back = look_back_periods.get(stock, 30)

            # Predict the next num_days business days
            predictions = predict_next_business_days(model_lstm, close_prices, look_back=look_back, days=num_days)
            
            # Create dates for the predictions
            last_date = dates[-1]
            prediction_dates = generate_business_days(last_date + timedelta(days=1), num_days)
            
            # Prepare data for plotting the historical and predicted prices
            fig, ax = plt.subplots()
            ax.plot(dates, close_prices, label='Historical Prices')
            ax.plot(prediction_dates, predictions, label='Predicted Prices', linestyle='--', color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(f'{stock} Stock Prices', fontsize=10, fontweight='bold')
            ax.legend()

            st.pyplot(fig)

            st.write(" ")
            
            # Plot only the predicted stock prices
            fig2, ax2 = plt.subplots()
            ax2.plot(prediction_dates, predictions, marker='o', color='blue')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Predicted Price')
            ax2.set_title(f'Predicted Stock Prices for the Next {num_days} Business Days ({stock})', fontsize=10, fontweight='bold')
            
            # Use DayLocator to specify spacing of tick marks and set the format for the date labels
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            plt.xticks(rotation=45)
            
            st.pyplot(fig2)
            
            st.write(" ")
            
            # Show predictions in a table format with increased height
            prediction_df = pd.DataFrame({
                'Date': prediction_dates,
                'Predicted Price': predictions.flatten()
            })
            st.markdown(f"##### Predicted Stock Prices for the Next {num_days} Business Days ({stock})")
            st.dataframe(prediction_df, height=400, width = 1000)  # Increased height for better visibility
