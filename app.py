import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(page_title="Stock Market Predictor", layout="wide")
st.title("Stock Market Predictor")


COMPANY_TO_SYMBOL = {
    "GOOGLE": "GOOG",
    "ALPHABET": "GOOG",
    "APPLE": "AAPL",
    "TESLA": "TSLA",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "META": "META",
    "FACEBOOK": "META",
    "NVIDIA": "NVDA",
    "NETFLIX": "NFLX",
    "IBM": "IBM",
    "INTEL": "INTC",
    "AMD": "AMD",
    "TATA MOTORS": "TATAMOTORS.NS",
    "TCS": "TCS.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "RELIANCE": "RELIANCE.NS",
    "INFOSYS": "INFY.NS",
    "WIPRO": "WIPRO.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
}


def get_stock_symbol(user_input):
    """Convert common company names to Yahoo Finance symbols.

    If the user already enters a valid stock symbol like AAPL or GOOG,
    the same value is returned.
    """
    cleaned_input = user_input.strip().upper()
    return COMPANY_TO_SYMBOL.get(cleaned_input, cleaned_input)



@st.cache_resource
def get_model():
    """Load the Keras model from the same folder as this app.py file."""
    model_path = os.path.join(os.path.dirname(__file__), "Stock Predictions Model.keras")

    if not os.path.exists(model_path):
        st.error(
            "Model file not found. Keep 'Stock Predictions Model.keras' in the same folder as app.py."
        )
        st.stop()

    return load_model(model_path)


model = get_model()

user_stock_input = st.text_input("Enter Company Name or Stock Symbol", "Google")
stock = get_stock_symbol(user_stock_input)
st.caption(f"Using Yahoo Finance symbol: {stock}")
start = "2014-01-01"
end = "2024-02-28"

if not user_stock_input.strip():
    st.warning("Please enter a valid company name or stock symbol.")
    st.stop()

try:
    data = yf.download(stock, start=start, end=end, progress=False)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if data.empty:
    st.warning("No data fetched. Symbol may be incorrect or data may not be available.")
    st.stop()

# Some versions of yfinance return multi-index columns. This converts them to normal columns.
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

try:
    ticker = yf.Ticker(stock)
    infor = ticker.info or {}
    news = ticker.news or []
except Exception:
    infor = {}
    news = []

company_name = infor.get("longName") or infor.get("shortName") or stock
st.subheader(company_name)

col1, col2 = st.columns(2)

with col1:
    st.write("Quote Type:", infor.get("quoteType", "Not available"))
    st.write("Exchange:", infor.get("exchange", "Not available"))

with col2:
    current_price = infor.get("currentPrice") or infor.get("regularMarketPrice")
    currency = infor.get("currency", "Not available")

    if current_price is not None:
        st.write("Current Price:", current_price, currency)
    else:
        st.write("Currency:", currency)

    st.write("TimeZone:", infor.get("timeZoneFullName", "Not available"))

# ------------------------------------------------------------------------------------------------------------

st.text("")
st.subheader("Stock Data")
st.write(data)

# ------------------------------------------------------------------------------------------------------------

st.text("")
st.subheader("Moving Averages")


def plot_graph(dataframe, ma_50_days=None, ma_100_days=None, ma_200_days=None):
    fig = plt.figure(figsize=(8, 6))

    if ma_50_days is not None:
        plt.plot(ma_50_days, label="MA 50")

    if ma_100_days is not None:
        plt.plot(ma_100_days, label="MA 100")

    if ma_200_days is not None:
        plt.plot(ma_200_days, label="MA 200")

    plt.plot(dataframe["Close"], label="Closing Price")
    plt.legend()
    plt.title("Moving Average and Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    st.pyplot(fig)
    plt.close(fig)


def calculate_moving_averages(dataframe):
    ma_50_days = dataframe["Close"].rolling(50).mean()
    ma_100_days = dataframe["Close"].rolling(100).mean()
    ma_200_days = dataframe["Close"].rolling(200).mean()
    return ma_50_days, ma_100_days, ma_200_days


def plot_price_vs_ma50(dataframe):
    ma_50_days, _, _ = calculate_moving_averages(dataframe)
    plot_graph(dataframe, ma_50_days=ma_50_days)


def plot_ma50_vs_ma100(dataframe):
    ma_50_days, ma_100_days, _ = calculate_moving_averages(dataframe)
    plot_graph(dataframe, ma_50_days=ma_50_days, ma_100_days=ma_100_days)


def plot_ma100_vs_ma200(dataframe):
    _, ma_100_days, ma_200_days = calculate_moving_averages(dataframe)
    plot_graph(dataframe, ma_100_days=ma_100_days, ma_200_days=ma_200_days)


selected_graph = st.selectbox(
    "Select Graph", ["Price vs MA50", "MA50 vs MA100", "MA100 vs MA200"]
)

if selected_graph == "Price vs MA50":
    plot_price_vs_ma50(data)
elif selected_graph == "MA50 vs MA100":
    plot_ma50_vs_ma100(data)
elif selected_graph == "MA100 vs MA200":
    plot_ma100_vs_ma200(data)

# ------------------------------------------------------------------------------------------------------------

st.text("")
st.subheader("Original Price vs Predicted Price")

if len(data) < 150:
    st.warning("Not enough data for prediction. Try another stock or increase the date range.")
else:
    data_train = pd.DataFrame(data["Close"][0 : int(len(data) * 0.80)])
    data_test = pd.DataFrame(data["Close"][int(len(data) * 0.80) : len(data)])

    scaler = MinMaxScaler(feature_range=(0, 1))

    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)

    data_test_scale = scaler.fit_transform(data_test)

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i - 100 : i])
        y.append(data_test_scale[i, 0])

    x = np.array(x)
    y = np.array(y)

    if len(x) == 0:
        st.warning("Not enough test data available for prediction.")
    else:
        predict = model.predict(x, verbose=0)

        scale_factor = 1 / scaler.scale_[0]
        predict = predict.reshape(-1) * scale_factor
        y = y * scale_factor

        fig4 = plt.figure(figsize=(8, 6))
        plt.plot(predict, label="Predicted Price")
        plt.plot(y, label="Original Price")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig4)
        plt.close(fig4)

# ------------------------------------------------------------------------------------------------------------

st.text("")
st.subheader("Fibonacci Retracement Levels")

latest_2_years_data = data.tail(2 * 365)

highest_value = float(latest_2_years_data["Close"].max())
lowest_value = float(latest_2_years_data["Close"].min())

highest_date = latest_2_years_data["Close"].idxmax()
lowest_date = latest_2_years_data["Close"].idxmin()

retracement_levels = {
    0: highest_value,
    23.6: highest_value - 0.236 * (highest_value - lowest_value),
    38.2: highest_value - 0.382 * (highest_value - lowest_value),
    50: (highest_value + lowest_value) / 2,
    61.8: highest_value - 0.618 * (highest_value - lowest_value),
    100: lowest_value,
}

fig = plt.figure(figsize=(10, 6))
plt.plot(latest_2_years_data.index, latest_2_years_data["Close"], label="Close Price")
plt.scatter(highest_date, highest_value, marker="o", label="Highest Value")
plt.scatter(lowest_date, lowest_value, marker="o", label="Lowest Value")
plt.plot([highest_date, lowest_date], [highest_value, lowest_value], linestyle="--")

for level, price in retracement_levels.items():
    plt.axhline(price, linestyle="--", label=f"{level}% Fibonacci")

plt.legend()
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Fibonacci Retracement Levels")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Recent News")

with col2:
    show_news = st.button(label="Show News")


def create_card(title, publisher, thumbnail_url, link):
    image_html = ""
    if thumbnail_url:
        image_html = f"""
        <div style="flex: 30%; height: 125px; margin: 10px; border-radius: 15px; overflow: hidden;">
            <img src="{thumbnail_url}" style="width: 100%; height: 100%; object-fit: cover;">
        </div>
        """

    card_html = f"""
    <div style="background-color: #F0F0F0; font-size: 0.875em; border-radius: 15px; padding: 10px; margin-top: 20px; box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 24px;">
        <div style="display: flex;">
            {image_html}
            <div style="flex: 70%; padding: 10px; color: black;">
                <h5>{title}</h5>
                <p>{publisher}</p>
                <a href="{link}" target="_blank" style="color: #009FFF;">Read more...</a>
            </div>
        </div>
    </div>
    """
    return card_html


if show_news:
    if not news:
        st.info("No recent news found for this stock.")
    else:
        for article in news[:5]:
            title = article.get("title", "No title")
            publisher = article.get("publisher", "Unknown publisher")
            link = article.get("link", "#")

            thumbnail_url = None
            thumbnail = article.get("thumbnail")

            if isinstance(thumbnail, dict):
                resolutions = thumbnail.get("resolutions", [])
                if resolutions:
                    thumbnail_url = resolutions[0].get("url")

            card_html = create_card(title, publisher, thumbnail_url, link)
            st.markdown(card_html, unsafe_allow_html=True)
