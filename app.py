import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import numpy as np
import requests
from datetime import datetime
from volatility_model import calculate_volatility, forecast_volatility
from llm_explainer import VolatilityExplainer


# Replace with your own NewsAPI key
NEWS_API_KEY = "c1f51b3b7bb24a2884a1d9665eeb4a72"

st.set_page_config(page_title="📈 Should I Buy Now?", layout="wide")
st.title("📈 Should I Buy Now?")

# --- USER INPUT ---
ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", value="AAPL")
period = st.selectbox("How much history to analyze?", ["6mo", "1y", "2y"], index=1)

# --- SLIDERS FOR CUSTOM THRESHOLDS ---
st.sidebar.header("🎛️ Custom Indicator Thresholds")
rsi_buy = st.sidebar.slider("RSI Buy Threshold (oversold)", 0, 50, 30)
macd_confidence = st.sidebar.slider("MACD Strength (above signal line)", 0.0, 5.0, 0.5)
volume_boost = st.sidebar.slider("Volume Spike Factor", 1.0, 3.0, 1.5)
min_score = st.sidebar.slider("Min Score for Buy", 1, 4, 3)

# --- NEWS SENTIMENT FUNCTION ---
def get_news_sentiment(ticker_symbol):
    url = f"https://newsapi.org/v2/everything?q={ticker_symbol}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        if not articles:
            return "Unknown", 0.0
        headlines = [a["title"] for a in articles]
        from textblob import TextBlob
        scores = [TextBlob(title).sentiment.polarity for title in headlines]
        avg_sentiment = np.mean(scores)
        label = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"
        return label, avg_sentiment
    except Exception as e:
        return "Unknown", 0.0

# --- MAIN LOGIC ---
if ticker_input:
    try:
        ticker = yf.Ticker(ticker_input.upper())
        hist = ticker.history(period=period)
        st.write("Raw DataFrame:", hist.head())


        if hist.empty:
            st.warning("No data available for this ticker.")
        else:
            hist.dropna(inplace=True)
            hist.reset_index(inplace=True)
            hist.columns = [col.capitalize() for col in hist.columns]


            # --- VOLATILITY METRICS ---
            hist = calculate_volatility(hist)

            # --- VOLATILITY FORECAST ---
            forecast = forecast_volatility(hist)

            # --- INDICATORS ---
            hist["RSI"] = RSIIndicator(close=hist["Close"]).rsi()
            hist["200_MA"] = SMAIndicator(close=hist["Close"], window=200).sma_indicator()
            macd = MACD(close=hist["Close"])
            hist["MACD"] = macd.macd()
            hist["Signal"] = macd.macd_signal()
            atr = AverageTrueRange(high=hist["High"], low=hist["Low"], close=hist["Close"])
            hist["ATR"] = atr.average_true_range()
            hist["Volume_MA"] = hist["Volume"].rolling(window=20).mean()

            # --- FUNDAMENTALS ---
            info = ticker.info
            pe = info.get("trailingPE", "N/A")
            eps = info.get("earningsQuarterlyGrowth", "N/A")
            market_cap = info.get("marketCap", "N/A")

            # --- LATEST DATA ---
            latest = hist.iloc[-1]
            current_price = latest["Close"]
            ma_200 = latest["200_MA"]
            rsi = latest["RSI"]
            macd_val = latest["MACD"]
            signal_val = latest["Signal"]
            atr_val = latest["ATR"]
            volume = latest["Volume"]
            avg_volume = latest["Volume_MA"]

            stop_loss = round(current_price - atr_val * 1.5, 2)
            take_profit = round(current_price + atr_val * 2, 2)

            # --- SCORING SYSTEM ---
            score = 0
            if current_price > ma_200:
                score += 1
            if macd_val - signal_val > macd_confidence:
                score += 1
            if rsi <= rsi_buy:
                score += 1
            if volume > avg_volume * volume_boost:
                score += 1

            # --- LLM EXPLANATION ---
            vol_level = round(hist["Rolling Volatility"].iloc[-1], 2)
            explainer = VolatilityExplainer(model_name="qwen2.5:7b")
            explanation = explainer.generate_explanation(vol_level)


            # --- NEWS SENTIMENT ---
            sentiment_label, sentiment_score = get_news_sentiment(ticker_input.upper())

            # --- RECOMMENDATION ---
            if score >= min_score and sentiment_label != "Negative":
                recommendation = "✅ Strong Buy"
                signal_color = "green"
            elif score == min_score - 1:
                recommendation = "👀 Watch"
                signal_color = "orange"
            else:
                recommendation = "⚠️ Avoid for now"
                signal_color = "gray"

            # --- SUMMARY ---
            st.subheader(f"📊 Technical Summary for {ticker_input.upper()}")
            st.markdown(f"**Current Price:** £{round(current_price, 2)}")
            st.markdown(f"**200-Day MA:** £{round(ma_200, 2)}")
            st.markdown(f"**RSI (14):** {round(rsi, 2)}")
            st.markdown(f"**MACD:** {round(macd_val, 2)} | **Signal:** {round(signal_val, 2)}")
            st.markdown(f"**Volume:** {volume:,.0f} vs Avg {int(avg_volume):,}")
            st.markdown(f"**ATR:** {round(atr_val, 2)} → Stop-Loss: £{stop_loss}, Take-Profit: £{take_profit}")
            # --- VOLATILITY EXPLANATION ---
            st.subheader("🧠 Volatility Explanation")
            st.markdown(explanation)
            st.markdown(f"**News Sentiment:** *{sentiment_label}* ({round(sentiment_score, 2)})")
            st.markdown(f"### Recommendation: <span style='color:{signal_color}; font-size:28px'>**{recommendation}**</span>", unsafe_allow_html=True)

            # --- FUNDAMENTALS ---
            st.subheader("📉 Fundamentals")
            st.markdown(f"**P/E Ratio:** {pe}")
            st.markdown(f"**EPS Growth:** {eps}")
            st.markdown(f"**Market Cap:** {market_cap:,}" if market_cap != "N/A" else "**Market Cap:** N/A")

            # --- PRICE CHART ---
            st.subheader("📈 Price vs 200-Day MA")
            chart_df = hist[["Close", "200_MA"]].copy()
            chart_df["Stop Loss"] = stop_loss
            chart_df["Take Profit"] = take_profit
            st.line_chart(chart_df)

            # --- RSI & MACD ---
            st.subheader("📊 RSI & MACD")
            st.line_chart(hist[["RSI"]])
            st.line_chart(hist[["MACD", "Signal"]])
            # --- VOLATILITY FORECAST CHART ---
            st.subheader("📉 Forecasted Volatility")
            st.line_chart(forecast[["ds", "yhat"]].set_index("ds"))

            # --- ENTRY INFO ---
            st.info(f"**Suggested Entry:** ~£{round(current_price, 2)} | Stop: £{stop_loss} | Target: £{take_profit}")

            # --- BACKTESTING ---
            st.subheader("🧪 Backtest: How Past 'Buy' Signals Performed")
            backtest_results = []
            for i in range(len(hist) - 10):
                row = hist.iloc[i]
                if (
                    row["Close"] > row["200_MA"] and
                    row["MACD"] > row["Signal"] and
                    row["RSI"] <= rsi_buy
                ):
                    entry_price = row["Close"]
                    exit_price = hist["Close"].iloc[i + 5]
                    change_pct = (exit_price - entry_price) / entry_price * 100
                    backtest_results.append(change_pct)

            if backtest_results:
                avg_return = round(np.mean(backtest_results), 2)
                win_rate = round(sum(x > 0 for x in backtest_results) / len(backtest_results) * 100, 2)
                st.markdown(f"**Simulated Trades:** {len(backtest_results)}")
                st.markdown(f"**Average 5-Day Return:** {avg_return}%")
                st.markdown(f"**Win Rate:** {win_rate}%")
            else:
                st.info("⚠️ No historical setups matched current strategy criteria for backtesting.")

            # --- EXPLAINER ---
            with st.expander("❓ What do these indicators mean?"):
                st.markdown("""
                - **RSI (Relative Strength Index):** Measures momentum. Below 30 = oversold (buy zone), above 70 = overbought.
                - **MACD (Moving Average Convergence Divergence):** Trend direction. Bullish when MACD > Signal.
                - **200-Day Moving Average:** If price is above it, the long-term trend is bullish.
                - **ATR (Average True Range):** Volatility indicator. Used for stop-loss and profit targets.
                - **Volume:** Tracks activity. Spikes suggest strong buying/selling interest.
                - **News Sentiment:** Pulls latest news and estimates tone using NLP.
                - **Backtesting:** Simulates previous trades using your conditions to test strategy accuracy.
                """)
    except Exception as e:
        st.error(f"Error fetching or processing data: {e}")

