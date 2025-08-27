import pandas as pd
from ta.volatility import AverageTrueRange
from prophet import Prophet

def calculate_volatility (df, window=14):
    df["Daily Return"] = df["Close"].pct_change()
    df["Rolling Volatility"] = df["Daily Return"].rolling(window=window).std() * 100
    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"])
    df["ATR"] = atr.average_true_range()
    return df


def forecast_volatility(df, periods=7):
    df = df[["Date", "Daily Return"]].dropna().rename(columns={"Date": "ds", "Daily Return": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast