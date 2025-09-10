import pandas as pd
from ta.volatility import AverageTrueRange
from prophet import Prophet

def calculate_volatility (df, window=14):
    df["Daily Return"] = df["Close"].pct_change()
    df["Rolling Volatility"] = df["Daily Return"].rolling(window=window).std() * 100
    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"])
    df["ATR"] = atr.average_true_range()
    return df


def forecast_volatility(df, horizon=7):
    df = df.copy()
    df["ds"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df["y"] = df["Rolling Volatility"]

    model = Prophet()
    model.fit(df[["ds", "y"]])

    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
