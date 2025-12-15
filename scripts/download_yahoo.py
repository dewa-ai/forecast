import yfinance as yf
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

pairs = {
    "usd_idr": "USDIDR=X",
    "usd_twd": "USDTWD=X",
    "usd_eur": "USDEUR=X",
}

start = "2010-01-01"
end = None  # until today

for name, ticker in pairs.items():
    df = yf.download(ticker, start=start, end=end, interval="1d")
    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["date", "value"]
    df = df.dropna()

    out = f"data/{name}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out} ({len(df)} rows)")
