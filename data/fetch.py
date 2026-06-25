import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ["XLE", "XLU", "XLF", "XLV", "XLI", "XLB"]

def fetch_prices(start="2005-01-01", end="2020-01-01"):
    print("Fetching price data...")
    raw = yf.download(TICKERS, start=start, end=end)["Close"]
    prices = raw.dropna()
    log_prices = np.log(prices)
    print(f"Got {len(log_prices)} days of data")
    return log_prices

if __name__ == "__main__":
    df = fetch_prices()
    df.to_csv("data/prices.csv")
    print("Saved to data/prices.csv")
