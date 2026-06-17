import yfinance as yf
import pandas as pd
import numpy as np


TICKERS = ["XLE", "XLF", "XLV", "XLI", "XLK", "XLU"]

def fetch_prices(start="2010-01-01", end="2020-01-01"):
    print("Fetching price data...")
    raw = yf.download(TICKERS, start=start, end=end)["Close"]
    prices = raw.dropna()
    
    
    log_prices = np.log(prices)
    
    print(f"Got {len(log_prices)} trading days across {len(TICKERS)} assets")
    print(log_prices.tail())
    return log_prices

if __name__ == "__main__":
    df = fetch_prices()
    df.to_csv("data/prices.csv")
    print("\nSaved to data/prices.csv ✅")