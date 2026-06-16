import yfinance as yf
import pandas as pd

# 6 highly correlated assets — tech sector
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]

def fetch_prices(start="2020-01-01", end="2024-01-01"):
    print("Fetching price data...")
    raw = yf.download(TICKERS, start=start, end=end)["Close"]
    prices = raw.dropna()
    print(f"Got {len(prices)} trading days across {len(TICKERS)} assets")
    print(prices.tail())
    return prices

if __name__ == "__main__":
    df = fetch_prices()
    df.to_csv("data/prices.csv")
    print("\nSaved to data/prices.csv ✅")