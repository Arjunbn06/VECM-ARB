import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from engine.johansen import run_johansen
from engine.spread import build_spread

def compute_zscore(spread, window=60):
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    zscore = (spread - mean) / std
    return zscore.dropna()

def generate_signals(zscore, entry=2.0, exit=0.0):
    signals = pd.Series(index=zscore.index, dtype=float)
    signals[:] = 0
    position = 0
    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if position == 0:
            if z > entry:
                position = 1
            elif z < -entry:
                position = -1
        elif position == 1:
            if z <= exit:
                position = 0
        elif position == -1:
            if z >= exit:
                position = 0
        signals.iloc[i] = position
    return signals

def plot_signals(zscore, signals):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.plot(zscore, color="darkorange", linewidth=0.8, label="z-score")
    ax1.axhline(2,  color="red",   linestyle="--", linewidth=1, label="entry long")
    ax1.axhline(-2, color="green", linestyle="--", linewidth=1, label="entry short")
    ax1.axhline(0,  color="black", linestyle="--", linewidth=0.8, label="exit")
    ax1.set_title("Z-Score with Trade Signals")
    ax1.legend()
    ax2.plot(signals, color="purple", linewidth=0.8, label="position")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax2.set_title("Position (1=long, -1=short, 0=flat)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("data/signals.png", dpi=150)
    print("Saved signals chart to data/signals.png")
    plt.show()

if __name__ == "__main__":
    prices = pd.read_csv(
        "data/prices.csv",
        index_col=0,
        parse_dates=True
    )
    johansen_results = run_johansen(prices, significance="90%")
    spread = build_spread(prices, johansen_results["beta"])
    zscore = compute_zscore(spread, window=60)
    signals = generate_signals(zscore, entry=2.0, exit=0.0)
    long_trades  = (signals == 1).sum()
    short_trades = (signals == -1).sum()
    flat_days    = (signals == 0).sum()
    print(f"\nSIGNAL SUMMARY")
    print(f"{'='*40}")
    print(f"Long days  : {long_trades}")
    print(f"Short days : {short_trades}")
    print(f"Flat days  : {flat_days}")
    plot_signals(zscore, signals)
