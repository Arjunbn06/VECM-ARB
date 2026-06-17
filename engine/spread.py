import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from engine.johansen import run_johansen

def build_spread(prices, beta):
    """
    construct the spread using hedge ratios from johansen
    spread = prices × beta (first cointegrating vector)
    """
    # use only first cointegrating vector
    spread = prices.values @ beta[:, 0]
    spread = pd.Series(spread, index=prices.index, name="spread")
    return spread

def plot_spread(spread):
    # calculate z-score
    mean = spread.mean()
    std  = spread.std()
    zscore = (spread - mean) / std

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # plot spread
    ax1.plot(spread, color="steelblue", linewidth=0.8)
    ax1.axhline(mean, color="black", linestyle="--", linewidth=1, label="mean")
    ax1.set_title("Cointegrated Spread")
    ax1.set_ylabel("Spread Value")
    ax1.legend()

    # plot z-score
    ax2.plot(zscore, color="darkorange", linewidth=0.8)
    ax2.axhline(0,  color="black",  linestyle="--", linewidth=1)
    ax2.axhline(2,  color="red",    linestyle="--", linewidth=1, label="+2 (sell)")
    ax2.axhline(-2, color="green",  linestyle="--", linewidth=1, label="-2 (buy)")
    ax2.set_title("Z-Score of Spread")
    ax2.set_ylabel("Z-Score")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("data/spread.png", dpi=150)
    print("Saved spread chart to data/spread.png ✅")
    plt.show()

if __name__ == "__main__":
    prices = pd.read_csv(
        "data/prices.csv",
        index_col=0,
        parse_dates=True
    )

    results = run_johansen(prices, significance="90%")
    spread  = build_spread(prices, results["beta"])
    plot_spread(spread)

    print(f"\nSpread stats:")
    print(f"  mean : {spread.mean():.4f}")
    print(f"  std  : {spread.std():.4f}")
    print(f"  min  : {spread.min():.4f}")
    print(f"  max  : {spread.max():.4f}")