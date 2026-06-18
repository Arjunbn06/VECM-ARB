import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import VECM
from engine.johansen import run_johansen

def fit_vecm(prices, rank):
    model = VECM(
        prices,
        k_ar_diff=1,
        coint_rank=rank,
        deterministic="n"
    )
    result = model.fit()
    return result

def analyze_alpha(result, tickers):
    alpha = result.alpha

    print("\n" + "="*50)
    print("SPEED OF ADJUSTMENT (α)")
    print("="*50)
    print("negative α = mean reverting (GOOD)")
    print("positive α = explosive (BAD)")
    print()

    for i, ticker in enumerate(tickers):
        a = alpha[i, 0]
        direction = "✅ mean reverting" if a < 0 else "❌ explosive"
        days_to_revert = abs(round(1/a)) if a != 0 else float("inf")
        print(f"{ticker:6s}: α = {a:8.4f}  →  {direction}  (~{abs(days_to_revert)} days to revert)")

    return alpha

def plot_vecm_fit(result, prices):
    fitted = result.fittedvalues
    actual = prices.diff().dropna()

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, ticker in enumerate(prices.columns):
        axes[i].plot(actual.index[1:], actual[ticker].iloc[1:],
                    color="steelblue", linewidth=0.6, label="actual", alpha=0.7)
        axes[i].plot(actual.index[1:], fitted[:, i],
                    color="red", linewidth=0.6, label="fitted", alpha=0.7)
        axes[i].set_title(f"{ticker} — actual vs fitted")
        axes[i].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("data/vecm_fit.png", dpi=150)
    print("\nSaved VECM fit chart to data/vecm_fit.png ✅")
    plt.show()

def compute_half_life(alpha):
    strongest_alpha = alpha[np.argmin(alpha[:, 0]), 0]
    half_life = np.log(0.5) / np.log(1 + strongest_alpha)

    print(f"\n{'='*50}")
    print(f"HALF LIFE OF MEAN REVERSION")
    print(f"{'='*50}")
    print(f"Strongest α   : {strongest_alpha:.4f}")
    print(f"Half life     : {abs(half_life):.1f} days")

    if abs(half_life) < 10:
    print("Signal        : ⚡ very fast reversion — strong signal")
elif abs(half_life) < 30:
    print("Signal        : ✅ good reversion speed — tradeable")
elif abs(half_life) < 60:
    print("Signal        : ⚠️  slow but institutional — tradeable with sizing")
else:
    print("Signal        : ❌ too slow — not tradeable")
    return half_life

if __name__ == "__main__":
    prices = pd.read_csv(
        "data/prices.csv",
        index_col=0,
        parse_dates=True
    )

    johansen_results = run_johansen(prices, significance="90%")
    rank = johansen_results["rank"]

    print("\nFitting VECM model...")
    vecm_result = fit_vecm(prices, rank)

    alpha = analyze_alpha(vecm_result, list(prices.columns))

    half_life = compute_half_life(alpha)

    plot_vecm_fit(vecm_result, prices)

    print("\n✅ VECM model complete")
    print(f"   rank      : {rank}")
    print(f"   half life : {abs(half_life):.1f} days")