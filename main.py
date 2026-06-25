import pandas as pd
from data.fetch import fetch_prices
from engine.johansen import run_johansen
from engine.spread import build_spread
from engine.signals import compute_zscore, generate_signals
from backtest.engine import backtest_all_relationships, compute_metrics, plot_backtest

if __name__ == "__main__":
    print("VECM-ARB | Distributed Cross-Asset Statistical Arbitrage Engine")
    print("="*60)

    print("\n[1/5] Fetching market data...")
    prices = fetch_prices()
    prices.to_csv("data/prices.csv")

    print("\n[2/5] Running Johansen cointegration test...")
    johansen_results = run_johansen(prices, significance="90%")
    rank = johansen_results["rank"]
    beta = johansen_results["beta"]

    print("\n[3/5] Building spread and z-score...")
    spread = build_spread(prices, beta)
    zscore = compute_zscore(spread, window=60)

    print("\n[4/5] Generating trading signals...")
    signals = generate_signals(zscore, entry=2.0, exit=0.0)
    print(f"Long days  : {(signals==1).sum()}")
    print(f"Short days : {(signals==-1).sum()}")
    print(f"Flat days  : {(signals==0).sum()}")

    print("\n[5/5] Running backtest...")
    daily_pnl = backtest_all_relationships(prices, beta, rank)
    metrics = compute_metrics(daily_pnl)
    plot_backtest(metrics)

    print("\nDone. Check data/ folder for all charts.")
