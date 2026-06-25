# VECM-ARB — Cross-Asset Statistical Arbitrage Engine

## Overview
A multivariate statistical arbitrage system built on Vector Error Correction Models (VECM) and Johansen cointegration theory. Identifies stable long-run equilibrium relationships across 6 S&P 500 sector ETFs and generates market-neutral trading signals when the spread deviates beyond 2 standard deviations.

## Backtest Results (2005–2020)
| Metric | Value |
|---|---|
| Sharpe Ratio | 0.425 |
| Total Return | 11.33% |
| Annual Return | 0.72% |
| Max Drawdown | -2.59% |
| Win Rate | 51.82% |
| Cointegrating Rank | 1 (90% significance) |

## System Architecture|
## Key Concepts
- **Johansen Test** — multivariate cointegration test via eigenvalue decomposition across N assets simultaneously
- **Hedge Ratios (β)** — eigenvectors defining the market-neutral portfolio weights
- **VECM** — models error correction speed (α) — how fast spread reverts to equilibrium
- **Half Life** — log-decay formula: `log(0.5) / log(1+α)` — days to 50% reversion
- **Z-Score Signals** — entry at ±2σ, exit at 0, rolling 60-day window

## Assets
S&P 500 Sector ETFs: XLE, XLU, XLF, XLV, XLI, XLB

## Tech Stack
- Python 3.9+
- statsmodels — Johansen test, VECM fitting
- pandas / numpy — data pipeline
- matplotlib — visualization
- yfinance — market data

## Run
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
