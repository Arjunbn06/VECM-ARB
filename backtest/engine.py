import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from engine.johansen import run_johansen
from engine.spread import build_spread
from engine.signals import compute_zscore, generate_signals

def backtest(prices, signals, beta):
    returns = prices.pct_change().dropna()
    signals_aligned = signals.reindex(returns.index).fillna(0)
    beta_vec = beta[:, 0]
    beta_vec = beta_vec / np.sum(np.abs(beta_vec))
    daily_pnl = pd.Series(index=returns.index, dtype=float)
    for i in range(len(returns)):
        pos = signals_aligned.iloc[i]
        if pos == 0:
            daily_pnl.iloc[i] = 0
        else:
            asset_returns = returns.iloc[i].values
            pnl = pos * np.dot(beta_vec, asset_returns)
            daily_pnl.iloc[i] = pnl
    return daily_pnl

def compute_metrics(daily_pnl):
    daily_pnl = daily_pnl.fillna(0)
    cumulative_pnl = (1 + daily_pnl).cumprod()
    total_return = cumulative_pnl.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(daily_pnl)) - 1
    volatility = daily_pnl.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility != 0 else 0
    rolling_max = cumulative_pnl.cummax()
    drawdown = (cumulative_pnl - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    winning_days = (daily_pnl > 0).sum()
    losing_days = (daily_pnl < 0).sum()
    win_rate = winning_days / (winning_days + losing_days)
    print(f"\n{'='*50}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"Total Return   : {total_return*100:.2f}%")
    print(f"Annual Return  : {annual_return*100:.2f}%")
    print(f"Volatility     : {volatility*100:.2f}%")
    print(f"Sharpe Ratio   : {sharpe:.3f}")
    print(f"Max Drawdown   : {max_drawdown*100:.2f}%")
    print(f"Win Rate       : {win_rate*100:.2f}%")
    print(f"Winning Days   : {winning_days}")
    print(f"Losing Days    : {losing_days}")
    return {
        "cumulative_pnl" : cumulative_pnl,
        "drawdown"       : drawdown,
        "sharpe"         : sharpe,
        "total_return"   : total_return,
        "max_drawdown"   : max_drawdown,
        "win_rate"       : win_rate
    }

def backtest_all_relationships(prices, beta, rank):
    all_pnl = []
    returns = prices.pct_change().dropna()
    for r in range(rank):
        beta_vec = beta[:, r]
        beta_vec = beta_vec / np.sum(np.abs(beta_vec))
        spread = pd.Series(
            prices.values @ beta[:, r],
            index=prices.index
        )
        zscore = compute_zscore(spread, window=60)
        signals = generate_signals(zscore, entry=2.0, exit=0.0)
        signals_aligned = signals.reindex(returns.index).fillna(0)
        daily_pnl = pd.Series(index=returns.index, dtype=float)
        for i in range(len(returns)):
            pos = signals_aligned.iloc[i]
            if pos == 0:
                daily_pnl.iloc[i] = 0
            else:
                pnl = pos * np.dot(beta_vec, returns.iloc[i].values)
                daily_pnl.iloc[i] = pnl
        all_pnl.append(daily_pnl)
    combined = pd.concat(all_pnl, axis=1).fillna(0).mean(axis=1)
    return combined

def plot_backtest(metrics):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.plot(metrics["cumulative_pnl"], color="green", linewidth=1)
    ax1.axhline(1, color="black", linestyle="--", linewidth=0.8)
    ax1.set_title(f"Cumulative PnL - Sharpe: {metrics['sharpe']:.3f}")
    ax1.set_ylabel("Portfolio Value")
    ax2.fill_between(metrics["drawdown"].index,
                     metrics["drawdown"].values, 0,
                     color="red", alpha=0.4)
    ax2.set_title(f"Drawdown - Max: {metrics['max_drawdown']*100:.2f}%")
    ax2.set_ylabel("Drawdown %")
    plt.tight_layout()
    plt.savefig("data/backtest.png", dpi=150)
    print("Saved backtest chart to data/backtest.png")
    plt.show()

if __name__ == "__main__":
    prices = pd.read_csv(
        "data/prices.csv",
        index_col=0,
        parse_dates=True
    )
    johansen_results = run_johansen(prices, significance="90%")
    rank = johansen_results["rank"]
    beta = johansen_results["beta"]
    print(f"\nBacktesting all {rank} cointegrating relationships...")
    daily_pnl = backtest_all_relationships(prices, beta, rank)
    metrics = compute_metrics(daily_pnl)
    plot_backtest(metrics)
