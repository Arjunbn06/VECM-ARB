import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def run_johansen(prices: pd.DataFrame, significance: str = "95%"):
    """
    Run Johansen cointegration test on price data.
    
    prices     : DataFrame of asset prices (rows=dates, cols=tickers)
    significance: confidence level — "90%", "95%", or "99%"
    
    returns: dict with eigenvalues, eigenvectors (beta), and rank
    """
    
    # map significance to Johansen critical value index
    sig_map = {"90%": 0, "95%": 1, "99%": 2}
    sig_idx = sig_map[significance]
    
    # run the test
    # det_order=0 means we assume no deterministic trend
    # k_ar_diff=1 means we use 1 lag
    result = coint_johansen(prices, det_order=0, k_ar_diff=1)
    
    # how many cointegrating relationships exist?
    # we check trace statistic against critical value
    trace_stats = result.lr1        # trace statistics
    crit_vals   = result.cvt        # critical values
    
    rank = 0
    for i in range(len(trace_stats)):
        if trace_stats[i] > crit_vals[i, sig_idx]:
            rank += 1
    
    print(f"\n{'='*50}")
    print(f"JOHANSEN TEST RESULTS ({significance} significance)")
    print(f"{'='*50}")
    print(f"Trace Statistics : {np.round(trace_stats, 2)}")
    print(f"Critical Values  : {np.round(crit_vals[:, sig_idx], 2)}")
    print(f"\nCointegrating relationships found: {rank}")
    
    if rank == 0:
        print("WARNING: No cointegration found!")
        return None
    
    # eigenvectors = the beta weights (hedge ratios)
    # each column is one cointegrating relationship
    beta = result.evec[:, :rank]
    
    print(f"\nHedge Ratios (β) — first relationship:")
    for ticker, weight in zip(prices.columns, beta[:, 0]):
        print(f"  {ticker}: {weight:.4f}")
    
    return {
        "rank"        : rank,
        "beta"        : beta,
        "eigenvalues" : result.eig,
        "trace_stats" : trace_stats,
        "crit_vals"   : crit_vals
    }

if __name__ == "__main__":
    # load the prices we fetched on Day 1
    prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    print(f"Loaded {len(prices)} days of data for {list(prices.columns)}")
    
    results = run_johansen(prices, significance="90%")