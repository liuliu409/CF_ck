"""
Walk-Forward Backtest Engine
============================
Implements a rolling window rebalancing framework with:
1. Turn-over constrained MVO using CVXPY
2. Out-of-Sample path concatenation
3. Robust estimation on training windows
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import datetime
from typing import Dict, List, Tuple
import os

from config import (
    VN30_SYMBOLS, START_DATE, END_DATE, RISK_FREE_RATE,
    MIN_WEIGHT, MAX_WEIGHT, TURNOVER_LIMIT, REBALANCE_FREQ,
    LOOKBACK_WINDOW, FORECAST_HORIZON, TRANSACTION_COST_MODEL,
    FACTOR_WEIGHTS
)
from robust_utils import (
    apply_ledoit_wolf, estimate_ic_premium, 
    map_ic_to_returns, winsorize_scores,
    calculate_quality_factor, calculate_performance_metrics
)
from part1_multifactor_model import (
    calculate_size_factor, calculate_value_factor, 
    calculate_momentum_factor, calculate_composite_score
)

def run_wfa_backtest(all_data: Dict[str, pd.DataFrame]):
    """
    Main loop for Walk-Forward Analysis.
    """
    # 1. Prepare Returns Data
    returns_dict = {s: df.set_index('date')['close'].pct_change() for s, df in all_data.items()}
    full_returns = pd.DataFrame(returns_dict).dropna()
    
    # 2. Define Rebalance Dates
    # Frequency matches REBALANCE_FREQ (e.g., 'Q' for Quarter End)
    rebalance_dates = pd.date_range(start=full_returns.index[252 * LOOKBACK_WINDOW], 
                                   end=full_returns.index[-1], 
                                   freq=REBALANCE_FREQ)
    
    portfolio_weights = []
    portfolio_returns = []
    current_weights = np.array([1.0/len(VN30_SYMBOLS)] * len(VN30_SYMBOLS)) # Initial: Equal Weighted
    
    print(f"\nStarting Walk-Forward Analysis ({len(rebalance_dates)} periods)...")
    
    for i in range(len(rebalance_dates)-1):
        t_start = rebalance_dates[i]
        t_end = rebalance_dates[i+1]
        
        # Training Window: Look back X years from t_start
        train_start = t_start - pd.DateOffset(years=LOOKBACK_WINDOW)
        train_returns = full_returns.loc[train_start:t_start]
        
        # --- A. FACTOR CALCULATION (Train) ---
        snapshot_data = {s: df[df['date'] <= t_start] for s, df in all_data.items()}
        
        size = calculate_size_factor(snapshot_data)
        value = calculate_value_factor(snapshot_data)
        mome = calculate_momentum_factor(snapshot_data)
        qual = calculate_quality_factor(snapshot_data)
        
        # Apply Sector Neutralization to isolate Pure Alpha
        from robust_utils import neutralize_by_sector, standardize
        
        size_pure = neutralize_by_sector(size)
        value_pure = neutralize_by_sector(value)
        mome_pure = neutralize_by_sector(mome)
        qual_pure = neutralize_by_sector(qual)
        
        # Aggregate Neutralized Scores
        weights = FACTOR_WEIGHTS
        composite_score = (
            weights['size'] * size_pure +
            weights['value'] * value_pure +
            weights['momentum'] * mome_pure +
            weights['quality'] * qual_pure
        )
        
        scores = winsorize_scores(standardize(composite_score))
        
        # --- B. ROBUST ESTIMATION (Train) ---
        cov_matrix = apply_ledoit_wolf(train_returns)
        ic = estimate_ic_premium(scores, train_returns, horizon=FORECAST_HORIZON)
        
        # Volatility estimation for IC mapping
        asset_vols = train_returns.std() * np.sqrt(252)
        expected_returns = map_ic_to_returns(scores, asset_vols, ic)
        
        # --- C. CONSTRAINED OPTIMIZATION (t_start) ---
        n = len(VN30_SYMBOLS)
        w = cp.Variable(n)
        
        objective = cp.Maximize(expected_returns.values @ w - 0.5 * cp.quad_form(w, cov_matrix.values))
        
        constraints = [
            cp.sum(w) == 1,
            w >= MIN_WEIGHT,
            w <= MAX_WEIGHT,
            cp.norm(w - current_weights, 1) <= TURNOVER_LIMIT # Turnover Constraint
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        if prob.status != cp.OPTIMAL:
            print(f"  ⚠ Optimization failed at {t_start}, using current weights")
            opt_weights = current_weights
        else:
            opt_weights = w.value
            
        current_weights = opt_weights
        
        # --- D. PERFORMANCE MEASUREMENT (OOS Path) ---
        # Calculate returns from t_start to t_end using the weights found at t_start
        oos_returns = full_returns.loc[t_start:t_end]
        strategy_returns = oos_returns @ opt_weights
        
        # Subtract transaction costs for rebalance
        strategy_returns.iloc[0] -= np.sum(np.abs(opt_weights - current_weights)) * TRANSACTION_COST_MODEL
        
        portfolio_returns.append(strategy_returns)
        portfolio_weights.append(pd.Series(opt_weights, index=VN30_SYMBOLS, name=t_start))
        
        print(f"  Period {i+1}/{len(rebalance_dates)-1}: {t_start.date()} to {t_end.date()} | IC: {ic:.3f} | Ret: {strategy_returns.mean()*252:.2%}")

    # Combine Results
    all_strategy_returns = pd.concat(portfolio_returns)
    all_strategy_returns = all_strategy_returns[~all_strategy_returns.index.duplicated(keep='first')]
    
    return all_strategy_returns, pd.DataFrame(portfolio_weights)

if __name__ == "__main__":
    from part1_multifactor_model import fetch_all_stocks
    from config import START_DATE, END_DATE
    
    print("Running Robust WFA Verification...")
    data = fetch_all_stocks(VN30_SYMBOLS, START_DATE, END_DATE)
    strategy_returns, weights_history = run_wfa_backtest(data)
    
    # Save Results
    os.makedirs('output/robust', exist_ok=True)
    strategy_returns.to_csv('output/robust/wfa_returns.csv')
    weights_history.to_csv('output/robust/wfa_weights.csv')
    
    print("\n✓ Robust WFA Backtest Completed")
    metrics = calculate_performance_metrics(strategy_returns, rf_rate=0.02)
    print(f"Annualized Return: {metrics['ann_return']:.2%}")
    print(f"Annualized Vol: {metrics['ann_vol']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio: {metrics['calmar']:.2f}")
