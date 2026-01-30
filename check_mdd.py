import pandas as pd
import numpy as np

def calculate_mdd(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

# 1. Strategy MDD (OOS)
try:
    strat_returns = pd.read_csv('output/robust/wfa_returns.csv', index_col=0, parse_dates=True).iloc[:, 0]
    strat_mdd = calculate_mdd(strat_returns)
    print(f"Strategy (WFA OOS) MDD: {strat_mdd:.2%}")
except Exception as e:
    print(f"Error reading strategy returns: {e}")

# 2. Benchmark MDD (Equal Weighted over full history)
try:
    master_returns = pd.read_csv('output/eda_vn30/data/master_cleaned_returns.csv', index_col=0, parse_dates=True)
    ew_returns = master_returns.mean(axis=1)
    benchmark_mdd = calculate_mdd(ew_returns)
    print(f"Buy & Hold (EW VN30 Full) MDD: {benchmark_mdd:.2%}")
except Exception as e:
    print(f"Error reading master returns: {e}")
