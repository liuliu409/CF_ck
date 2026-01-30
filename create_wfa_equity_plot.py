import pandas as pd
import matplotlib.pyplot as plt
import os

def create_equity_curve():
    returns_path = 'output/robust/wfa_returns.csv'
    if not os.path.exists(returns_path):
        print(f"File {returns_path} not found. Run backtest_engine.py first.")
        return

    # Load results
    strat_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    # Calculate cumulative returns (Equity Curve)
    equity_curve = (1 + strat_returns).cumprod()
    
    # Plotting
    plt.figure(figsize=(12, 7), dpi=300)
    plt.plot(equity_curve.index, equity_curve.values, label='VN30 Alpha Engine (WFA)', color='navy', linewidth=2.5)
    
    # Styling
    plt.title('Strategy Performance: Walk-Forward Equity Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Growth ($1.0 Initial)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Save
    output_path = 'output/robust/equity_curve.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig('equity_curve.png', bbox_inches='tight') # Copy to root for easy include
    print(f"Equity curve saved to {output_path}")

if __name__ == "__main__":
    create_equity_curve()
