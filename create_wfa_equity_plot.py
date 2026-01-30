import pandas as pd
import matplotlib.pyplot as plt
import os

def create_equity_curve():
    benchmarks_path = 'output/robust/wfa_benchmarks_combined.csv'
    if not os.path.exists(benchmarks_path):
        print(f"File {benchmarks_path} not found. Run backtest_engine.py first.")
        return

    # Load results
    benchmarks = pd.read_csv(benchmarks_path, index_col=0, parse_dates=True)
    
    # Calculate cumulative returns (Equity Curves)
    equity_curves = (1 + benchmarks).cumprod()
    
    # Plotting
    plt.figure(figsize=(14, 8), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Strategy (Primary focus)
    plt.plot(equity_curves.index, equity_curves['Strategy'], 
             label='VN30 Alpha Engine (WFA)', color='#1f77b4', linewidth=3)
    
    # Passive Buy & Hold
    plt.plot(equity_curves.index, equity_curves['Buy_and_Hold'], 
             label='Buy & Hold VN30 (Equal Weighted)', color='#7f7f7f', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Risk-Matched Benchmark
    plt.plot(equity_curves.index, equity_curves['Vol_Scaled_B&H'], 
             label='Volatility-Scaled Buy & Hold (Risk-Matched)', color='#d62728', linestyle=':', linewidth=2)
    
    # Styling
    plt.title('Strategy vs. Benchmarks: Walk-Forward Equity Curves', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Growth ($1.0 Initial)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    
    # Add subtle annotation about risk-matching
    plt.annotate('Red curve is Buy & Hold scaled to Strategy Volatility', 
                 xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10, 
                 fontstyle='italic', bbox=dict(facecolor='white', alpha=0.6))

    # Save
    output_path = 'output/robust/equity_curve.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig('equity_curve.png') # Copy to root for easy include
    print(f"Updated equity curve saved to {output_path}")

if __name__ == "__main__":
    create_equity_curve()
