import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from xnoapi import client
from xnoapi.vn.data.stocks import Quote
from datetime import datetime

# Import configuration
from config import XNOAPI_KEY, VN30_SYMBOLS, START_DATE

# Initialize API
client(apikey=XNOAPI_KEY)

def fetch_returns(symbols, limit=5):
    all_returns = []
    print(f"Fetching data for first {limit} stocks to show Winsorization impact...")
    for sym in symbols[:limit]:
        try:
            quote = Quote(sym)
            data = quote.history(start=START_DATE, interval="1D")
            df = pd.DataFrame(data)
            if not df.empty and 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                # Store as (Stock, Raw Return)
                temp_df = pd.DataFrame({'Ticker': sym, 'Raw_Return': returns})
                all_returns.append(temp_df)
                print(f"  ✓ {sym}")
        except Exception as e:
            print(f"  ✗ Error fetching {sym}: {e}")
    return pd.concat(all_returns) if all_returns else pd.DataFrame()

def plot_winsorization(df):
    if df.empty:
        print("No data to plot.")
        return

    # Apply Winsorization at 1% and 99%
    df['Winsorized_Return'] = winsorize(df['Raw_Return'], limits=[0.01, 0.01])

    plt.figure(figsize=(12, 8), dpi=300)
    
    # Identify indices where changes occurred (extreme outliers)
    changed = df['Raw_Return'] != df['Winsorized_Return']
    
    # Plot original outliers in red
    plt.scatter(df.index[changed], df.loc[changed, 'Raw_Return'], 
                color='red', alpha=0.5, label='Original Outlier', s=50, marker='x')
    
    # Plot winsorized values in green
    plt.scatter(df.index[changed], df.loc[changed, 'Winsorized_Return'], 
                color='green', alpha=0.8, label='Winsorized (Folded In)', s=60, marker='o')
    
    # Plot normal data in blue
    plt.scatter(df.index[~changed], df.loc[~changed, 'Raw_Return'], 
                color='blue', alpha=0.2, label='Stable Data', s=10)
    
    # Add horizontal lines for limits
    lower_limit = df.loc[changed, 'Winsorized_Return'].min()
    upper_limit = df.loc[changed, 'Winsorized_Return'].max()
    plt.axhline(y=upper_limit, color='gray', linestyle='--', alpha=0.5, label='99th Percentile')
    plt.axhline(y=lower_limit, color='gray', linestyle='--', alpha=0.5, label='1st Percentile')

    plt.title('Winsorization Impact: Raw vs. Cleaned Signal', fontsize=16, fontweight='bold')
    plt.ylabel('Daily Return (%)', fontsize=12)
    plt.xlabel('Observation Index', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    
    # Final adjustment to labels for percentage
    plt.gca().set_yticklabels(['{:.1f}%'.format(x*100) for x in plt.gca().get_yticks()])

    # Save
    output_path = "output/eda_vn30/figures/winsorization_scatter.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    # Focus on a few symbols known for high kurtosis or typical stocks
    target_symbols = ['SSB', 'VJC', 'VHM', 'TCB', 'FPT']
    df = fetch_returns(target_symbols)
    plot_winsorization(df)
