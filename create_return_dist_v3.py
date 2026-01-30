import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis, skew
from xnoapi import client
from xnoapi.vn.data.stocks import Quote

# Import configuration
from config import XNOAPI_KEY, VN30_SYMBOLS, START_DATE

client(apikey=XNOAPI_KEY)

def fetch_returns(symbols):
    all_returns = []
    print(f"Fetching data for {len(symbols)} stocks...")
    for sym in symbols:
        try:
            quote = Quote(sym)
            data = quote.history(start=START_DATE, interval="1D")
            df = pd.DataFrame(data)
            if not df.empty and 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                all_returns.append(returns)
                print(f"  ✓ {sym}")
        except Exception as e:
            print(f"  ✗ Error fetching {sym}: {e}")
    return pd.concat(all_returns) if all_returns else pd.Series()

def plot_distribution(returns):
    if returns.empty:
        print("No returns data to plot.")
        return

    # Calculate statistics
    mu, std = returns.mean(), returns.std()
    kurt = kurtosis(returns)
    sk = skew(returns)
    
    # Filter for visualization (clip at 4 sigma to focus on the bulk while still showing tails)
    plot_data = returns[abs(returns - mu) < 4 * std]

    plt.figure(figsize=(14, 9), dpi=300)
    plt.style.use('bmh') # Use a high-visibility style

    # 1. Plot Histogram (using matplotlib directly for more control)
    n, bins, patches = plt.hist(plot_data, bins=100, density=True, 
                               alpha=0.6, color='dodgerblue', edgecolor='white', 
                               label='VN30 Daily Returns')
    
    # 2. Add Normal Curve
    x = np.linspace(plot_data.min(), plot_data.max(), 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r-', lw=3, label=f'Normal Curve ($\sigma$={std:.2%})')
    
    # 3. Add Vertical lines for mean and 2-sigma
    plt.axvline(mu, color='black', linestyle='--', lw=2, label=f'Mean: {mu:.4%}')
    plt.axvline(mu - 2*std, color='orange', linestyle=':', lw=2, label='-2$\sigma$ Limit')
    plt.axvline(mu + 2*std, color='orange', linestyle=':', lw=2)

    # 4. Annotate Statistics
    stats_text = (f"Skewness: {sk:.2f}\n"
                  f"Kurtosis: {kurt:.2f}\n"
                  f"Total Obs: {len(returns):,}")
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=12, fontweight='bold')

    # Formatting
    plt.title('Return Distribution vs. Theoretical Normal (Fat Tails Detected)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Daily Return (%)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.gca().set_xticklabels(['{:.1f}%'.format(x*100) for x in plt.gca().get_xticks()])
    
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save
    plt.savefig("return_distribution.png", bbox_inches='tight')
    plt.savefig("output/eda_vn30/figures/return_distribution.png", bbox_inches='tight')
    
    print(f"Chart re-generated successfully.")
    plt.close()

if __name__ == "__main__":
    returns = fetch_returns(VN30_SYMBOLS)
    plot_distribution(returns)
