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
    all_raw_returns = []
    print(f"Fetching data for {len(symbols)} stocks...")
    for sym in symbols:
        try:
            quote = Quote(sym)
            data = quote.history(start=START_DATE, interval="1D")
            df = pd.DataFrame(data)
            if not df.empty and 'close' in df.columns:
                # Calculate daily returns
                rets = df['close'].pct_change().values # Get raw numpy array
                # Drop the first NaN from pct_change
                rets = rets[~np.isnan(rets)]
                all_raw_returns.append(rets)
                print(f"  ✓ {sym}")
        except Exception as e:
            print(f"  ✗ Error fetching {sym}: {e}")
    
    if not all_raw_returns:
        return np.array([])
    
    # Flatten everything into a single 1D numpy array
    combined = np.concatenate(all_raw_returns)
    # Final safety check: drop any NaNs or Infs
    combined = combined[np.isfinite(combined)]
    
    print(f"Total valid observations: {len(combined)}")
    return combined

def plot_distribution(returns):
    if len(returns) == 0:
        print("No returns data to plot.")
        return

    # Calculate statistics
    mu = np.mean(returns)
    std = np.std(returns)
    kurt_val = kurtosis(returns)
    skew_val = skew(returns)
    
    # Filter for visualization (clip at 5 sigma to show tails clearly)
    # We use a broad range for bins but cap the view
    plot_min = mu - 5 * std
    plot_max = mu + 5 * std
    plot_data = returns[(returns >= plot_min) & (returns <= plot_max)]

    plt.figure(figsize=(14, 9), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid') # Reliable high-quality style

    # 1. Plot Histogram
    # We use density=True to ensure area scales to 1.0 (matching Normal PDF)
    plt.hist(plot_data, bins=100, density=True, 
             alpha=0.7, color='steelblue', edgecolor='white', 
             label='VN30 Observed Returns')
    
    # 2. Plot Normal Distribution Curve
    # Use a high-resolution X-axis for smooth curve
    x = np.linspace(plot_data.min(), plot_data.max(), 1000)
    y = norm.pdf(x, mu, std)
    plt.plot(x, y, 'firebrick', lw=4, label=f'Normal ($\mu$={mu:.2%}, $\sigma$={std:.2%})')
    
    # 3. Add statistical annotations
    stats_box = (f"Kurtosis: {kurt_val:.2f}\n"
                 f"Skewness: {skew_val:.2f}\n"
                 f"Sample Size: {len(returns):,}")
    
    plt.gca().text(0.95, 0.95, stats_box, transform=plt.gca().transAxes,
                   fontsize=14, fontweight='bold', verticalalignment='top', 
                   horizontalalignment='right', 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

    # Formatting labels and title
    plt.title('VN30 Daily Return Distribution vs. Normal Model', fontsize=20, fontweight='bold', pad=25)
    plt.xlabel('Daily Return (%)', fontsize=16)
    plt.ylabel('Density / Frequency', fontsize=16)
    
    # Force percentage labels on X-axis
    ticks = plt.gca().get_xticks()
    plt.gca().set_xticks(ticks) # Fix the ticks first to avoid warning
    plt.gca().set_xticklabels(['{:.1f}%'.format(t*100) for t in ticks])
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=14, frameon=True, shadow=True)
    
    # Clean up and save
    plt.tight_layout()
    plt.savefig("return_distribution.png")
    plt.savefig("output/eda_vn30/figures/return_distribution.png")
    
    print(f"Chart generated and saved to return_distribution.png")
    plt.close()

if __name__ == "__main__":
    data = fetch_returns(VN30_SYMBOLS)
    plot_distribution(data)
