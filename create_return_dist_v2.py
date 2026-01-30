import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from xnoapi import client
from xnoapi.vn.data.stocks import Quote
from datetime import datetime

# Import configuration
from config import XNOAPI_KEY, VN30_SYMBOLS, START_DATE

# Initialize API
client(apikey=XNOAPI_KEY)

def fetch_returns(symbols):
    all_returns = []
    print(f"Fetching data for {len(symbols)} stocks...")
    for sym in symbols:
        try:
            quote = Quote(sym)
            # Use a slightly shorter range or check specific period if needed, 
            # but START_DATE from config is usually best.
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

    # Statistics for the normal curve
    mu, std = returns.mean(), returns.std()
    
    # Filtering for visualization clarity (remove extreme outliers for the plot range)
    filtered_returns = returns[abs(returns - mu) < 5 * std]

    plt.figure(figsize=(12, 8), dpi=300)
    
    # Plot histogram
    sns.histplot(filtered_returns, kde=False, element="step", color="skyblue", stat="density", label="VN30 Returns")
    
    # Plot Normal Curve
    x = np.linspace(filtered_returns.min(), filtered_returns.max(), 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2, label=f'Normal Distribution\n($\mu$={mu:.4f}, $\sigma$={std:.4f})')
    
    # Add titles and labels
    plt.title('VN30 Return Distribution vs. Normal Curve (Fat Tails)', fontsize=16, fontweight='bold')
    plt.xlabel('Daily Return', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # Legend and Grid
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Path handling to match LaTeX reference if needed, 
    # though visual_assets_guide says output/eda_vn30/figures/return_distribution.png
    # The user's LaTeX code just says {return_distribution.png}, so I will also save a copy in the root or current dir if they want.
    # But I'll stick to the project structure and let them know.
    
    output_path = "output/eda_vn30/figures/return_distribution.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    
    # Also save a copy in the same directory as the LaTeX might expect
    plt.savefig("return_distribution.png")
    
    print(f"Chart saved to {output_path} and return_distribution.png")
    plt.close()

if __name__ == "__main__":
    returns = fetch_returns(VN30_SYMBOLS)
    plot_distribution(returns)
