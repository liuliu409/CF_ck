"""
VN30 Exploratory Data Analysis (EDA)
=====================================
Comprehensive analysis of VN30 stocks over 10 years.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from scipy.stats import skew, kurtosis
from xnoapi import client
from xnoapi.vn.data.stocks import Quote

# Import configuration
from config import XNOAPI_KEY, VN30_SYMBOLS, STOCK_SECTORS

# Constants
OUTPUT_DIR = "output/eda_vn30"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "individual_reports")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
START_DATE = "2016-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Create directories
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize API
client(apikey=XNOAPI_KEY)

def fetch_data(symbols):
    all_dfs = {}
    print(f"Fetching 10 years of data for {len(symbols)} stocks...")
    for sym in symbols:
        try:
            quote = Quote(sym)
            data = quote.history(start=START_DATE, interval="1D")
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            df = df.rename(columns={'time': 'date'})
            df = df.sort_values('date').reset_index(drop=True)
            df['symbol'] = sym
            # Calculate returns
            df['daily_return'] = df['close'].pct_change()
            all_dfs[sym] = df
            print(f"  ✓ {sym}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ Error fetching {sym}: {e}")
    return all_dfs

def calculate_statistics(all_data):
    stats = []
    for sym, df in all_data.items():
        returns = df['daily_return'].dropna()
        if len(returns) < 2: continue
        
        stats.append({
            'Ticker': sym,
            'Mean': returns.mean(),
            'Volatility': returns.std(),
            'Skewness': skew(returns),
            'Kurtosis': kurtosis(returns),
            'Annualized_Return': returns.mean() * 252,
            'Annualized_Vol': returns.std() * np.sqrt(252)
        })
    return pd.DataFrame(stats)

def sector_neutralize_corr(returns_df, sectors):
    # Map sectors
    groups = returns_df.columns.map(sectors)
    # Simple neutralization: subtract sector mean
    neutral_returns = returns_df.copy()
    for sector in set(groups):
        sector_cols = returns_df.columns[groups == sector]
        if len(sector_cols) > 0:
            sector_avg = returns_df[sector_cols].mean(axis=1)
            for col in sector_cols:
                neutral_returns[col] = returns_df[col] - sector_avg
    return neutral_returns.corr()

def plot_correlation_heatmap(corr_df):
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_df, annot=False, cmap='coolwarm', center=0)
    plt.title("Sector-Neutralized Correlation Matrix (VN30)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"))
    plt.close()

def plot_risk_return(stats_df):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=stats_df, x='Annualized_Vol', y='Annualized_Return', hue='Ticker', legend=False)
    for i in range(stats_df.shape[0]):
        plt.text(stats_df.Annualized_Vol[i], stats_df.Annualized_Return[i], stats_df.Ticker[i], fontsize=9)
    plt.title("Risk-Return Scatter Plot (Annualized)", fontsize=16)
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "risk_return_scatter.png"))
    plt.close()

def plot_momentum_heatmap(all_data):
    momentum = {}
    for sym, df in all_data.items():
        if len(df) < 252: continue
        price = df['close']
        momentum[sym] = {
            '1M': (price.iloc[-1] / price.iloc[-21]) - 1 if len(price) >= 21 else np.nan,
            '3M': (price.iloc[-1] / price.iloc[-63]) - 1 if len(price) >= 63 else np.nan,
            '6M': (price.iloc[-1] / price.iloc[-126]) - 1 if len(price) >= 126 else np.nan,
            '12M': (price.iloc[-1] / price.iloc[-252]) - 1 if len(price) >= 252 else np.nan,
        }
    mom_df = pd.DataFrame(momentum).T
    plt.figure(figsize=(12, 10))
    sns.heatmap(mom_df, annot=True, fmt=".2%", cmap='RdYlGn', center=0)
    plt.title("Price Momentum Heatmap (VN30)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "momentum_heatmap.png"))
    plt.close()

def generate_individual_reports(all_data):
    print("Generating individual PDF reports...")
    for sym, df in all_data.items():
        report_path = os.path.join(REPORTS_DIR, f"{sym}_report.pdf")
        with PdfPages(report_path) as pdf:
            # Price and MAs
            plt.figure(figsize=(11, 8.5))
            df['MA50'] = df['close'].rolling(window=50).mean()
            df['MA200'] = df['close'].rolling(window=200).mean()
            
            plt.subplot(2, 1, 1)
            plt.plot(df['date'], df['close'], label='Close Price', color='blue', alpha=0.6)
            plt.plot(df['date'], df['MA50'], label='MA-50', color='orange')
            plt.plot(df['date'], df['MA200'], label='MA-200', color='red')
            plt.title(f"{sym} Price Analysis & Moving Averages")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Volume Distribution
            plt.subplot(2, 1, 2)
            sns.histplot(df['volume'], kde=True, color='purple')
            plt.title(f"{sym} Volume Distribution")
            plt.xlabel("Volume")
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()

def detect_anomalies(all_data):
    anomalies = []
    for sym, df in all_data.items():
        returns = df['daily_return'].dropna()
        z_scores = (returns - returns.mean()) / returns.std()
        outliers = df.loc[z_scores[abs(z_scores) > 3].index]
        for idx, row in outliers.iterrows():
            anomalies.append({
                'Ticker': sym,
                'Date': row['date'],
                'Return': row['daily_return'],
                'Z-Score': z_scores[idx]
            })
    return pd.DataFrame(anomalies)

def main(all_data=None):
    # 1. Fetch Data if not provided
    if all_data is None:
        all_data = fetch_data(VN30_SYMBOLS)
    
    if not all_data:
        print("No data fetched. Exiting.")
        return

    # 2. Statistics
    stats_df = calculate_statistics(all_data)
    stats_df.to_csv(os.path.join(DATA_DIR, "summary_statistics.csv"), index=False)
    print(f"✓ Statistics saved to {DATA_DIR}/summary_statistics.csv")

    # 3. Visualization
    # Prep returns dataframe for correlation
    returns_dict = {sym: df.set_index('date')['daily_return'] for sym, df in all_data.items()}
    returns_df = pd.DataFrame(returns_dict).dropna()
    
    corr_df = sector_neutralize_corr(returns_df, STOCK_SECTORS)
    plot_correlation_heatmap(corr_df)
    plot_risk_return(stats_df)
    plot_momentum_heatmap(all_data)
    print(f"✓ Figures saved to {FIGURES_DIR}/")

    # 4. Individual Reports
    generate_individual_reports(all_data)
    print(f"✓ Individual reports saved to {REPORTS_DIR}/")

    # 5. Anomaly Detection
    anomalies_df = detect_anomalies(all_data)
    anomalies_df.to_csv(os.path.join(DATA_DIR, "anomalies.csv"), index=False)
    print(f"✓ Anomalies saved to {DATA_DIR}/anomalies.csv")

    # 6. Generate README Report
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
        f.write("# VN30 Exploratory Data Analysis Report\n\n")
        f.write("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write("## Summary Statistics\n")
        f.write(stats_df.describe().to_markdown() + "\n\n")
        f.write("## Visualizations\n")
        f.write("- [Correlation Heatmap](figures/correlation_heatmap.png)\n")
        f.write("- [Risk-Return Scatter Plot](figures/risk_return_scatter.png)\n")
        f.write("- [Momentum Heatmap](figures/momentum_heatmap.png)\n\n")
        f.write("## Anomalies Detected\n")
        f.write(f"Total anomalies detected: {len(anomalies_df)}\n")
        f.write(anomalies_df.head(10).to_markdown() + "\n\n")
        f.write("Individual PDFs for each stock are located in `individual_reports/`.\n")
    print(f"✓ README report created in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
