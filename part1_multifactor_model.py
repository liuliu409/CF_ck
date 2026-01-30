"""
Part 1: Multi-Factor Model for VN30 Stocks
===========================================

This script implements a multi-factor model using:
1. Size Factor (Market Capitalization)
2. Value Factor (Price-to-Book or Price-to-Earnings)
3. Momentum Factor (12-month price momentum)

Author: Financial Computing Project
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import xnoapi for data fetching
from xnoapi import client
from xnoapi.vn.data.stocks import Quote

# Import configuration
from config import (
    XNOAPI_KEY, VN30_SYMBOLS, START_DATE, END_DATE,
    FACTOR_WEIGHTS, MOMENTUM_LOOKBACK, MOMENTUM_SKIP_LAST, VALUE_METRIC,
    OUTPUT_DIR, FIGURES_DIR, DATA_DIR, FIGURE_DPI, FIGURE_FORMAT
)
from robust_utils import calculate_value_factor_ep

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize xnoapi client
client(apikey=XNOAPI_KEY)

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a single stock using xnoapi.
    
    Args:
        symbol: Stock ticker (e.g., 'FPT')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    try:
        print(f"Fetching data for {symbol}...")
        
        # Use xnoapi to fetch quote data
        quote = Quote(symbol)
        data = quote.history(start=start_date, interval="1D")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Process time column and set as index temporarily
        df["time"] = pd.to_datetime(df["time"])
        
        # Rename columns to standardized format
        df = df.rename(columns={
            "time": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })
        
        # Filter by end_date if provided
        if end_date:
            df = df[df['date'] <= end_date]
        
        df = df.sort_values('date').reset_index(drop=True)
        df['symbol'] = symbol
        
        print(f"  ✓ {symbol}: {len(df)} rows from {df['date'].min()} to {df['date'].max()}")
        return df
        
    except Exception as e:
        print(f"  ✗ Error fetching {symbol}: {e}")
        return pd.DataFrame()


def fetch_all_stocks(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for all stocks in the universe.
    
    Returns:
        Dictionary mapping symbol to DataFrame
    """
    print(f"\n{'='*60}")
    print(f"FETCHING DATA FOR {len(symbols)} STOCKS")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    all_data = {}
    for symbol in symbols:
        df = fetch_stock_data(symbol, start_date, end_date)
        if not df.empty:
            all_data[symbol] = df
    
    print(f"\n✓ Successfully fetched data for {len(all_data)}/{len(symbols)} stocks\n")
    return all_data


# =============================================================================
# FACTOR CALCULATIONS
# =============================================================================

def calculate_size_factor(all_data: Dict[str, pd.DataFrame]) -> pd.Series:
    """
    Calculate Size Factor based on market capitalization.
    
    Size Factor = -rank(market_cap)
    Negative sign because we prefer smaller companies (size premium)
    
    Note: Since we don't have shares outstanding data from xnoapi,
    we'll use price as a proxy for market cap (assuming similar share counts)
    or you can manually input market cap data.
    
    Returns:
        Series with symbol as index and size factor score as values
    """
    print("Calculating Size Factor...")
    
    size_scores = {}
    
    for symbol, df in all_data.items():
        # Use latest closing price as proxy for market cap
        # In practice, you should use: price * shares_outstanding
        latest_price = df['close'].iloc[-1]
        size_scores[symbol] = latest_price
    
    # Convert to Series and rank
    size_series = pd.Series(size_scores)
    
    # Negative rank: smaller companies get higher scores
    size_factor = -size_series.rank()
    
    # Normalize to z-scores
    size_factor = (size_factor - size_factor.mean()) / size_factor.std()
    
    print(f"  ✓ Size Factor calculated for {len(size_factor)} stocks")
    return size_factor


def calculate_value_factor(all_data: Dict[str, pd.DataFrame], 
                           metric: str = 'ep') -> pd.Series:
    """
    Calculate Value Factor based on Earnings Yield (E/P).
    
    Value Factor = z-score(Neutralized(Winsorized(E/P)))
    
    Why Earnings Yield (E/P)?
    In emerging markets like Vietnam, P/B can be distorted by asset revaluations
    and non-transparent balance sheets. E/P (the inverse of P/E) provides a 
    more direct link to profitability and cash-flow generation. It also 
    handles negative earnings mathematically (negative yield) whereas P/E
    becomes discontinuous.
    
    Args:
        metric: 'ep' for Earnings Yield (default)
    
    Returns:
        Series with symbol as index and value factor score as values
    """
    if metric == 'ep':
        print("Calculating Value Factor using Earnings Yield (E/P)...")
        # Delegate to robust implementation which includes:
        # 1. LTM Earnings / Market Cap logic (via ep_ratios template)
        # 2. Sector median imputation
        # 3. 5%-95% winsorization
        # 4. Sector neutralization
        value_factor = calculate_value_factor_ep(all_data)
        
        print(f"  ✓ Value Factor calculated for {len(value_factor)} stocks using Earnings Yield")
        print(f"  ⚠ Note: Using Earnings Yield (E/P). Values winsorized and sector-neutralized.")
        return value_factor
    else:
        # Fallback for other metrics (if ever needed)
        print(f"Calculating Value Factor (using {metric.upper()})...")
        # (Previous P/B logic or error handling)
        return pd.Series(dtype=float)


def calculate_momentum_factor(all_data: Dict[str, pd.DataFrame],
                              lookback: int = 252,
                              skip_last: int = 21) -> pd.Series:
    """
    Calculate Momentum Factor based on past 12-month returns.
    
    Momentum = (Price_t / Price_t-12) - 1
    Skip last month to avoid short-term reversal
    
    Args:
        lookback: Number of trading days to look back (252 = 1 year)
        skip_last: Number of days to skip at the end (21 = 1 month)
    
    Returns:
        Series with symbol as index and momentum factor score as values
    """
    print(f"Calculating Momentum Factor ({lookback} days, skip last {skip_last})...")
    
    momentum_scores = {}
    
    for symbol, df in all_data.items():
        if len(df) < lookback + skip_last:
            print(f"  ⚠ {symbol}: Insufficient data for momentum calculation")
            continue
        
        # Get price from (lookback + skip_last) days ago
        start_price = df['close'].iloc[-(lookback + skip_last)]
        # Get price from skip_last days ago (not today)
        end_price = df['close'].iloc[-skip_last] if skip_last > 0 else df['close'].iloc[-1]
        
        # Calculate momentum return
        momentum = (end_price / start_price) - 1
        momentum_scores[symbol] = momentum
    
    momentum_series = pd.Series(momentum_scores)
    
    # Positive rank: higher momentum gets higher scores
    momentum_factor = momentum_series.rank()
    
    # Normalize to z-scores
    momentum_factor = (momentum_factor - momentum_factor.mean()) / momentum_factor.std()
    
    print(f"  ✓ Momentum Factor calculated for {len(momentum_factor)} stocks")
    return momentum_factor


def calculate_composite_score(size_factor: pd.Series,
                              value_factor: pd.Series,
                              momentum_factor: pd.Series,
                              weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    Calculate composite factor score as weighted average of individual factors.
    
    Args:
        size_factor: Size factor scores
        value_factor: Value factor scores
        momentum_factor: Momentum factor scores
        weights: Dictionary with keys 'size', 'value', 'momentum'
    
    Returns:
        DataFrame with individual factors and composite score
    """
    if weights is None:
        weights = FACTOR_WEIGHTS
    
    print("\nCalculating Composite Factor Scores...")
    print(f"  Weights: Size={weights['size']:.2f}, Value={weights['value']:.2f}, Momentum={weights['momentum']:.2f}")
    
    # Combine all factors into DataFrame
    factors_df = pd.DataFrame({
        'size_factor': size_factor,
        'value_factor': value_factor,
        'momentum_factor': momentum_factor
    })
    
    # Calculate composite score
    factors_df['composite_score'] = (
        weights['size'] * factors_df['size_factor'] +
        weights['value'] * factors_df['value_factor'] +
        weights['momentum'] * factors_df['momentum_factor']
    )
    
    # Rank stocks by composite score
    factors_df['rank'] = factors_df['composite_score'].rank(ascending=False)
    
    # Sort by rank
    factors_df = factors_df.sort_values('composite_score', ascending=False)
    
    print(f"  ✓ Composite scores calculated for {len(factors_df)} stocks\n")
    
    return factors_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_factor_heatmap(factors_df: pd.DataFrame, save_path: str = None):
    """
    Plot heatmap of factor scores for all stocks.
    """
    plt.figure(figsize=(10, 8))
    
    # Select only factor columns
    factor_cols = ['size_factor', 'value_factor', 'momentum_factor', 'composite_score']
    plot_data = factors_df[factor_cols].T
    
    sns.heatmap(plot_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Factor Score (z-score)'})
    
    plt.title('Multi-Factor Model: Factor Scores Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Stock Symbol', fontsize=12)
    plt.ylabel('Factor', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        print(f"  ✓ Saved heatmap to {save_path}")
    
    plt.show()


def plot_factor_rankings(factors_df: pd.DataFrame, save_path: str = None):
    """
    Plot bar chart of composite factor scores.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create color map: green for positive, red for negative
    colors = ['green' if x > 0 else 'red' for x in factors_df['composite_score']]
    
    ax.bar(factors_df.index, factors_df['composite_score'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Stock Symbol', fontsize=12)
    ax.set_ylabel('Composite Factor Score', fontsize=12)
    ax.set_title('Stock Rankings by Composite Factor Score', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        print(f"  ✓ Saved rankings to {save_path}")
    
    plt.show()


def plot_factor_correlation(factors_df: pd.DataFrame, save_path: str = None):
    """
    Plot correlation matrix of factors.
    """
    factor_cols = ['size_factor', 'value_factor', 'momentum_factor']
    corr_matrix = factors_df[factor_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, cbar_kws={'label': 'Correlation'})
    
    plt.title('Factor Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        print(f"  ✓ Saved correlation matrix to {save_path}")
    
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for Part 1: Multi-Factor Model
    """
    print("\n" + "="*60)
    print("PART 1: MULTI-FACTOR MODEL FOR VN30 STOCKS")
    print("="*60 + "\n")
    
    # Step 1: Fetch data
    all_data = fetch_all_stocks(VN30_SYMBOLS, START_DATE, END_DATE)
    
    if len(all_data) == 0:
        print("❌ No data fetched. Please check your API key and connection.")
        return
    
    # Step 2: Calculate individual factors
    print("\n" + "-"*60)
    print("CALCULATING FACTORS")
    print("-"*60 + "\n")
    
    size_factor = calculate_size_factor(all_data)
    value_factor = calculate_value_factor(all_data, metric=VALUE_METRIC)
    momentum_factor = calculate_momentum_factor(all_data, MOMENTUM_LOOKBACK, MOMENTUM_SKIP_LAST)
    
    # Step 3: Calculate composite scores
    factors_df = calculate_composite_score(size_factor, value_factor, momentum_factor)
    
    # Step 4: Display results
    print("\n" + "-"*60)
    print("FACTOR SCORES SUMMARY")
    print("-"*60 + "\n")
    print(factors_df.round(3))
    
    print("\n" + "-"*60)
    print("TOP 5 STOCKS (Highest Composite Score)")
    print("-"*60)
    print(factors_df.head(5)[['composite_score', 'rank']])
    
    print("\n" + "-"*60)
    print("BOTTOM 5 STOCKS (Lowest Composite Score)")
    print("-"*60)
    print(factors_df.tail(5)[['composite_score', 'rank']])
    
    # Step 5: Save results
    output_file = os.path.join(DATA_DIR, 'factor_scores.csv')
    factors_df.to_csv(output_file)
    print(f"\n✓ Factor scores saved to {output_file}")
    
    # Step 6: Create visualizations
    print("\n" + "-"*60)
    print("GENERATING VISUALIZATIONS")
    print("-"*60 + "\n")
    
    plot_factor_heatmap(factors_df, os.path.join(FIGURES_DIR, 'factor_heatmap.png'))
    plot_factor_rankings(factors_df, os.path.join(FIGURES_DIR, 'factor_rankings.png'))
    plot_factor_correlation(factors_df, os.path.join(FIGURES_DIR, 'factor_correlation.png'))
    
    print("\n" + "="*60)
    print("✓ PART 1 COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")
    print(f"Results saved to: {DATA_DIR}/")
    print(f"Figures saved to: {FIGURES_DIR}/")
    print("\nNext step: Run part2_portfolio_optimization.py")
    
    return factors_df, all_data


if __name__ == "__main__":
    # Run main function
    factors_df, all_data = main()
