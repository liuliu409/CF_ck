"""
Robust Utilities Module
=======================
Advanced quantitative functions for:
1. Ledoit-Wolf Covariance Shrinkage
2. IC-based Expected Return Estimation
3. Factor Score Winsorization
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from typing import Dict, List, Tuple
from config import STOCK_SECTORS

def winsorize_scores(scores: pd.Series, method: str = 'std', limit: float = 3.0) -> pd.Series:
    """
    Cap extreme scores to prevent optimization instability.
    method: 'std' (standard deviations) or 'percentile' (clipping outliers)
    """
    if method == 'std':
        mean = scores.mean()
        std = scores.std()
        return scores.clip(lower=mean - limit * std, upper=mean + limit * std)
    elif method == 'percentile':
        # limit acts as the percentile to clip (e.g. 0.05 for 5%-95%)
        lower = scores.quantile(limit)
        upper = scores.quantile(1 - limit)
        return scores.clip(lower=lower, upper=upper)
    return scores

def fill_sector_medians(scores: pd.Series) -> pd.Series:
    """
    Fill missing values using sector-aware medians.
    """
    df = pd.DataFrame({'score': scores})
    df['sector'] = df.index.map(STOCK_SECTORS)
    
    # Fill with sector median first
    df['score'] = df.groupby('sector')['score'].transform(lambda x: x.fillna(x.median()))
    
    # If still missing (sector has no data), fill with global median
    df['score'] = df['score'].fillna(df['score'].median())
    
    return df['score']

def standardize(scores: pd.Series) -> pd.Series:
    """
    Z-score normalization.
    """
    if scores.std() == 0:
        return scores - scores.mean()
    return (scores - scores.mean()) / scores.std()

def neutralize_by_sector(scores: pd.Series) -> pd.Series:
    """
    Perform sector-neutralization by de-meaning within industry groups.
    Isolates idiosyncratic alpha from sector momentum.
    """
    # Create temp dataframe for grouping
    df = pd.DataFrame({'score': scores})
    df['sector'] = df.index.map(STOCK_SECTORS)
    
    # De-mean within sector
    # If a sector has only 1 stock, de-meaning makes it 0 (pure alpha check)
    def sector_demean(x):
        if len(x) > 1:
            return x - x.mean()
        return x
        
    df['neut_score'] = df.groupby('sector')['score'].transform(sector_demean)
    
    # Re-standardize the neutralized result
    return standardize(df['neut_score'])

def calculate_quality_factor(all_data: Dict[str, pd.DataFrame], lookback: int = 252) -> pd.Series:
    """
    Calculate Quality Factor using Volatility as a proxy (Low Vol Anomaly).
    Quality = -rank(annualized_volatility)
    """
    vols = {}
    for symbol, df in all_data.items():
        if len(df) < lookback:
            vols[symbol] = np.nan
            continue
        # Recent returns
        rets = df['close'].pct_change().tail(lookback)
        vols[symbol] = rets.std() * np.sqrt(252)
    
    vol_series = pd.Series(vols).dropna()
    
    # Negative rank: Lower volatility gets higher score (Quality preference)
    quality_factor = -vol_series.rank()
    return standardize(quality_factor)

def calculate_value_factor_ep(all_data: Dict[str, pd.DataFrame]) -> pd.Series:
    """
    Calculate Value Factor using Earnings Yield (E/P).
    E/P is generally more stable than P/E and handles negative earnings better
    without mathematical discontinuities.
    
    E/P = LTM Earnings / Market Cap
    """
    # Placeholder Earnings Yield (E/P) ratios based on realistic VN30 estimates
    # E/P = 1 / P/E. Examples: P/E of 10x = 0.10 E/P
    # These are illustrative for the project scope.
    ep_ratios = {
        'ACB': 0.15, 'BCM': 0.04, 'BID': 0.08, 'BVH': 0.07, 'CTG': 0.10, 
        'FPT': 0.05, 'GAS': 0.08, 'GVR': 0.04, 'HDB': 0.14, 'HPG': 0.11, 
        'MBB': 0.18, 'MSN': 0.04, 'MWG': 0.06, 'PLX': 0.09, 'POW': 0.12, 
        'SAB': 0.05, 'SHB': 0.22, 'SSB': 0.08, 'SSI': 0.09, 'STB': 0.13, 
        'TCB': 0.16, 'TPB': 0.15, 'VCB': 0.07, 'VHM': 0.18, 'VIB': 0.16, 
        'VIC': 0.05, 'VJC': 0.03, 'VNM': 0.06, 'VPB': 0.12, 'VRE': 0.12
    }
    
    # 1. Gather scores
    scores = pd.Series(index=all_data.keys(), dtype=float)
    for symbol in all_data.keys():
        scores[symbol] = ep_ratios.get(symbol, np.nan)
    
    # 2. Fill missing with sector medians
    scores = fill_sector_medians(scores)
    
    # 3. Winsorize (5% - 95% percentile clipping)
    scores = winsorize_scores(scores, method='percentile', limit=0.05)
    
    # 4. Sector Neutralization (Isolate pure value signal)
    scores = neutralize_by_sector(scores)
    
    # 5. Final Standardization
    return standardize(scores)

def apply_ledoit_wolf(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Ledoit-Wolf shrunk covariance matrix.
    Reduces estimation error in MVO by shrinking sample covariance 
    towards a structured target.
    """
    lw = LedoitWolf()
    # Fit data and annualize (assuming daily returns)
    shrunk_cov = lw.fit(returns_df.dropna()).covariance_ * 252
    return pd.DataFrame(shrunk_cov, index=returns_df.columns, columns=returns_df.columns)

def estimate_ic_premium(factor_scores: pd.Series, returns_df: pd.DataFrame, horizon: int = 21) -> float:
    """
    Estimate the Information Coefficient (IC) for a set of factor scores.
    IC = Spearman rank correlation between score at t and return in [t, t+horizon]
    """
    # Calculate forward returns for the period
    # Note: In a rolling loop, this would be computed on the training window
    fwd_returns = returns_df.shift(-horizon).rolling(window=horizon).sum().iloc[-1]
    
    # Align scores and returns
    common_index = factor_scores.index.intersection(fwd_returns.index)
    if len(common_index) < 5:
        return 0.05 # Default floor IC for stability
    
    ic = factor_scores[common_index].corr(fwd_returns[common_index], method='spearman')
    
    # Handle NaNs or extreme volatility in IC
    if np.isnan(ic):
        return 0.02 # Minimal floor
        
    return ic

def map_ic_to_returns(scores: pd.Series, 
                      volatility: pd.Series, 
                      ic: float, 
                      vol_target: float = 0.15) -> pd.Series:
    """
    Grinold's Fundamental Law: E[r] = Vol * Score * IC
    Converts standardized factor scores into expected returns grounded 
    in statistical predictive power.
    """
    # Scores should be Z-scored already
    # Apply IC scaling and asset volatility
    expected_returns = volatility * scores * ic
    
    # Scale to a realistic mean return if necessary
    # (Optional: Adjustment for long-term equity risk premium)
    expected_returns += 0.08 # 8% base risk premium for VN stocks
    
    return expected_returns

def scale_to_matched_volatility(asset_returns: pd.Series, 
                                target_returns: pd.Series) -> pd.Series:
    """
    Scales a return series to match the annualized volatility of a target series.
    Logic: Scaled_Returns = Asset_Returns * (Target_Vol / Asset_Vol)
    
    Used to create a 'Risk-Matched' benchmark for fair comparison.
    """
    asset_vol = asset_returns.std() * np.sqrt(252)
    target_vol = target_returns.std() * np.sqrt(252)
    
    if asset_vol == 0:
        return asset_returns
        
    scale_factor = target_vol / asset_vol
    return asset_returns * scale_factor

def calculate_performance_metrics(returns: pd.Series, rf_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics:
    - Annualized Return
    - Annualized Volatility
    - Sharpe Ratio
    - Max Drawdown
    - Calmar Ratio
    """
    if len(returns) == 0:
        return {}

    # 1. Basic Stats
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - rf_rate) / ann_vol if ann_vol > 0 else 0

    # 2. Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # 3. Calmar Ratio
    # Absolute max drawdown for ratio
    abs_mdd = abs(max_drawdown)
    calmar = ann_return / abs_mdd if abs_mdd > 0 else 0

    return {
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'calmar': calmar
    }
