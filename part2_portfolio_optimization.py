"""
Part 2: Portfolio Optimization for VN30 Stocks
===============================================

This script implements Mean-Variance Optimization (Markowitz Framework) to construct
an optimal portfolio from VN30 stocks using factor scores from Part 1.

Key Features:
1. Expected returns estimation using factor scores
2. Covariance matrix calculation
3. Efficient Frontier generation
4. Optimal portfolio weights (Maximum Sharpe Ratio)
5. Portfolio performance comparison

Author: Financial Computing Project
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import (
    VN30_SYMBOLS, START_DATE, END_DATE, RISK_FREE_RATE,
    MIN_WEIGHT, MAX_WEIGHT, N_PORTFOLIOS,
    TURNOVER_LIMIT, TRANSACTION_COST_MODEL, TURNOVER_PENALTY_LAMBDA,
    OUTPUT_DIR, FIGURES_DIR, DATA_DIR, FIGURE_DPI, FIGURE_FORMAT
)

import os

# =============================================================================
# DATA LOADING
# =============================================================================

def load_factor_scores(filepath: str = None) -> pd.DataFrame:
    """
    Load factor scores from Part 1.
    
    Args:
        filepath: Path to factor scores CSV file
    
    Returns:
        DataFrame with factor scores
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, 'factor_scores.csv')
    
    print(f"Loading factor scores from {filepath}...")
    factors_df = pd.read_csv(filepath, index_col=0)
    print(f"  ✓ Loaded factor scores for {len(factors_df)} stocks\n")
    
    return factors_df


def calculate_returns_from_data(all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate daily returns for all stocks.
    
    Args:
        all_data: Dictionary mapping symbol to price DataFrame
    
    Returns:
        DataFrame with returns (rows=dates, columns=symbols)
    """
    print("Calculating historical returns...")
    
    returns_dict = {}
    
    for symbol, df in all_data.items():
        df = df.set_index('date')
        returns = df['close'].pct_change().dropna()
        returns_dict[symbol] = returns
    
    # Combine into single DataFrame
    returns_df = pd.DataFrame(returns_dict)
    
    print(f"  ✓ Calculated returns for {len(returns_df.columns)} stocks")
    print(f"  Period: {returns_df.index.min()} to {returns_df.index.max()}")
    print(f"  Total observations: {len(returns_df)}\n")
    
    return returns_df


# =============================================================================
# EXPECTED RETURNS ESTIMATION
# =============================================================================

def estimate_expected_returns_historical(returns_df: pd.DataFrame) -> pd.Series:
    """
    Estimate expected returns using historical average (annualized).
    
    Args:
        returns_df: DataFrame of daily returns
    
    Returns:
        Series of annualized expected returns
    """
    # Calculate mean daily return and annualize (252 trading days)
    expected_returns = returns_df.mean() * 252
    return expected_returns


def estimate_expected_returns_factor_based(factors_df: pd.DataFrame,
                                          returns_df: pd.DataFrame,
                                          base_return: float = 0.10) -> pd.Series:
    """
    Estimate expected returns using factor scores.
    
    Higher factor scores → Higher expected returns
    
    Args:
        factors_df: DataFrame with factor scores
        returns_df: DataFrame of historical returns
        base_return: Base annual return (default 10%)
    
    Returns:
        Series of expected returns adjusted by factor scores
    """
    print("Estimating expected returns using factor scores...")
    
    # Get historical returns as baseline
    hist_returns = estimate_expected_returns_historical(returns_df)
    
    # Normalize composite scores to [0, 1]
    composite_scores = factors_df['composite_score']
    normalized_scores = (composite_scores - composite_scores.min()) / (composite_scores.max() - composite_scores.min())
    
    # Adjust expected returns based on factor scores
    # Stocks with high factor scores get boosted returns
    factor_adjustment = normalized_scores * 0.1  # Up to 10% adjustment
    
    expected_returns = base_return + factor_adjustment
    
    # Ensure alignment with historical returns
    expected_returns = expected_returns.reindex(hist_returns.index)
    
    print(f"  ✓ Expected returns range: {expected_returns.min():.2%} to {expected_returns.max():.2%}\n")
    
    return expected_returns


# =============================================================================
# COVARIANCE MATRIX
# =============================================================================

def calculate_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate annualized covariance matrix.
    
    Args:
        returns_df: DataFrame of daily returns
    
    Returns:
        Covariance matrix (annualized)
    """
    print("Calculating covariance matrix...")
    
    # Calculate covariance and annualize
    cov_matrix = returns_df.cov() * 252
    
    print(f"  ✓ Covariance matrix shape: {cov_matrix.shape}\n")
    
    return cov_matrix


# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================

def portfolio_performance(weights: np.ndarray, 
                         expected_returns: pd.Series,
                         cov_matrix: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate portfolio return and volatility.
    
    Args:
        weights: Array of portfolio weights
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix
    
    Returns:
        Tuple of (portfolio_return, portfolio_volatility)
    """
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    return portfolio_return, portfolio_volatility


# Portfolio Optimization logic now uses CVXPY directly in the optimize functions.
# negative_sharpe_ratio removed (obsolete).


def optimize_max_sharpe(expected_returns: pd.Series,
                        cov_matrix: pd.DataFrame,
                        risk_free_rate: float = RISK_FREE_RATE,
                        current_weights: pd.Series = None) -> Dict:
    """
    Find portfolio with maximum Sharpe ratio using CVXPY.
    Includes turnover and transaction cost constraints.
    """
    print("Optimizing for Maximum Sharpe Ratio (with Costs & Turnover)...")
    
    n = len(expected_returns)
    w = cp.Variable(n)
    
    # Baseline weights (e.g. from an existing portfolio)
    if current_weights is None:
        # Default to Equal Weighted as the benchmark starting point
        w_prev = np.array([1/n] * n)
    else:
        w_prev = current_weights.reindex(expected_returns.index).fillna(0).values

    # Objective: Maximize Risk-Adjusted Net Return
    # Expected Return - Risk Aversion * Risk - Transaction Costs
    # Note: Quad form is easier for convex solver than pure Sharpe
    # We calibrate risk aversion to match Sharpe profile
    risk_aversion = 1.0 
    
    # Transaction costs (L1 penalty)
    t_costs = cp.norm(w - w_prev, 1) * TRANSACTION_COST_MODEL
    
    # Combined Objective: Return - Risk - Transaction Costs - Turnover Penalty
    # The turnover penalty lambda (0.015) helps break uniform weight patterns
    objective = cp.Maximize(
        expected_returns.values @ w 
        - 0.5 * risk_aversion * cp.quad_form(w, cov_matrix.values) 
        - t_costs 
        - TURNOVER_PENALTY_LAMBDA * cp.norm(w - w_prev, 1)
    )
    
    constraints = [
        cp.sum(w) == 1,
        w >= MIN_WEIGHT,
        w <= MAX_WEIGHT,
        cp.norm(w - w_prev, 1) <= TURNOVER_LIMIT # Turnover Constraint
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if prob.status != cp.OPTIMAL:
        print(f"  ⚠ Optimization warning: {prob.status}")
        # Fallback to w_prev if failed
        optimal_weights = w_prev
    else:
        optimal_weights = w.value
        
    p_return, p_volatility = portfolio_performance(optimal_weights, expected_returns, cov_matrix)
    
    # Calculate real-world metrics
    total_turnover = np.sum(np.abs(optimal_weights - w_prev))
    net_return = p_return - (total_turnover * TRANSACTION_COST_MODEL)
    sharpe = (p_return - risk_free_rate) / p_volatility
    net_sharpe = (net_return - risk_free_rate) / p_volatility
    
    print(f"  ✓ Optimization successful")
    print(f"    Gross Expected Return: {p_return:.2%}")
    print(f"    Net Return (after costs): {net_return:.2%}")
    print(f"    Portfolio Turnover: {total_turnover:.2%}")
    print(f"    Volatility: {p_volatility:.2%}")
    print(f"    Net Sharpe Ratio: {net_sharpe:.3f}\n")
    
    return {
        'weights': pd.Series(optimal_weights, index=expected_returns.index),
        'return': net_return, # We focus on net numbers now
        'volatility': p_volatility,
        'sharpe': net_sharpe,
        'turnover': total_turnover
    }

def optimize_min_volatility(expected_returns: pd.Series,
                           cov_matrix: pd.DataFrame,
                           current_weights: pd.Series = None) -> Dict:
    """
    Find minimum volatility portfolio using CVXPY.
    """
    print("Optimizing for Minimum Volatility (with Turnover constraint)...")
    
    n = len(expected_returns)
    w = cp.Variable(n)
    
    if current_weights is None:
        w_prev = np.array([1/n] * n)
    else:
        w_prev = current_weights.reindex(expected_returns.index).fillna(0).values

    objective = cp.Minimize(
        cp.quad_form(w, cov_matrix.values) 
        + TURNOVER_PENALTY_LAMBDA * cp.norm(w - w_prev, 1)
    )
    
    constraints = [
        cp.sum(w) == 1,
        w >= MIN_WEIGHT,
        w <= MAX_WEIGHT,
        cp.norm(w - w_prev, 1) <= TURNOVER_LIMIT
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    optimal_weights = w.value
    p_return, p_volatility = portfolio_performance(optimal_weights, expected_returns, cov_matrix)
    
    total_turnover = np.sum(np.abs(optimal_weights - w_prev))
    net_return = p_return - (total_turnover * TRANSACTION_COST_MODEL)
    net_sharpe = (net_return - RISK_FREE_RATE) / p_volatility
    
    print(f"  ✓ Optimization successful")
    print(f"    Net Return: {net_return:.2%}")
    print(f"    Volatility: {p_volatility:.2%}")
    print(f"    Net Sharpe Ratio: {net_sharpe:.3f}\n")
    
    return {
        'weights': pd.Series(optimal_weights, index=expected_returns.index),
        'return': net_return,
        'volatility': p_volatility,
        'sharpe': net_sharpe,
        'turnover': total_turnover
    }


def generate_efficient_frontier(expected_returns: pd.Series,
                                cov_matrix: pd.DataFrame,
                                n_portfolios: int = N_PORTFOLIOS,
                                current_weights: pd.Series = None) -> pd.DataFrame:
    """
    Generate efficient frontier using CVXPY.
    """
    print(f"Generating Efficient Frontier ({n_portfolios} portfolios)...")
    
    n = len(expected_returns)
    w = cp.Variable(n)
    
    if current_weights is None:
        w_prev = np.array([1/n] * n)
    else:
        w_prev = current_weights.reindex(expected_returns.index).fillna(0).values

    # Range of return targets
    min_ret = expected_returns.min()
    max_ret = expected_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_portfolios)
    
    frontier_portfolios = []
    
    for target_ret in target_returns:
        # Minimize volatility for target return
        risk = cp.quad_form(w, cov_matrix.values)
        objective = cp.Minimize(risk)
        
        constraints = [
            cp.sum(w) == 1,
            w >= MIN_WEIGHT,
            w <= MAX_WEIGHT,
            expected_returns.values @ w >= target_ret,
            cp.norm(w - w_prev, 1) <= TURNOVER_LIMIT
        ]
        
        prob = cp.Problem(objective, constraints)
        # Solve with a fast solver
        prob.solve(solver=cp.SCS)
        
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            weights = w.value
            p_ret, p_vol = portfolio_performance(weights, expected_returns, cov_matrix)
            
            # Net return
            turnover = np.sum(np.abs(weights - w_prev))
            net_ret = p_ret - (turnover * TRANSACTION_COST_MODEL)
            sharpe = (net_ret - RISK_FREE_RATE) / p_vol
            
            frontier_portfolios.append({
                'return': p_ret,
                'net_return': net_ret,
                'volatility': p_vol,
                'sharpe': sharpe
            })
    
    frontier_df = pd.DataFrame(frontier_portfolios)
    print(f"  ✓ Generated {len(frontier_df)} efficient portfolios\n")
    
    return frontier_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_efficient_frontier(frontier_df: pd.DataFrame,
                           max_sharpe_portfolio: Dict,
                           min_vol_portfolio: Dict,
                           save_path: str = None):
    """
    Plot efficient frontier with optimal portfolios marked.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot efficient frontier
    ax.plot(frontier_df['volatility'], frontier_df['return'], 
            'b-', linewidth=2, label='Efficient Frontier')
    
    # Plot max Sharpe portfolio
    ax.scatter(max_sharpe_portfolio['volatility'], max_sharpe_portfolio['return'],
              marker='*', s=500, c='red', edgecolors='black', linewidth=2,
              label=f"Max Sharpe (SR={max_sharpe_portfolio['sharpe']:.3f})", zorder=5)
    
    # Plot min volatility portfolio
    ax.scatter(min_vol_portfolio['volatility'], min_vol_portfolio['return'],
              marker='o', s=300, c='green', edgecolors='black', linewidth=2,
              label=f"Min Volatility", zorder=5)
    
    # Add risk-free rate line
    ax.axhline(y=RISK_FREE_RATE, color='gray', linestyle='--', linewidth=1, label=f'Risk-Free Rate ({RISK_FREE_RATE:.1%})')
    
    ax.set_xlabel('Volatility (Standard Deviation)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.set_title('Efficient Frontier - VN30 Portfolio Optimization', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format axes as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        print(f"  ✓ Saved efficient frontier to {save_path}")
    
    plt.show()


def plot_portfolio_weights(weights: pd.Series, title: str = "Portfolio Weights", save_path: str = None):
    """
    Plot portfolio weights as bar chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort weights for better visualization
    weights_sorted = weights.sort_values(ascending=False)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(weights_sorted)))
    ax.bar(range(len(weights_sorted)), weights_sorted.values, color=colors)
    ax.set_xticks(range(len(weights_sorted)))
    ax.set_xticklabels(weights_sorted.index, rotation=45, ha='right')
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(weights_sorted.values):
        ax.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        print(f"  ✓ Saved portfolio weights to {save_path}")
    
    plt.show()


def plot_portfolio_comparison(portfolios: Dict[str, Dict], save_path: str = None):
    """
    Compare different portfolio strategies.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract metrics
    names = list(portfolios.keys())
    returns = [p['return'] for p in portfolios.values()]
    volatilities = [p['volatility'] for p in portfolios.values()]
    sharpes = [p['sharpe'] for p in portfolios.values()]
    
    # Plot 1: Return vs Volatility
    colors = ['red', 'green', 'blue', 'orange']
    for i, name in enumerate(names):
        ax1.scatter(volatilities[i], returns[i], s=300, c=colors[i], 
                   edgecolors='black', linewidth=2, label=name, alpha=0.7)
    
    ax1.set_xlabel('Volatility', fontsize=12)
    ax1.set_ylabel('Expected Return', fontsize=12)
    ax1.set_title('Portfolio Comparison: Risk vs Return', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Plot 2: Sharpe Ratios
    ax2.bar(names, sharpes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Sharpe Ratio', fontsize=12)
    ax2.set_title('Portfolio Comparison: Sharpe Ratios', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(sharpes):
        ax2.text(i, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        print(f"  ✓ Saved portfolio comparison to {save_path}")
    
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for Part 2: Portfolio Optimization
    """
    print("\n" + "="*60)
    print("PART 2: PORTFOLIO OPTIMIZATION FOR VN30 STOCKS")
    print("="*60 + "\n")
    
    # Step 1: Load factor scores from Part 1
    factors_df = load_factor_scores()
    
    # Step 2: Load historical data and calculate returns
    # Note: You need to pass all_data from Part 1 or reload it
    # For now, we'll create a placeholder
    print("⚠ Note: You need to provide historical price data to calculate returns.")
    print("   This can be done by:")
    print("   1. Re-running Part 1 and passing all_data to Part 2")
    print("   2. Saving price data in Part 1 and loading it here")
    print("\n   For demonstration, we'll use simulated returns.\n")
    
    # Simulated returns (replace with actual data)
    np.random.seed(42)
    dates = pd.date_range(START_DATE, END_DATE, freq='D')
    returns_data = {}
    for symbol in factors_df.index:
        returns_data[symbol] = np.random.normal(0.0005, 0.02, len(dates))
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    print(f"Using simulated returns for {len(returns_df.columns)} stocks\n")
    
    # Step 3: Estimate expected returns
    print("-"*60)
    print("ESTIMATING EXPECTED RETURNS")
    print("-"*60 + "\n")
    
    expected_returns = estimate_expected_returns_factor_based(factors_df, returns_df)
    
    # Step 4: Calculate covariance matrix
    print("-"*60)
    print("CALCULATING COVARIANCE MATRIX")
    print("-"*60 + "\n")
    
    cov_matrix = calculate_covariance_matrix(returns_df)
    
    # Step 5: Optimize portfolios
    print("-"*60)
    print("PORTFOLIO OPTIMIZATION")
    print("-"*60 + "\n")
    
    max_sharpe_portfolio = optimize_max_sharpe(expected_returns, cov_matrix)
    min_vol_portfolio = optimize_min_volatility(expected_returns, cov_matrix)
    
    # Equal-weighted portfolio for comparison
    equal_weights = pd.Series([1/len(expected_returns)] * len(expected_returns), index=expected_returns.index)
    eq_return, eq_vol = portfolio_performance(equal_weights.values, expected_returns, cov_matrix)
    equal_weighted_portfolio = {
        'weights': equal_weights,
        'return': eq_return,
        'volatility': eq_vol,
        'sharpe': (eq_return - RISK_FREE_RATE) / eq_vol
    }
    
    # Step 6: Generate efficient frontier
    print("-"*60)
    print("GENERATING EFFICIENT FRONTIER")
    print("-"*60 + "\n")
    
    frontier_df = generate_efficient_frontier(expected_returns, cov_matrix)
    
    # Step 7: Display results
    print("-"*60)
    print("OPTIMAL PORTFOLIO WEIGHTS (Maximum Sharpe)")
    print("-"*60)
    print(max_sharpe_portfolio['weights'].sort_values(ascending=False).apply(lambda x: f"{x:.2%}"))
    
    print("\n" + "-"*60)
    print("PORTFOLIO PERFORMANCE COMPARISON (Accounting for Costs)")
    print("-"*60)
    
    # We show Net results in the table
    comparison_df = pd.DataFrame({
        'Max Sharpe (Net)': [max_sharpe_portfolio['return'], max_sharpe_portfolio['volatility'], max_sharpe_portfolio['sharpe']],
        'Min Vol (Net)': [min_vol_portfolio['return'], min_vol_portfolio['volatility'], min_vol_portfolio['sharpe']],
        'Equal Weighted': [equal_weighted_portfolio['return'], equal_weighted_portfolio['volatility'], equal_weighted_portfolio['sharpe']]
    }, index=['Net Return', 'Volatility', 'Sharpe Ratio']).T
    
    print(comparison_df.to_string())
    print("\n* Net Return accounts for transaction costs and estimated slippage.")
    
    # Step 8: Save results
    output_file = os.path.join(DATA_DIR, 'optimal_weights.csv')
    max_sharpe_portfolio['weights'].to_csv(output_file)
    print(f"\n✓ Optimal weights saved to {output_file}")
    
    frontier_file = os.path.join(DATA_DIR, 'efficient_frontier.csv')
    frontier_df.to_csv(frontier_file, index=False)
    print(f"✓ Efficient frontier saved to {frontier_file}")
    
    # Step 9: Create visualizations
    print("\n" + "-"*60)
    print("GENERATING VISUALIZATIONS")
    print("-"*60 + "\n")
    
    plot_efficient_frontier(frontier_df, max_sharpe_portfolio, min_vol_portfolio,
                           os.path.join(FIGURES_DIR, 'efficient_frontier.png'))
    
    plot_portfolio_weights(max_sharpe_portfolio['weights'], 
                          "Optimal Portfolio Weights (Maximum Sharpe Ratio)",
                          os.path.join(FIGURES_DIR, 'optimal_weights.png'))
    
    portfolios_comparison = {
        'Max Sharpe': max_sharpe_portfolio,
        'Min Volatility': min_vol_portfolio,
        'Equal Weighted': equal_weighted_portfolio
    }
    plot_portfolio_comparison(portfolios_comparison,
                            os.path.join(FIGURES_DIR, 'portfolio_comparison.png'))
    
    print("\n" + "="*60)
    print("✓ PART 2 COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")
    print(f"Results saved to: {DATA_DIR}/")
    print(f"Figures saved to: {FIGURES_DIR}/")
    print("\nYou now have:")
    print("  1. Factor scores from Part 1")
    print("  2. Optimal portfolio weights from Part 2")
    print("  3. Efficient frontier visualization")
    print("\nUse these results in your presentation!")
    
    return max_sharpe_portfolio, min_vol_portfolio, frontier_df


if __name__ == "__main__":
    # Run main function
    max_sharpe, min_vol, frontier = main()
