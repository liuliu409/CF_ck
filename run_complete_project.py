"""
Integrated Runner for Financial Computing Project
==================================================

This script runs both Part 1 (Multi-Factor Model) and Part 2 (Portfolio Optimization)
in sequence, passing data between them.

Usage:
    python run_complete_project.py


"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*80)
print(" "*20 + "FINANCIAL COMPUTING PROJECT")
print(" "*15 + "Multi-Factor Model & Portfolio Optimization")
print(" "*25 + "VN30 Stock Analysis")
print("="*80 + "\n")

# =============================================================================
# PART 1: MULTI-FACTOR MODEL
# =============================================================================

print("\n" + "█"*80)
print("█" + " "*78 + "█")
print("█" + " "*25 + "PART 1: MULTI-FACTOR MODEL" + " "*27 + "█")
print("█" + " "*78 + "█")
print("█"*80 + "\n")

try:
    from part1_multifactor_model import main as part1_main
    factors_df, all_data = part1_main()
    print("\n✓ Part 1 completed successfully!\n")
except Exception as e:
    print(f"\n❌ Error in Part 1: {e}\n")
    print("Please check:")
    print("  1. xnoapi is installed: pip install xnoapi")
    print("  2. API key is correct in config.py")
    print("  3. Internet connection is available")
    sys.exit(1)

# =============================================================================
# PART 2: PORTFOLIO OPTIMIZATION
# =============================================================================

print("\n" + "█"*80)
print("█" + " "*78 + "█")
print("█" + " "*22 + "PART 2: PORTFOLIO OPTIMIZATION" + " "*26 + "█")
print("█" + " "*78 + "█")
print("█"*80 + "\n")

try:
    # Import Part 2 functions
    from part2_portfolio_optimization import (
        calculate_returns_from_data,
        estimate_expected_returns_factor_based,
        calculate_covariance_matrix,
        optimize_max_sharpe,
        optimize_min_volatility,
        generate_efficient_frontier,
        plot_efficient_frontier,
        plot_portfolio_weights,
        plot_portfolio_comparison
    )
    from config import RISK_FREE_RATE, DATA_DIR, FIGURES_DIR
    
    print("-"*80)
    print("CALCULATING RETURNS FROM HISTORICAL DATA")
    print("-"*80 + "\n")
    
    # Calculate returns from actual data (not simulated)
    returns_df = calculate_returns_from_data(all_data)
    
    print("-"*80)
    print("ESTIMATING EXPECTED RETURNS")
    print("-"*80 + "\n")
    
    # Estimate expected returns using factor scores
    expected_returns = estimate_expected_returns_factor_based(factors_df, returns_df)
    
    print("-"*80)
    print("CALCULATING COVARIANCE MATRIX")
    print("-"*80 + "\n")
    
    # Calculate covariance matrix
    cov_matrix = calculate_covariance_matrix(returns_df)
    
    print("-"*80)
    print("PORTFOLIO OPTIMIZATION")
    print("-"*80 + "\n")
    
    # Optimize portfolios
    max_sharpe_portfolio = optimize_max_sharpe(expected_returns, cov_matrix)
    min_vol_portfolio = optimize_min_volatility(expected_returns, cov_matrix)
    
    # Equal-weighted portfolio for comparison
    import pandas as pd
    import numpy as np
    from part2_portfolio_optimization import portfolio_performance
    
    equal_weights = pd.Series([1/len(expected_returns)] * len(expected_returns), 
                             index=expected_returns.index)
    eq_return, eq_vol = portfolio_performance(equal_weights.values, expected_returns, cov_matrix)
    equal_weighted_portfolio = {
        'weights': equal_weights,
        'return': eq_return,
        'volatility': eq_vol,
        'sharpe': (eq_return - RISK_FREE_RATE) / eq_vol
    }
    
    print("-"*80)
    print("GENERATING EFFICIENT FRONTIER")
    print("-"*80 + "\n")
    
    # Generate efficient frontier
    frontier_df = generate_efficient_frontier(expected_returns, cov_matrix)
    
    print("-"*80)
    print("RESULTS SUMMARY")
    print("-"*80 + "\n")
    
    print("OPTIMAL PORTFOLIO WEIGHTS (Maximum Sharpe Ratio):")
    print("-"*80)
    weights_display = max_sharpe_portfolio['weights'].sort_values(ascending=False)
    for symbol, weight in weights_display.items():
        bar_length = int(weight * 100)
        bar = "█" * bar_length
        print(f"  {symbol:6s}: {weight:6.2%}  {bar}")
    
    print("\n" + "-"*80)
    print("PORTFOLIO PERFORMANCE COMPARISON")
    print("-"*80)
    
    comparison_df = pd.DataFrame({
        'Max Sharpe': [
            max_sharpe_portfolio['return'], 
            max_sharpe_portfolio['volatility'], 
            max_sharpe_portfolio['sharpe']
        ],
        'Min Volatility': [
            min_vol_portfolio['return'], 
            min_vol_portfolio['volatility'], 
            min_vol_portfolio['sharpe']
        ],
        'Equal Weighted': [
            equal_weighted_portfolio['return'], 
            equal_weighted_portfolio['volatility'], 
            equal_weighted_portfolio['sharpe']
        ]
    }, index=['Expected Return', 'Volatility', 'Sharpe Ratio']).T
    
    print(comparison_df.to_string())
    
    # Save results
    print("\n" + "-"*80)
    print("SAVING RESULTS")
    print("-"*80 + "\n")
    
    output_file = os.path.join(DATA_DIR, 'optimal_weights.csv')
    max_sharpe_portfolio['weights'].to_csv(output_file)
    print(f"✓ Optimal weights saved to {output_file}")
    
    frontier_file = os.path.join(DATA_DIR, 'efficient_frontier.csv')
    frontier_df.to_csv(frontier_file, index=False)
    print(f"✓ Efficient frontier saved to {frontier_file}")
    
    # Create visualizations
    print("\n" + "-"*80)
    print("GENERATING VISUALIZATIONS")
    print("-"*80 + "\n")
    
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
    
    print("\n✓ Part 2 completed successfully!\n")
    
except Exception as e:
    print(f"\n❌ Error in Part 2: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print(" "*30 + "PROJECT COMPLETED!")
print("="*80 + "\n")

print("📊 DELIVERABLES:")
print("-"*80)
print(f"  ✓ Factor Scores: {DATA_DIR}/factor_scores.csv")
print(f"  ✓ Optimal Weights: {DATA_DIR}/optimal_weights.csv")
print(f"  ✓ Efficient Frontier: {DATA_DIR}/efficient_frontier.csv")
print()
print(f"  ✓ Factor Heatmap: {FIGURES_DIR}/factor_heatmap.png")
print(f"  ✓ Factor Rankings: {FIGURES_DIR}/factor_rankings.png")
print(f"  ✓ Factor Correlation: {FIGURES_DIR}/factor_correlation.png")
print(f"  ✓ Efficient Frontier Plot: {FIGURES_DIR}/efficient_frontier.png")
print(f"  ✓ Optimal Weights Chart: {FIGURES_DIR}/optimal_weights.png")
print(f"  ✓ Portfolio Comparison: {FIGURES_DIR}/portfolio_comparison.png")

print("\n" + "="*80)
print(" "*25 + "NEXT STEPS FOR PRESENTATION")
print("="*80)
print("""
1. Review the presentation outline in:
   /home/hmachine/.gemini/antigravity/brain/.../presentation_outline.md

2. Use the generated figures in your slides:
   - All figures are saved in output/figures/
   - High resolution (300 DPI) for professional quality

3. Key results to highlight:
   - Top 3 stocks by factor score (from factor_scores.csv)
   - Optimal portfolio weights (from optimal_weights.csv)
   - Sharpe ratio improvement vs equal-weighted portfolio

4. Customize the presentation:
   - Add your university branding
   - Translate to Vietnamese if needed
   - Add additional analysis as required

5. For the code demonstration:
   - Show key functions from part1_multifactor_model.py
   - Show optimization code from part2_portfolio_optimization.py
   - Explain the transition between Part 1 and Part 2

""")

print("="*80 + "\n")
