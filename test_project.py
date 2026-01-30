"""
Quick Test Script for Financial Computing Project
==================================================

This script performs basic validation without requiring actual data fetching.
Tests the core functions and ensures code structure is correct.

Run with: conda activate vistral-4bit && python test_project.py
"""

import sys
import os
import numpy as np
import pandas as pd

print("\n" + "="*60)
print("FINANCIAL COMPUTING PROJECT - VALIDATION TEST")
print("="*60 + "\n")

# Test 1: Import all modules
print("Test 1: Importing modules...")
try:
    import config
    print("  ✓ config.py imported")
    
    # Import Part 1 functions (without running main)
    from part1_multifactor_model import (
        calculate_size_factor,
        calculate_value_factor,
        calculate_momentum_factor,
        calculate_composite_score
    )
    print("  ✓ part1_multifactor_model.py imported")
    
    # Import Part 2 functions
    from part2_portfolio_optimization import (
        estimate_expected_returns_historical,
        estimate_expected_returns_factor_based,
        calculate_covariance_matrix,
        optimize_max_sharpe,
        portfolio_performance
    )
    print("  ✓ part2_portfolio_optimization.py imported")
    
    print("  ✓ All modules imported successfully!\n")
except Exception as e:
    print(f"  ✗ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Verify configuration
print("Test 2: Checking configuration...")
try:
    assert len(config.VN30_SYMBOLS) == 10, "Should have 10 stocks"
    assert config.RISK_FREE_RATE >= 0, "Risk-free rate should be non-negative"
    assert config.MIN_WEIGHT >= 0, "Min weight should be non-negative"
    assert config.MAX_WEIGHT <= 1, "Max weight should be <= 1"
    assert sum(config.FACTOR_WEIGHTS.values()) == 1.0, "Factor weights should sum to 1"
    
    print(f"  ✓ VN30 stocks: {config.VN30_SYMBOLS}")
    print(f"  ✓ Factor weights: {config.FACTOR_WEIGHTS}")
    print(f"  ✓ Risk-free rate: {config.RISK_FREE_RATE:.1%}")
    print(f"  ✓ Weight constraints: [{config.MIN_WEIGHT:.1%}, {config.MAX_WEIGHT:.1%}]")
    print("  ✓ Configuration validated!\n")
except Exception as e:
    print(f"  ✗ Configuration error: {e}\n")
    sys.exit(1)

# Test 3: Test factor calculations with dummy data
print("Test 3: Testing factor calculations with dummy data...")
try:
    # Create dummy data
    np.random.seed(42)
    symbols = config.VN30_SYMBOLS
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    
    all_data = {}
    for symbol in symbols:
        df = pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(len(dates)).cumsum(),
            'high': 105 + np.random.randn(len(dates)).cumsum(),
            'low': 95 + np.random.randn(len(dates)).cumsum(),
            'close': 100 + np.random.randn(len(dates)).cumsum(),
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'symbol': symbol
        })
        df['close'] = df['close'].abs() + 50  # Ensure positive prices
        all_data[symbol] = df
    
    # Test size factor
    size_factor = calculate_size_factor(all_data)
    assert len(size_factor) == 10, "Size factor should have 10 values"
    print(f"  ✓ Size factor calculated: mean={size_factor.mean():.3f}, std={size_factor.std():.3f}")
    
    # Test value factor
    value_factor = calculate_value_factor(all_data)
    assert len(value_factor) == 10, "Value factor should have 10 values"
    print(f"  ✓ Value factor calculated: mean={value_factor.mean():.3f}, std={value_factor.std():.3f}")
    
    # Test momentum factor
    momentum_factor = calculate_momentum_factor(all_data)
    assert len(momentum_factor) == 10, "Momentum factor should have 10 values"
    print(f"  ✓ Momentum factor calculated: mean={momentum_factor.mean():.3f}, std={momentum_factor.std():.3f}")
    
    # Test composite score
    factors_df = calculate_composite_score(size_factor, value_factor, momentum_factor)
    assert len(factors_df) == 10, "Composite score should have 10 rows"
    assert 'composite_score' in factors_df.columns, "Should have composite_score column"
    print(f"  ✓ Composite scores calculated")
    print(f"    Top stock: {factors_df.index[0]} (score: {factors_df['composite_score'].iloc[0]:.3f})")
    print(f"    Bottom stock: {factors_df.index[-1]} (score: {factors_df['composite_score'].iloc[-1]:.3f})")
    
    print("  ✓ Factor calculations working!\n")
except Exception as e:
    print(f"  ✗ Factor calculation error: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test portfolio optimization with dummy data
print("Test 4: Testing portfolio optimization with dummy data...")
try:
    # Create dummy returns
    returns_data = {}
    for symbol in symbols:
        returns_data[symbol] = np.random.normal(0.0005, 0.02, len(dates))
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # Test expected returns calculation
    expected_returns_hist = estimate_expected_returns_historical(returns_df)
    assert len(expected_returns_hist) == 10, "Should have 10 expected returns"
    print(f"  ✓ Historical expected returns: mean={expected_returns_hist.mean():.2%}")
    
    # Test factor-based expected returns
    expected_returns_factor = estimate_expected_returns_factor_based(factors_df, returns_df)
    assert len(expected_returns_factor) == 10, "Should have 10 factor-based returns"
    print(f"  ✓ Factor-based expected returns: mean={expected_returns_factor.mean():.2%}")
    
    # Test covariance matrix
    cov_matrix = calculate_covariance_matrix(returns_df)
    assert cov_matrix.shape == (10, 10), "Covariance matrix should be 10x10"
    print(f"  ✓ Covariance matrix calculated: shape={cov_matrix.shape}")
    
    # Test portfolio optimization
    max_sharpe_portfolio = optimize_max_sharpe(expected_returns_factor, cov_matrix)
    assert 'weights' in max_sharpe_portfolio, "Should have weights"
    assert abs(max_sharpe_portfolio['weights'].sum() - 1.0) < 0.01, "Weights should sum to 1"
    assert all(max_sharpe_portfolio['weights'] >= 0), "All weights should be non-negative"
    
    print(f"  ✓ Max Sharpe portfolio optimized:")
    print(f"    Expected return: {max_sharpe_portfolio['return']:.2%}")
    print(f"    Volatility: {max_sharpe_portfolio['volatility']:.2%}")
    print(f"    Sharpe ratio: {max_sharpe_portfolio['sharpe']:.3f}")
    print(f"    Weights sum: {max_sharpe_portfolio['weights'].sum():.4f}")
    
    print("  ✓ Portfolio optimization working!\n")
except Exception as e:
    print(f"  ✗ Portfolio optimization error: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check output directories
print("Test 5: Checking output directories...")
try:
    assert os.path.exists(config.OUTPUT_DIR), f"{config.OUTPUT_DIR} should exist"
    assert os.path.exists(config.FIGURES_DIR), f"{config.FIGURES_DIR} should exist"
    assert os.path.exists(config.DATA_DIR), f"{config.DATA_DIR} should exist"
    
    print(f"  ✓ Output directory: {config.OUTPUT_DIR}")
    print(f"  ✓ Figures directory: {config.FIGURES_DIR}")
    print(f"  ✓ Data directory: {config.DATA_DIR}")
    print("  ✓ All directories exist!\n")
except Exception as e:
    print(f"  ✗ Directory error: {e}\n")
    sys.exit(1)

# Summary
print("="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nThe project code is syntactically correct and functional.")
print("\nNext steps:")
print("  1. Run with real data: conda activate vistral-4bit && python run_complete_project.py")
print("  2. Or run parts separately:")
print("     - Part 1: python part1_multifactor_model.py")
print("     - Part 2: python part2_portfolio_optimization.py")
print("\nNote: You may need to update the P/B ratios in part1_multifactor_model.py")
print("      with actual fundamental data for accurate results.")
print("\n" + "="*60 + "\n")
