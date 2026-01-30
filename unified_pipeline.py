"""
Unified VN30 Quantitative Pipeline
====================================
Orchestrates: Raw Data -> Data Cleaning -> EDA -> Factor Models -> Backtest.
"""

import os
import logging
import pandas as pd
from datetime import datetime

# Import Stages
from vn30_data_cleaning import QuantitativeDataEngineer
from eda_vn30_analysis import main as run_eda
from part1_multifactor_model import main as run_factor_model
from backtest_engine import run_wfa_backtest

# Configuration
from config import XNOAPI_KEY, VN30_SYMBOLS, START_DATE, END_DATE

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    start_time = datetime.now()
    logging.info("="*80)
    logging.info("STARTING UNIFIED QUANTITATIVE PIPELINE: VN30")
    logging.info("="*80)

    # STAGE 1 & 2: Raw Data & Data Cleaning
    logging.info("\n[STAGE 1 & 2] DATA ACQUISITION & QUANTITATIVE CLEANING")
    engineer = QuantitativeDataEngineer(VN30_SYMBOLS, XNOAPI_KEY)
    engineer.fetch_all_data(START_DATE, END_DATE)
    cleaned_data = engineer.clean_and_align()
    logging.info("✓ Data Cleaning Stage Completed.")

    # STAGE 3: Exploratory Data Analysis (EDA)
    logging.info("\n[STAGE 3] EXPLORATORY DATA ANALYSIS")
    run_eda(cleaned_data)
    logging.info("✓ EDA Stage Completed. Reports saved to output/eda_vn30/")

    # STAGE 4: Multi-Factor Model
    logging.info("\n[STAGE 4] MULTI-FACTOR MODELING")
    factors_df, _ = run_factor_model(cleaned_data)
    logging.info("✓ Factor Modeling Stage Completed. Scores saved to output/data/factor_scores.csv")

    # STAGE 5: Walk-Forward Backtest
    logging.info("\n[STAGE 5] WALK-FORWARD BACKTEST ENGINE")
    strategy_returns, weights_history = run_wfa_backtest(cleaned_data)
    
    # Save Backtest Results
    os.makedirs('output/robust', exist_ok=True)
    strategy_returns.to_csv('output/robust/wfa_returns.csv')
    weights_history.to_csv('output/robust/wfa_weights.csv')
    logging.info("✓ Backtest Stage Completed. Results saved to output/robust/")

    # FINAL SUMMARY
    duration = datetime.now() - start_time
    logging.info("\n" + "="*80)
    logging.info(f"UNIFIED PIPELINE COMPLETED SUCCESSFULLY IN {duration}")
    logging.info("="*80)
    
    # Calculate and Display Final Metrics
    from robust_utils import calculate_performance_metrics
    metrics = calculate_performance_metrics(strategy_returns, rf_rate=0.02)
    logging.info(f"Performance Metrics (OOS):")
    logging.info(f"  Annualized Return: {metrics['ann_return']:.2%}")
    logging.info(f"  Annualized Vol: {metrics['ann_vol']:.2%}")
    logging.info(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
    logging.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    logging.info(f"  Calmar Ratio: {metrics['calmar']:.2f}")

if __name__ == "__main__":
    main()
