"""
Configuration file for Financial Computing Project
VN30 Multi-Factor Model and Portfolio Optimization
"""

# =============================================================================
# API CONFIGURATION
# =============================================================================

# xnoapi API Key
XNOAPI_KEY = "niZhzM_MG4nmnpIJlNWasV85_dyyrr3DnWKR3MJlcOatWcCAQfQRGg6MGsTDIECaKgYAedqwsZ1Nidr2iFm9Ekbi2eDpJiplZEiVgtZydDbJXViJ.SoCI5Oh5r33Rc22"

# =============================================================================
# STOCK UNIVERSE
# =============================================================================

# VN30 Stock List (Full 30 stocks)
VN30_SYMBOLS = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
    'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]

# =============================================================================
# DATE RANGES
# =============================================================================

# Analysis period
START_DATE = "2016-01-01"
END_DATE = "2026-01-29"

# Training/Testing split for backtesting
TRAIN_START = "2016-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2026-01-29"

# =============================================================================
# FACTOR MODEL PARAMETERS
# =============================================================================

# Factor weights (updated for 4-factor model including Quality)
FACTOR_WEIGHTS = {
    'size': 0.25,
    'value': 0.25,
    'momentum': 0.25,
    'quality': 0.25
}

# VN30 Sector Classifications (Full 30 stocks)
STOCK_SECTORS = {
    'ACB': 'Banking', 'BID': 'Banking', 'CTG': 'Banking', 'HDB': 'Banking',
    'MBB': 'Banking', 'SHB': 'Banking', 'SSB': 'Banking', 'STB': 'Banking',
    'TCB': 'Banking', 'TPB': 'Banking', 'VCB': 'Banking', 'VIB': 'Banking',
    'VPB': 'Banking',
    'VHM': 'Real Estate', 'VIC': 'Real Estate', 'VRE': 'Real Estate',
    'MSN': 'Consumer', 'MWG': 'Consumer', 'SAB': 'Consumer', 'VJC': 'Consumer',
    'VNM': 'Consumer',
    'BCM': 'Industrial', 'GVR': 'Industrial', 'HPG': 'Industrial',
    'GAS': 'Utilities', 'PLX': 'Utilities', 'POW': 'Utilities',
    'FPT': 'Technology',
    'BVH': 'Financial', 'SSI': 'Financial'
}

# Momentum calculation parameters
MOMENTUM_LOOKBACK = 252  # 12 months (trading days)
MOMENTUM_SKIP_LAST = 21  # Skip last month to avoid reversal

# Value factor choice: 'ep' (Earnings Yield) or 'pb' (Price-to-Book)
VALUE_METRIC = 'ep' 

# Quality factor choice: 'low_vol' (Low Volatility)
QUALITY_METRIC = 'low_vol' 

# =============================================================================
# PORTFOLIO OPTIMIZATION PARAMETERS
# =============================================================================

# Risk-free rate (annual)
RISK_FREE_RATE = 0.02  # 2% annual

# Optimization constraints
MIN_WEIGHT = 0.0  # Minimum weight per stock
MAX_WEIGHT = 0.15 # Maximum 15% per stock (VN-recommended cap)

# Number of portfolios to generate for efficient frontier
N_PORTFOLIOS = 100

# Rebalancing frequency (for backtesting)
REBALANCE_FREQ = 'Q'  # 'M' = Monthly, 'Q' = Quarterly, 'Y' = Yearly

# =============================================================================
# TRANSACTION COSTS
# =============================================================================

# Transaction cost (percentage)
TRANSACTION_COST = 0.003  # 0.30% (30bps) one-way baseline

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# Output directories
OUTPUT_DIR = "output"
FIGURES_DIR = "output/figures"
DATA_DIR = "output/data"

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'  # 'png', 'pdf', 'svg'

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

# Number of decimal places for display
DECIMAL_PLACES = 4

# Pandas display options
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
# =============================================================================
# ROBUST BACKTESTING PARAMETERS
# =============================================================================

# Training window for rolling estimation (Years)
LOOKBACK_WINDOW = 3 

# Forecast horizon for IC estimation (Days)
FORECAST_HORIZON = 21 # 1 month ahead

# Turnover constraint (Maximum allowed absolute weight change per rebalance)
# 0.2 means no more than 20% of the portfolio can be flipped
TURNOVER_LIMIT = 0.5 

# Transaction cost assumption (Percentage)
# Includes brokerage, spread, and taxes for VN market
TRANSACTION_COST_MODEL = 0.003 # 0.30% (30bps one-way)

# Turnover Penalty (lambda for objective function)
TURNOVER_PENALTY_LAMBDA = 0.015

# Rebalancing Frequency
REBALANCE_FREQ = 'Q' # Quarterly rebalancing

