# Presentation Outline: VN30 Multi-Factor Quantitative Strategy

## Section 1: Introduction & Objective
- **Goal**: Build a robust investment strategy for the VN30 universe.
- **Key Challenges**: Sector bias (Banking/Real Estate), extreme volatility ("Fat Tails"), and backtest overfitting.
- **Solution**: A unified pipeline integrating Data Cleaning, Multi-Factor Logic, and Walk-Forward Analysis.

## Section 2: Data & Exploratory Analysis
- **Universe**: 30 stocks from 2016-2026.
- **Data Cleaning**: Winsorization at 1%/99% level to manage outliers.
- **Key Findings**: 
    - High kurtosis in VN stocks confirms the need for robust covariance estimation (Ledoit-Wolf).
    - Sector-neutral correlation is essential for true diversification.

## Section 3: The 4-Factor Model
- **Size**: Small-cap bias within sectors.
- **Value**: High Earnings Yield (E/P).
- **Momentum**: 1-year relative strength.
- **Quality**: Low-volatility preference.
- **Neutralization**: Cross-sectional de-meaning within categories to isolate alpha.

## Section 4: Portfolio Optimization & Backtesting
- **Optimization**: Mean-Variance with Turnover Constraints (50% max/quarter).
- **Robustness**: Ledoit-Wolf shrinkage to stabilize the covariance matrix.
- **Verification**: Walk-Forward Analysis (WFA) to ensure out-of-sample performance.

## Section 5: Final Results
| Metric | Performance |
| :--- | :--- |
| **Ann. Return** | 20.16% |
| **Sharpe Ratio** | 1.08 |
| **Max Drawdown** | -14.96% |
| **Calmar Ratio** | 1.35 |

## Section 6: Conclusion
- **Top Picks**: SHB, MBB, POW, HPG, VHM.
- **Summary**: The model significantly outperforms the equal-weighted benchmark by isolating factor alpha and managing sector risk.
