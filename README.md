# VN30 Quantitative Equity Platform

A state-of-the-art quantitative finance framework for VN30 stocks, evolving from a static prototype into a **Robust Walk-Forward Analysis (WFA)** system.

## 🚀 Major Milestones Achieved
- [x] **Walk-Forward Analysis (WFA)**: Robust out-of-sample backtesting framework.
- [x] **Portfolio Optimization**: Mean-Variance Optimization with Ledoit-Wolf shrinkage.
- [x] **Combined Alpha Factors**: 4-factor model (Size, Value, Momentum, Quality) with Sector Neutralization.
- [x] **Robust Value Factor**: Earnings Yield (E/P) based value signal with sector-neutralization.
- [x] **Risk Management**: Explicit turnover constraints and transaction cost modeling.

## 📋 Project Overview
This project implements a complete quantitative investment pipeline:
- **Part 1 (Factors)**: A 4-factor model (Size, Value, Momentum, Quality) with **Sector Neutralization**.
- **Part 2 (Optimization)**: Mean-Variance Optimization (MVO) using **Ledoit-Wolf Shrinkage** and **Turnover Constraints**.
- **Framework**: Robust **Walk-Forward Analysis** to eliminate look-ahead bias and ensure out-of-sample validity.

## 🎯 Objectives
1. **Eliminate Bias**: Use rolling windows to ensure the model never "sees" the future during backtesting.
2. **Isolate Pure Alpha**: Neutralize factor scores against industry groups to remove sector-wide noise.
3. **Stabilize Risk**: Apply shrinkage to the covariance matrix to prevent extreme, unstable portfolio weights.
4. **Statistical Rigor**: Map expected returns using the **Information Coefficient (IC)** rather than heuristic guesses.

## 📊 Data
- **Universe**: Full VN30 (30 stocks including ACB, FPT, HPG, VHM, VIC, VNM, etc.)
- **Data Source**: xnoapi (Vietnamese stock market data API)
- **Period**: 10 Years (2016 - 2026)

## 🛠 Project Components
1.  **[backtest_engine.py](./backtest_engine.py)**: The core Walk-Forward Analysis rebalancing loop.
2.  **[robust_utils.py](./robust_utils.py)**: Mathematical kernels for shrinkage, IC, and neutralization.
3.  **[config.py](./config.py)**: Central configuration for symbols, sectors, and hyperparameters.
4.  **[run_complete_project.py](./run_complete_project.py)**: Static prototype for quick full-sample analysis.

## 🚀 Quick Start
### Prerequisites
```bash
# Core dependencies
pip install pandas numpy matplotlib seaborn scipy xnoapi
# Advanced quantitative tools
pip install scikit-learn cvxpy
```

### Running the Project
**Option 1: Robust Build (Recommended for Final Submission)**
*Implements Walk-Forward Analysis, Shrinkage, and Turnover Constraints.*
```bash
python backtest_engine.py
```

**Option 2: Standard Build (Prototype)**
*Full-sample static analysis for quick overview.*
```bash
python run_complete_project.py
```

## 🔬 Advanced Methodology
### 1. Factor Evolution (Pure Alpha)
Instead of raw scores, we now use **Sector-Neutralization**. By de-meaning scores within industries (e.g., comparing a Bank to other Banks, not to a Tech firm), we isolate idiosyncratic company strength.
- **Size**: Preference for smaller market cap within sectors.
- **Value**: High Earnings Yield (E/P) relative to industry peers.
- **Momentum**: 12-month relative strength.
- **Quality**: Low-volatility anomaly (Low Vol = High Quality).

### 2. Risk & Return Modeling
- **Ledoit-Wolf Shrinkage**: Pulls the covariance matrix toward a target, reducing estimation error and diversification risk.
- **IC-Return Mapping**: $E[r] \propto \sigma \times \text{Score} \times \text{IC}$. Return forecasts are now anchored in the factor's actual historical predictive accuracy.
- **Turnover Limits**: An L1-norm constraint in the optimizer limits rebalancing to 50% per quarter, preventing alpha erosion from transaction fees.

## 📈 Performance Summary (Full 30-Stock Universe)

| Metric | Static Prototype (30stk) | Basic WFA (Robust) | Advanced WFA (Pure Alpha) |
|--------|--------------------------|-------------------|--------------------------|
| **Annualized Return** | 16.37% | 13.17% | 20.16% |
| **Annualized Vol** | 22.45% | 20.42% | 16.87% |
| **Sharpe Ratio** | **0.64** | **0.55** | **1.08** |
| **Max Drawdown** | -24.12% | -21.05% | **-14.96%** |
| **Calmar Ratio** | 0.68 | 0.63 | **1.35** |

*Note: The Advanced WFA (Pure Alpha) model leverages sector-neutralization, Earnings Yield (E/P), and Ledoit-Wolf shrinkage across all 30 assets, yielding a superior 1.08 Sharpe Ratio and resilient risk metrics (1.35 Calmar).*

## 🎓 Academic Deliverables
1.  **QUANT_AUDIT.md**: A critical assessment of why the robust model is superior to the static prototype.
2.  **Presentation Outline**: 30-slide structure covering methodology, transition, and results.
3.  **Visualization Suite**: High-resolution charts in `output/figures/` optimized for slide insertion.

---

