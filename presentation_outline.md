# VN30 Alpha Engine: A Multi-Factor Framework with Risk Optimization

## Slide 1: Title Slide
- **Title**: VN30 Alpha Engine: A Multi-Factor Framework with Risk Optimization
- **Subtitle**: Evolving from Static Prototypes to Robust Walk-Forward Analysis
- **Bullets**: 
    - Quantitative exploration of the Vietnamese Blue-Chip Universe (VN30).
    - Advanced Factor Engineering & Portfolio Optimization.
    - Presenter: [Your Name/Senior Quantitative Researcher].
- **Recommended Visual**: High-resolution branding logo of the project or a professional abstract background representing financial data connectivity.

## Slide 2: Market Overview: Why the VN30?
- **Title**: The VN30 Universe: Opportunities & Risks
- **Bullets**:
    - Focus on the 30 most liquid and largest-cap stocks on the HOSE.
    - Represents the heartbeat of the Vietnamese emerging economy.
    - Challenges: High retail participation, sector concentration (Banks/Real Estate), and liquidity shifts.
- **Recommended Visual**: Pie chart showing the sector distribution of the VN30 (Financials, Real Estate, Industrials, etc.).

## Slide 3: The Problem Statement: Why Static Models Fail
- **Title**: Beyond Static Models: The Reality of Market Regime Shifts
- **Bullets**:
    - Static prototypes suffer from **Look-Ahead Bias** and **Overfitting**.
    - Inability to adapt to structural changes (e.g., Post-COVID volatility).
    - "Garbage In, Garbage Out": Raw data without cleaning leads to unstable weights.
- **Recommended Visual**: A "Signal vs. Noise" conceptual diagram showing how raw data creates erratic portfolio rebalancing.

## Slide 4: Data Integrity: Universe & Period
- **Title**: Data Infrastructure: 10 Years of VN30 History
- **Bullets**:
    - **Timeframe**: 2016 – 2026 (Historical + Projected/Current).
    - **Source**: Integrated xnoapi client for high-fidelity daily OHLCV data.
    - **Inventory**: 30 Symbols (ACB, FPT, HPG, VHM, etc.) with full corporate action adjustments.
- **Recommended Visual**: Timeline graphic showing the 10-year span and a table listing top 10 tickers by liquidity.

## Slide 5: EDA: Handling 'Fat Tails' (Kurtosis)
- **Title**: Exploratory Data Analysis: Navigating Fat-Tail Risks
- **Bullets**:
    - VN30 returns exhibit high **Kurtosis** (Values > 5.0), signifying extreme outlier frequency.
    - Non-normal distributions make standard variance estimations dangerous.
    - Insight: Individual stock volatility is high (~20-25% annualized), necessitating robust covariance.
- **Recommended Visual**: A Histogram of VN30 daily returns overlaid with a Normal Distribution curve to highlight the "Fat Tails."

## Slide 6: Data Cleaning: Winsorization & Integrity
- **Title**: Data Hygiene: Winsorization & Price Reconstruction
- **Bullets**:
    - **Winsorization**: Capping outliers at 1st and 99th percentiles to prevent signal distortion.
    - Handling "Zero-Volume" days and missing price points via forward-filling logic.
    - Sector-median imputation for missing fundamental ratios (E/P).
- **Recommended Visual**: "Before vs. After" scatter plot showing how Winsorization pulls extreme outliers back to a manageable range.

## Slide 7: Alpha Engine (Part 1): Size & Value Factors
- **Title**: The Factor Core: Size (Small-Cap) & Value (E/P)
- **Bullets**:
    - **Size**: Captures the "Small-Cap Premium" within the VN30 universe.
    - **Value (Earnings Yield)**: Uses E/P (LTM) to identify undervalued cash-flow generators.
    - Rationale: E/P is more stable in VN than P/B due to asset revaluation complexities.
- **Recommended Visual**: A bar chart showing the cross-sectional distribution of Size and Value scores for the current top 5 vs. bottom 5 stocks.

## Slide 8: Alpha Engine (Part 2): Momentum & Quality
- **Title**: The Factor Core: Momentum & Quality (Low Vol)
- **Bullets**:
    - **Momentum**: 12-month relative strength (skipping the most recent month to avoid reversal).
    - **Quality**: Low-volatility anomaly; stocks with stable returns often outperform in VN risk-adjusted.
    - Composite Scoring: Z-score normalization enables equal-scale comparison across factors.
- **Recommended Visual**: A 2x2 grid showing the performance of Top Decile Momentum vs. Low Volatility stocks.

## Slide 10: Portfolio Construction: The MVO Framework
- **Title**: Mean-Variance Optimization (MVO)
- **Bullets**:
    - Objective: Maximize expected return for a targeted level of risk.
    - Dynamic Input: Alpha factor scores mapped to Expected Returns (E[r]).
    - Output: Optimized weights that respect the "Efficient Frontier."
- **Recommended Visual**: The Efficient Frontier curve, highlighting the "Global Minimum Variance" and "Tangency" portfolios.

## Slide 11: Covariance Stability: Ledoit-Wolf Shrinkage
- **Title**: Risk Stability: Ledoit-Wolf Shrinkage
- **Bullets**:
    - Standard Covariance is noisy when N (assets) is large relative to T (observations).
    - **Ledoit-Wolf**: Pulls the sample covariance toward a constant correlation matrix.
    - Result: More stable weights and significantly reduced turnover at the edges.
- **Recommended Visual**: Heatmap comparison of the Raw Covariance Matrix vs. the Shrinkage Covariance Matrix.

## Slide 12: Realistic Constraints: Turnover & Costs
- **Title**: Execution Realism: Turnover & Slippage Constraints
- **Bullets**:
    - **Turnover Limit**: Capped at 50% per quarter to prevent excessive trading.
    - **Transaction Costs**: modeled at 15-20bps (including taxes/fees/slippage) per leg.
    - Constraints prevent "Paper Profits" that disappear in actual execution.
- **Recommended Visual**: Line chart showing "Gross Return" vs. "Net Return" after accounting for 50% quarterly turnover costs.

## Slide 13: Robustness Testing: Walk-Forward Analysis (WFA)
- **Title**: The Golden Standard: Walk-Forward Analysis (WFA)
- **Bullets**:
    - Rolling Window Approach: 24-month training followed by 6-month out-of-sample testing.
    - Simulates the real-world process of re-training models periodically.
    - Unlike static splits, WFA reflects how the model adapts to evolving market regimes.
- **Recommended Visual**: A staggered "Box Diagram" showing the rolling windows of Train vs. OOS Test periods.

## Slide 14: Eliminating Look-Ahead Bias
- **Title**: Scientific Rigor: Eliminating Bias & Data Leakage
- **Bullets**:
    - Point-in-time data usage: Model only sees data available at the rebalance date.
    - No "future-peeking" in normalization (Z-scores are cross-sectional per window).
    - Rigorous validation ensures the 1.08 Sharpe isn't a result of "Post-hoc" filtering.
- **Recommended Visual**: A checklist icon graphic with "NO Look-ahead", "NO Overfitting", "NO Survival Bias" checked.

## Slide 15: Key Results: Advanced WFA vs. Prototype
- **Title**: Performance Results: The "Pure Alpha" Advantage
- **Bullets**:
    - **Sharpe Ratio**: 1.08 (vs. 0.64 in the Prototype).
    - **Annualized Return**: 20.16%.
    - **Calmar Ratio**: 1.35; indicating high return per unit of maximum drawdown.
- **Recommended Visual**: A comparative bar chart showing Metrics (Sharpe, Return, Vol) side-by-side for Prototype vs. Advanced WFA.

## Slide 16: Benchmarking: Strategy vs. Equal-Weight
- **Title**: Benchmarking: Outperforming the VN30 Equal-Weight
- **Bullets**:
    - Systematic factor tilting consistently beats passive "buy-and-hold."
    - Volatility reduction: Optimized portfolio exhibits smoother equity curves.
    - Information Ratio (IR) highlights the consistency of the Alpha Engine.
- **Recommended Visual**: Cumulative Equity Curve (Log Scale) comparing the Strategy, VN30 Equal-Weight, and VN-Index.

## Slide 17: Risk Metrics: Max Drawdown Analysis
- **Title**: Risk Management: Resilience in Bear Markets
- **Bullets**:
    - **Max Drawdown**: -14.96% (significantly shallower than VN30's ~25% benchmark dips).
    - Recovery Time: Faster re-entry after market corrections due to Momentum/Quality tilts.
    - Tail Risk: Lower VaR (Value-at-Risk) compared to concentrated portfolios.
- **Recommended Visual**: An "Underwater Chart" (Drawdown Plot) showing the strategy's depth vs. the market benchmark.

## Slide 18: Conclusion & Future Roadmap
- **Title**: Conclusion: Scalability & The Path Forward
- **Bullets**:
    - **Proven**: Multi-factor logic holds up under rigorous out-of-sample Walk-Forward testing.
    - **Scalability**: Framework easily adapts to the VN100 or Sector-Specific universes.
    - **Future Work**: Integration of Alternative Data (Sentiment/Order Flow) and Machine Learning (Random Forests) for factor blending.
- **Recommended Visual**: A roadmap arrow showing milestones: VN30 (Complete) -> VN100 (Next) -> ML-Driven Alpha (Future).
