# Portfolio Analyzer with Ray Dalio Investment Principles

A comprehensive Python tool to analyze investment portfolios with a focus on risk-adjusted returns and diversification, incorporating Ray Dalio's investment philosophy.

## Features

### 📊 Portfolio Analysis

- **Total Portfolio Value**: Calculates total portfolio value in CAD
- **Asset Allocation**: Analyzes distribution across stocks, ETFs, bonds, and cash
- **Currency Exposure**: Tracks USD vs CAD exposure
- **Top Holdings**: Identifies concentration in largest positions

### 🎯 Ray Dalio Investment Principles

1. **Diversification Across Uncorrelated Assets**: Measures correlation between holdings
2. **Risk Parity**: Evaluates equal risk contribution across positions
3. **All-Weather Portfolio**: Checks for presence of different asset classes
4. **Geographic Diversification**: Analyzes currency and regional exposure

### 📈 Advanced Risk Metrics

- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure (return per unit of total risk)
- **Sortino Ratio**: Downside risk-adjusted return measure
- **Calmar Ratio**: Return per unit of maximum drawdown
- **Information Ratio**: Excess return vs S&P 500 benchmark
- **Treynor Ratio**: Return per unit of systematic risk (beta)
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Value at Risk (VaR)**: 95% confidence interval for potential losses
- **Conditional VaR (CVaR)**: Expected shortfall beyond VaR
- **Beta**: Market sensitivity measure
- **Herfindahl-Hirschman Index (HHI)**: Portfolio concentration measure

### 🔥 Stress Testing & Scenario Analysis

- **Historical Stress Scenarios**: 2008 Crisis, COVID Crash, Stagflation, Dot-com Burst, Black Monday, Inflation Shock
- **Multi-Factor Stress Modeling**: Market crashes, volatility spikes, correlation increases, liquidity dry-ups, currency volatility
- **Portfolio Resilience Scoring**: 0-10 scale across cash buffer, diversification, volatility management, correlation management
- **Monte Carlo Simulation**: 10,000 random stress scenarios with probability analysis
- **Recovery Analysis**: Historical recovery times and required CAGR for portfolio recovery
- **Stress-Specific Recommendations**: Targeted improvements for stress resilience

### 💡 Smart Recommendations

- Concentration risk warnings
- Currency diversification suggestions
- Cash allocation optimization
- Risk-adjusted return improvements
- Position-specific recommendations
- Stress resilience enhancements

### 📊 Visualizations

- Asset allocation pie chart
- Currency exposure bar chart
- Top holdings breakdown
- Risk-return scatter plot
- Maximum drawdown analysis
- Portfolio metrics summary

## Installation

1. **Install Python dependencies**:

```bash
pip install -r requirements.txt
```

2. **Ensure your CSV file is in the correct format**:
   - File should be named `stocks.csv`
   - Required columns: `Asset`, `Ticker`, `Group`, `Price`, `Holding`, `Value`, `US/CAD`, `Stock/ETF`, `Value in Cad`

## Template

📊 **Portfolio Template**: Use this [Google Sheets template](https://docs.google.com/spreadsheets/d/1xygwEZ20z1qCHMe32qti941VmXrzjs5SHZcZwi9TFSg/edit?gid=0#gid=0) to format your portfolio data correctly. Make a copy and fill in your holdings, then export as CSV.

## Usage

### Command-Line Options

The analyzer supports multiple analysis types with command-line flags:

```bash
# Complete analysis (default)
python portfolio_analyzer.py

# Basic portfolio analysis (no risk metrics)
python portfolio_analyzer.py --analysis basic

# Comprehensive risk analysis only
python portfolio_analyzer.py --analysis risk

# Ray Dalio principles analysis only
python portfolio_analyzer.py --analysis dalio

# Visualizations only
python portfolio_analyzer.py --analysis visual

# Downside protection strategies
python portfolio_analyzer.py --analysis protection

# Stress testing and scenario analysis
python portfolio_analyzer.py --analysis stress

# Custom CSV file
python portfolio_analyzer.py --csv my_portfolio.csv

# Custom historical data period
python portfolio_analyzer.py --period 1y
```

### Analysis Types

1. **`basic`**: Portfolio overview, asset allocation, and diversification analysis
2. **`risk`**: Comprehensive risk metrics for each position with detailed analysis
3. **`dalio`**: Ray Dalio principles assessment and recommendations
4. **`visual`**: Generate portfolio visualizations and charts
5. **`complete`**: Full analysis including all above components (default)
6. **`protection`**: Downside protection strategies while keeping upside open
7. **`stress`**: Comprehensive stress testing under severe financial conditions

### Stress Testing Analysis

The stress testing analysis simulates how your portfolio would perform under severe financial conditions:

```bash
python portfolio_analyzer.py --analysis stress
```

This analysis includes:

1. **Historical Stress Scenarios**:

   - **2008 Financial Crisis**: -50% market crash, 18 months duration
   - **2020 COVID Crash**: -35% market crash, 3 months duration
   - **1970s Stagflation**: -25% market crash, 10 years duration
   - **Dot-com Bubble Burst**: -45% market crash, 2 years duration
   - **Black Monday 1987**: -22% market crash, 1 day duration
   - **2022 Inflation Shock**: -20% market crash, 12 months duration

2. **Enhanced Complex Instrument Modeling**:

   - **Inverse ETFs** (SH, SDS, PSQ, etc.): Modeled to gain during market crashes
   - **Leveraged ETFs** (TQQQ, SPXL, etc.): Amplified losses with higher volatility impact
   - **Volatility ETFs** (VXX, UVXY, etc.): Direct volatility exposure with contango/backwardation
   - **Options ETFs** (XYLD, QYLD, etc.): Strategy breakdown risk during stress
   - **Currency ETFs** (UUP, FXE, etc.): Flight to safety and correlation breakdown
   - **Commodity ETFs** (GLD, SLV, etc.): Inflation hedge with storage cost considerations

3. **Multi-Factor Stress Modeling**:

   - Market crash impacts (percentage declines with instrument-specific multipliers)
   - Volatility spikes (2-5x normal volatility with instrument-specific sensitivity)
   - Correlation increases (diversification becomes less effective)
   - Liquidity dry-ups (harder to sell positions, especially complex instruments)
   - Currency volatility (exchange rate impacts)

4. **Enhanced Portfolio Resilience Scoring (0-10 scale)**:

   - **Cash Buffer Score**: Emergency cash availability
   - **Diversification Score**: Portfolio diversification effectiveness
   - **Volatility Management Score**: Risk management quality
   - **Correlation Management Score**: Asset uncorrelation level
   - **Complex Instrument Management Score**: Risk management for derivatives

5. **Advanced Monte Carlo Simulation**:

   - 10,000 random stress scenarios with instrument-specific modeling
   - Probability analysis for different loss levels
   - Value at Risk (VaR) at 95% and 99% confidence levels
   - Expected loss distributions with complex instrument considerations

6. **Complex Instrument-Specific Recommendations**:

   - **Leveraged ETFs**: Reduce exposure to <5% for high-risk scenarios
   - **Inverse ETFs**: Monitor sizing and manage decay with monthly rebalancing
   - **Volatility ETFs**: Use only for short-term volatility plays
   - **Options ETFs**: Monitor strategy performance and implied volatility
   - **Currency ETFs**: Monitor interest rate differentials and correlations
   - **Commodity ETFs**: Monitor supply/demand fundamentals and storage costs

7. **Recovery Analysis**:

   - Historical recovery times for each scenario
   - Required compound annual growth rate for recovery
   - Recovery feasibility assessment with complex instrument considerations

8. **Enhanced Stress-Specific Recommendations**:
   - Complex instrument exposure management
   - Cash buffer optimization
   - Defensive asset allocation
   - Concentration risk reduction
   - Currency diversification
   - Hedging strategies for complex instruments

### Testing Enhanced Stress Testing

To test the enhanced stress testing with complex instruments:

```bash
python test_complex_instruments.py
```

This will run a demonstration with a portfolio containing:

- Standard ETFs (VOO)
- Inverse ETFs (SH)
- Leveraged ETFs (TQQQ)
- Volatility ETFs (VXX)
- Options ETFs (XYLD)
- Commodity ETFs (GLD)
- Currency ETFs (UUP)
- Cash positions

### Complex Instrument Risk Management

The enhanced stress testing provides specific guidance for managing complex instruments:

**Leveraged ETFs (TQQQ, SPXL, etc.):**

- ⚠️ HIGH RISK: Amplified losses in stress scenarios
- Use for short-term momentum trades only
- Set tight stop-losses (-10% to -15%)
- Hedge with inverse ETFs or puts

**Inverse ETFs (SH, SDS, etc.):**

- 🛡️ PROTECTION: Can provide downside protection
- Monitor for correlation breakdown
- Rebalance monthly to manage decay
- Best during high volatility periods

**Volatility ETFs (VXX, UVXY, etc.):**

- 📊 VOLATILITY: Direct exposure to market volatility
- Use only for volatility hedging
- Monitor VIX term structure and contango
- Short-term holding only (days to weeks)

**Options ETFs (XYLD, QYLD, etc.):**

- 🎲 OPTIONS: Income generation with strategy risk
- Monitor strategy performance and volatility
- Be aware of gamma risk and strategy breakdown
- Consider as income supplement

**Currency ETFs (UUP, FXE, etc.):**

- 🌍 CURRENCY: Currency diversification and speculation
- Monitor interest rate differentials and policies
- Be aware of currency volatility and correlations
- Consider as portfolio hedge

**Commodity ETFs (GLD, SLV, etc.):**

- 🏭 COMMODITY: Inflation hedge and diversification
- Monitor supply/demand fundamentals
- Be aware of storage costs and contango
- Consider as inflation hedge

### Protection Analysis

The protection analysis provides comprehensive strategies to protect your portfolio's downside while maintaining upside potential:

```bash
python portfolio_analyzer.py --analysis protection
```

This analysis includes:

1. **Current Risk Assessment**: Identifies concentration risks and high-volatility positions
2. **Protective Put Options**: Immediate downside protection strategies
3. **Portfolio Insurance**: Defensive asset allocation recommendations
4. **Dynamic Asset Allocation**: Stop-loss and trailing stop strategies
5. **Covered Call Income**: Income generation while maintaining upside
6. **Risk Parity Rebalancing**: Concentration reduction strategies
7. **Implementation Plan**: 3-phase execution strategy
8. **Cost-Benefit Analysis**: Protection costs vs. benefits

### Expected CSV Format

```csv
Asset,Ticker,Group,Price,Holding,Value,US/CAD,Stock/ETF,Value in Cad
Apple,AAPL,AAPL,201,129.00,25929,USD,Stock,35601.81319
Microsoft,MSFT,MSFT,477.4,125,59675,USD,Stock,81936.75815
...
```

## Output

The analyzer provides:

1. **Console Output**:

   - Portfolio summary statistics
   - Detailed asset allocation analysis
   - Advanced risk metrics for each position
   - Ray Dalio principles assessment
   - Personalized recommendations
   - Risk metrics comparison summary
   - Stress testing results and resilience scoring
   - Monte Carlo simulation statistics

2. **Visual Dashboard**:
   - Interactive matplotlib visualizations
   - Saved as `portfolio_analysis.png`

## Advanced Risk Metrics Explained

### Sharpe Ratio

- **Formula**: (Return - Risk Free Rate) / Volatility
- **Interpretation**: Higher is better, >1.0 is excellent
- **Purpose**: Measures risk-adjusted returns considering total risk

### Sortino Ratio

- **Formula**: (Return - Risk Free Rate) / Downside Deviation
- **Interpretation**: Higher is better, focuses on downside risk only
- **Purpose**: Better measure for risk-averse investors

### Calmar Ratio

- **Formula**: Annual Return / Maximum Drawdown
- **Interpretation**: Higher is better, shows return per unit of worst decline
- **Purpose**: Measures recovery from worst losses

### Information Ratio

- **Formula**: (Portfolio Return - Benchmark Return) / Tracking Error
- **Interpretation**: Higher is better, >0.5 is good
- **Purpose**: Measures active management skill vs S&P 500

### Treynor Ratio

- **Formula**: (Return - Risk Free Rate) / Beta
- **Interpretation**: Higher is better, measures systematic risk
- **Purpose**: Evaluates return per unit of market risk

### Beta

- **Formula**: Covariance(Portfolio, Market) / Variance(Market)
- **Interpretation**:
  - β = 1: Moves with market
  - β > 1: More volatile than market
  - β < 1: Less volatile than market
- **Purpose**: Measures market sensitivity

## Stress Testing Metrics Explained

### Portfolio Resilience Score

- **Components**: Cash buffer, diversification, volatility management, correlation management
- **Scale**: 0-10 (10 = excellent resilience)
- **Interpretation**:
  - 8-10: Excellent resilience to stress scenarios
  - 6-7: Good resilience with room for improvement
  - 4-5: Moderate resilience, may struggle in severe stress
  - 0-3: Poor resilience, vulnerable to stress scenarios

### Monte Carlo VaR

- **95% VaR**: 95% confidence that losses won't exceed this level
- **99% VaR**: 99% confidence that losses won't exceed this level
- **Purpose**: Quantifies potential losses under stress conditions

### Recovery Analysis

- **Historical Recovery Times**: Based on actual market recovery periods
- **Required CAGR**: Compound annual growth rate needed to recover losses
- **Feasibility Assessment**: Whether recovery targets are realistic

## Ray Dalio's Investment Philosophy

This tool incorporates key principles from Ray Dalio's investment strategy:

### 1. Diversification

- **Uncorrelated Assets**: Seeks assets that don't move together
- **Multiple Asset Classes**: Stocks, bonds, commodities, real estate
- **Geographic Spread**: International diversification

### 2. Risk Parity

- **Equal Risk Contribution**: Each asset contributes equally to portfolio risk
- **Volatility Targeting**: Adjusts allocations based on risk levels
- **Dynamic Rebalancing**: Maintains target risk allocations

### 3. All-Weather Portfolio

- **Economic Environment Adaptation**: Performs in different market conditions
- **Inflation Protection**: Assets that benefit from inflation
- **Deflation Protection**: Assets that benefit from deflation

### 4. Systematic Approach

- **Rules-Based**: Removes emotional decision making
- **Data-Driven**: Uses historical analysis and correlations
- **Continuous Monitoring**: Regular portfolio health checks

## Key Metrics Explained

### Maximum Drawdown

- **Definition**: Largest peak-to-trough decline
- **Interpretation**: Lower is better, shows worst-case scenario
- **Purpose**: Risk tolerance assessment

### Herfindahl-Hirschman Index (HHI)

- **Formula**: Sum of squared portfolio weights
- **Interpretation**:
  - <0.15: Well diversified
  - 0.15-0.25: Moderately concentrated
  - > 0.25: Highly concentrated

### Diversification Score

- **Formula**: 1 - Average Correlation
- **Interpretation**: Higher is better, shows uncorrelated assets
- **Purpose**: Measures portfolio diversification effectiveness

## Example Output

```
Portfolio Analysis Summary:
Total Portfolio Value: $1,234,567.89 CAD
Number of positions: 25
Stock/ETF positions: 18
Cash positions: 7
--------------------------------------------------

📊 ASSET ALLOCATION:
   Stock: 45.2%
   ETF: 38.7%
   Cash: 12.1%
   Bond: 4.0%

💱 CURRENCY EXPOSURE:
   USD: 65.3%
   CAD: 34.7%

🎯 RAY DALIO PRINCIPLES ANALYSIS:
   Diversification Score: 0.72
   Risk Parity Score: 0.68
   Asset Class Diversity: 4/4
   Geographic Diversification: 0.35

🔥 STRESS TESTING ANALYSIS:
   Portfolio Resilience Score: 7.2/10
   Worst Scenario: 2008 Financial Crisis (-42.3% loss)
   Monte Carlo 95% VaR: -18.7%
   Probability of >20% loss: 12.3%

📈 RISK METRICS COMPARISON SUMMARY:
   Sharpe Ratio:
     Best: NVIDIA (NVDA) = 1.85
     Worst: Tesla (TSLA) = 0.32
     Average: 1.06

💡 RECOMMENDATIONS:
   ⚠️  HIGH USD EXPOSURE: Consider increasing CAD-denominated investments
   💡  LOW CASH: Consider maintaining 5-10% cash for opportunities
   🛡️  Add defensive assets (bonds, gold) for stress protection
```

## Troubleshooting

### Common Issues

1. **Missing Data**: Ensure all required columns are present in CSV
2. **Network Issues**: Historical data requires internet connection
3. **Ticker Symbols**: Use correct ticker symbols (e.g., AAPL, not APPLE)
4. **Currency Conversion**: Ensure "Value in Cad" column is properly calculated

### Error Messages

- **"Could not fetch data for [TICKER]"**: Check ticker symbol validity
- **"No data available"**: Verify CSV file format and data quality
- **"Visualization error"**: Check matplotlib installation and display settings

## Contributing

Feel free to enhance the analyzer with:

- Additional risk metrics
- More sophisticated correlation analysis
- Alternative visualization options
- Integration with other data sources
- Additional stress testing scenarios
- Machine learning-based risk prediction

## License

This project is open source and available under the MIT License.
