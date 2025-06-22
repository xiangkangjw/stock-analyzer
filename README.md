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

### 📈 Risk Metrics

- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Value at Risk (VaR)**: 95% confidence interval for potential losses
- **Herfindahl-Hirschman Index (HHI)**: Portfolio concentration measure

### 💡 Smart Recommendations

- Concentration risk warnings
- Currency diversification suggestions
- Cash allocation optimization
- Risk-adjusted return improvements

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

## Usage

### Basic Usage

```bash
python portfolio_analyzer.py
```

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
   - Risk metrics for each position
   - Ray Dalio principles assessment
   - Personalized recommendations

2. **Visual Dashboard**:
   - Interactive matplotlib visualizations
   - Saved as `portfolio_analysis.png`

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

### Sharpe Ratio

- **Formula**: (Return - Risk Free Rate) / Volatility
- **Interpretation**: Higher is better, >1.0 is excellent
- **Purpose**: Measures risk-adjusted returns

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

💡 RECOMMENDATIONS:
   ⚠️  HIGH USD EXPOSURE: Consider increasing CAD-denominated investments
   💡  LOW CASH: Consider maintaining 5-10% cash for opportunities
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

## License

This project is open source and available under the MIT License.
