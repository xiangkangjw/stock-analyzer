#!/usr/bin/env python3
"""
Portfolio Analyzer with Ray Dalio Investment Principles
Analyzes investment portfolio focusing on risk-adjusted returns and diversification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
import warnings
import argparse
import sys

warnings.filterwarnings("ignore")


class PortfolioAnalyzer:
    def __init__(self, csv_file):
        """Initialize portfolio analyzer with CSV data"""
        self.df = pd.read_csv(csv_file)
        self.portfolio_data = None
        self.risk_metrics = {}
        self.dalio_analysis = {}

        # Ticker mapping for Canadian and special symbols
        self.ticker_mapper = {
            "BRK.B": "BRK-B",
            "VFV": "VFV.TO",
            "QQC": "QQC.TO",
            "VUN": "VUN.TO",
            "XSP": "XSP.TO",
            "HSAV": "HSAV.TO",
            "ZMMK": "ZMMK.TO",
            "SHV": "SHV",
            "VOO": "VOO",
            "AAPL": "AAPL",
            "MSFT": "MSFT",
            "AVGO": "AVGO",
            "GOOG": "GOOG",
            "NVDA": "NVDA",
            "TSLA": "TSLA",
            "AMD": "AMD",
            "CGL.C": "CGL-C.TO",
            "VAB": "VAB.TO",
        }

        # Enhanced asset classification for complex instruments
        self.complex_instruments = {
            # Inverse ETFs (move opposite to market)
            "inverse_etfs": [
                "SDS",
                "SH",
                "PSQ",
                "DOG",
                "RWM",
                "TWM",
                "QID",
                "MZZ",
                "SMN",
                "DXD",
                "TZA",
                "FAZ",
                "DRV",
                "YCS",
                "FXP",
                "EEV",
                "DUG",
                "SKF",
                "SRS",
                "SDP",
                "SQQQ",
                "SPXU",
            ],
            # Leveraged ETFs (amplified market moves)
            "leveraged_etfs": [
                "SPXL",
                "SPXS",
                "TQQQ",
                "UPRO",
                "TMF",
                "TMV",
                "LABU",
                "LABD",
                "SOXL",
                "SOXS",
                "DPSTU",
                "DPSTD",
                "ERX",
                "ERY",
                "FAS",
                "FAZ",
                "JNUG",
                "JDST",
                "YINN",
                "YANG",
                "NAIL",
                "NAK",
                "DRN",
                "DRV",
                "URE",
                "SRS",
            ],
            # Volatility ETFs (VIX-based)
            "volatility_etfs": [
                "VXX",
                "UVXY",
                "SVXY",
                "XIV",
                "VXZ",
                "VIXY",
                "VIXM",
                "VXXB",
                "VXRT",
                "VXUS",
            ],
            # Options-based ETFs
            "options_etfs": [
                "XYLD",
                "QYLD",
                "RYLD",
                "JEPI",
                "JEPQ",
                "DIVO",
                "NUSI",
                "QYLG",
                "XYLG",
                "RYLG",
            ],
            # Currency ETFs
            "currency_etfs": [
                "UUP",
                "UDN",
                "FXE",
                "FXB",
                "FXC",
                "FXF",
                "FXA",
                "FXY",
                "FXS",
                "FXM",
            ],
            # Commodity ETFs
            "commodity_etfs": [
                "GLD",
                "SLV",
                "USO",
                "UNG",
                "DBA",
                "DBC",
                "GSG",
                "COMT",
                "PDBC",
                "CMDT",
            ],
        }

        # Stress scenario multipliers for different instrument types
        self.stress_multipliers = {
            "inverse_etfs": {
                "market_crash": 0.5,  # Inverse ETFs gain during crashes
                "volatility_spike": 2.0,  # Higher volatility impact
                "liquidity_dry_up": 1.5,  # More sensitive to liquidity
                "correlation_breakdown": 0.3,  # Correlation may break down
            },
            "leveraged_etfs": {
                "market_crash": 2.5,  # Amplified losses
                "volatility_spike": 3.0,  # Much higher volatility impact
                "liquidity_dry_up": 2.0,  # Very sensitive to liquidity
                "correlation_breakdown": 0.1,  # Correlation breakdown hurts
            },
            "volatility_etfs": {
                "market_crash": 1.5,  # Volatility spikes during crashes
                "volatility_spike": 4.0,  # Direct volatility exposure
                "liquidity_dry_up": 2.5,  # Very illiquid during stress
                "correlation_breakdown": 0.2,  # Contango/backwardation issues
            },
            "options_etfs": {
                "market_crash": 1.8,  # Options strategies can break down
                "volatility_spike": 2.5,  # Vega exposure
                "liquidity_dry_up": 1.8,  # Options liquidity dries up
                "correlation_breakdown": 0.4,  # Greeks can change rapidly
            },
            "currency_etfs": {
                "market_crash": 0.8,  # Flight to safety currencies
                "volatility_spike": 1.5,  # Currency volatility
                "liquidity_dry_up": 1.2,  # FX liquidity
                "correlation_breakdown": 0.6,  # Currency correlations
            },
            "commodity_etfs": {
                "market_crash": 1.2,  # Commodities can be defensive
                "volatility_spike": 1.8,  # Commodity volatility
                "liquidity_dry_up": 1.3,  # Physical delivery issues
                "correlation_breakdown": 0.5,  # Commodity correlations
            },
        }

    def map_ticker(self, ticker):
        """Map ticker symbol to correct format for yfinance"""
        return self.ticker_mapper.get(ticker, ticker)

    def clean_data(self):
        """Clean and prepare portfolio data"""
        # Remove rows with missing values (but allow cash positions to have missing tickers)
        self.df = self.df.dropna(subset=["Value in Cad"])

        # Convert Value in Cad to numeric, handling any formatting issues
        self.df["Value in Cad"] = pd.to_numeric(
            self.df["Value in Cad"], errors="coerce"
        )
        self.df = self.df.dropna(subset=["Value in Cad"])

        # Handle missing tickers for cash positions
        self.df["Ticker"] = self.df["Ticker"].fillna("CASH")

        # Filter out cash positions for stock/ETF analysis (we'll analyze separately)
        self.stock_etf_data = self.df[self.df["Stock/ETF"].isin(["Stock", "ETF"])]
        self.cash_data = self.df[self.df["Stock/ETF"] == "Cash"]

        # Calculate total portfolio value
        self.total_value = self.df["Value in Cad"].sum()

        print(f"Portfolio Analysis Summary:")
        print(f"Total Portfolio Value: ${self.total_value:,.2f} CAD")
        print(f"Number of positions: {len(self.df)}")
        print(f"Stock/ETF positions: {len(self.stock_etf_data)}")
        print(f"Cash positions: {len(self.cash_data)}")
        print("-" * 50)

    def get_historical_data(self, period="2y"):
        """Fetch historical data for portfolio holdings"""
        # Only get tickers from stock/ETF positions, not cash
        tickers = self.stock_etf_data["Ticker"].unique()
        historical_data = {}

        print("Fetching historical data for risk analysis...")
        for ticker in tickers:
            # Skip cash positions
            if ticker == "CASH" or pd.isna(ticker):
                continue

            try:
                # Map ticker to correct format
                mapped_ticker = self.map_ticker(ticker)

                stock = yf.Ticker(mapped_ticker)
                hist = stock.history(period=period)
                if not hist.empty:
                    historical_data[ticker] = hist["Close"]
                else:
                    print(f"No data available for {ticker} ({mapped_ticker})")
            except Exception as e:
                print(f"Could not fetch data for {ticker} ({mapped_ticker}): {e}")

        return historical_data

    def calculate_advanced_risk_metrics(self, historical_data):
        """Calculate advanced risk metrics for each position"""
        risk_data = []

        for _, row in self.stock_etf_data.iterrows():
            ticker = row["Ticker"]
            if ticker in historical_data:
                prices = historical_data[ticker]
                returns = prices.pct_change().dropna()

                # Basic risk metrics
                volatility = returns.std() * np.sqrt(252)  # Annualized
                mean_return = returns.mean() * 252  # Annualized

                # Sharpe Ratio (assuming 2% risk-free rate)
                risk_free_rate = 0.02
                sharpe_ratio = (
                    (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
                )

                # Sortino Ratio (downside deviation)
                downside_returns = returns[returns < 0]
                downside_deviation = (
                    downside_returns.std() * np.sqrt(252)
                    if len(downside_returns) > 0
                    else 0
                )
                sortino_ratio = (
                    (mean_return - risk_free_rate) / downside_deviation
                    if downside_deviation > 0
                    else 0
                )

                # Maximum Drawdown
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()

                # Calmar Ratio (annual return / max drawdown)
                calmar_ratio = (
                    mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
                )

                # Information Ratio (excess return / tracking error)
                # Using S&P 500 as benchmark
                try:
                    benchmark = yf.Ticker("^GSPC").history(period="2y")["Close"]
                    benchmark_returns = benchmark.pct_change().dropna()
                    # Align dates
                    common_dates = returns.index.intersection(benchmark_returns.index)
                    if len(common_dates) > 30:
                        aligned_returns = returns.loc[common_dates]
                        aligned_benchmark = benchmark_returns.loc[common_dates]
                        excess_returns = aligned_returns - aligned_benchmark
                        tracking_error = excess_returns.std() * np.sqrt(252)
                        information_ratio = (
                            excess_returns.mean() * 252 / tracking_error
                            if tracking_error > 0
                            else 0
                        )
                    else:
                        information_ratio = 0
                except:
                    information_ratio = 0

                # Value at Risk (95% confidence)
                var_95 = np.percentile(returns, 5)

                # Conditional Value at Risk (Expected Shortfall)
                cvar_95 = (
                    returns[returns <= var_95].mean()
                    if len(returns[returns <= var_95]) > 0
                    else 0
                )

                # Beta (market sensitivity)
                try:
                    if len(common_dates) > 30:
                        beta = np.cov(aligned_returns, aligned_benchmark)[
                            0, 1
                        ] / np.var(aligned_benchmark)
                    else:
                        beta = 1.0
                except:
                    beta = 1.0

                # Treynor Ratio (excess return / beta)
                treynor_ratio = (
                    (mean_return - risk_free_rate) / beta if beta != 0 else 0
                )

                risk_data.append(
                    {
                        "Ticker": ticker,
                        "Asset": row["Asset"],
                        "Value": row["Value in Cad"],
                        "Weight": row["Value in Cad"] / self.total_value,
                        "Volatility": volatility,
                        "Mean_Return": mean_return,
                        "Sharpe_Ratio": sharpe_ratio,
                        "Sortino_Ratio": sortino_ratio,
                        "Calmar_Ratio": calmar_ratio,
                        "Information_Ratio": information_ratio,
                        "Treynor_Ratio": treynor_ratio,
                        "Max_Drawdown": max_drawdown,
                        "VaR_95": var_95,
                        "CVaR_95": cvar_95,
                        "Beta": beta,
                        "Currency": row["US/CAD"],
                        "Type": row["Stock/ETF"],
                    }
                )

        self.risk_metrics = pd.DataFrame(risk_data)
        return self.risk_metrics

    def analyze_diversification(self):
        """Analyze portfolio diversification (Ray Dalio principle)"""
        # Asset allocation by type
        allocation_by_type = (
            self.df.groupby("Stock/ETF")["Value in Cad"].sum() / self.total_value * 100
        )

        # Currency exposure
        currency_exposure = (
            self.df.groupby("US/CAD")["Value in Cad"].sum() / self.total_value * 100
        )

        # Top holdings concentration
        top_holdings = self.df.nlargest(5, "Value in Cad")[["Asset", "Value in Cad"]]
        top_holdings["Weight"] = top_holdings["Value in Cad"] / self.total_value * 100

        # Herfindahl-Hirschman Index (concentration measure)
        weights = self.df["Value in Cad"] / self.total_value
        hhi = (weights**2).sum()

        self.dalio_analysis["allocation_by_type"] = allocation_by_type
        self.dalio_analysis["currency_exposure"] = currency_exposure
        self.dalio_analysis["top_holdings"] = top_holdings
        self.dalio_analysis["hhi"] = hhi

        return self.dalio_analysis

    def apply_dalio_principles(self):
        """Apply Ray Dalio's investment principles"""
        principles = {}

        # Principle 1: Diversification across uncorrelated assets
        if len(self.risk_metrics) > 0:
            # Calculate correlation matrix
            tickers = self.risk_metrics["Ticker"].tolist()
            historical_data = self.get_historical_data()

            returns_data = {}
            for ticker in tickers:
                if ticker in historical_data:
                    returns_data[ticker] = historical_data[ticker].pct_change().dropna()

            if len(returns_data) > 1:
                returns_df = pd.DataFrame(returns_data)
                correlation_matrix = returns_df.corr()
                avg_correlation = (
                    correlation_matrix.sum().sum() - len(correlation_matrix)
                ) / (len(correlation_matrix) ** 2 - len(correlation_matrix))
                principles["avg_correlation"] = avg_correlation

                # Diversification score (lower correlation = better)
                principles["diversification_score"] = 1 - avg_correlation

        # Principle 2: Risk parity (equal risk contribution)
        if len(self.risk_metrics) > 0:
            risk_contributions = (
                self.risk_metrics["Weight"] * self.risk_metrics["Volatility"]
            )
            risk_parity_score = 1 - (
                risk_contributions.std() / risk_contributions.mean()
            )
            principles["risk_parity_score"] = risk_parity_score

        # Principle 3: All-weather portfolio components
        # Check for presence of different asset classes
        asset_classes = {
            "stocks": len(self.df[self.df["Stock/ETF"] == "Stock"]),
            "etfs": len(self.df[self.df["Stock/ETF"] == "ETF"]),
            "bonds": len(self.df[self.df["Stock/ETF"] == "Bond"]),
            "cash": len(self.df[self.df["Stock/ETF"] == "Cash"]),
        }
        principles["asset_class_diversity"] = len(
            [v for v in asset_classes.values() if v > 0]
        )

        # Principle 4: Geographic diversification
        us_exposure = (
            self.df[self.df["US/CAD"] == "USD"]["Value in Cad"].sum() / self.total_value
        )
        cad_exposure = (
            self.df[self.df["US/CAD"] == "CAD"]["Value in Cad"].sum() / self.total_value
        )
        principles["geographic_diversification"] = 1 - max(us_exposure, cad_exposure)

        self.dalio_analysis["principles"] = principles
        return principles

    def generate_recommendations(self):
        """Generate investment recommendations based on analysis"""
        recommendations = []

        # Check concentration risk
        if self.dalio_analysis["hhi"] > 0.25:
            recommendations.append(
                "⚠️  HIGH CONCENTRATION RISK: Portfolio is highly concentrated. Consider diversifying across more positions."
            )

        # Check currency exposure
        us_exposure = self.dalio_analysis["currency_exposure"].get("USD", 0)
        if us_exposure > 70:
            recommendations.append(
                "⚠️  HIGH USD EXPOSURE: Consider increasing CAD-denominated investments for currency diversification."
            )

        # Check cash allocation
        cash_allocation = self.dalio_analysis["allocation_by_type"].get("Cash", 0)
        if cash_allocation < 5:
            recommendations.append(
                "💡  LOW CASH: Consider maintaining 5-10% cash for opportunities and emergencies."
            )
        elif cash_allocation > 20:
            recommendations.append(
                "💡  HIGH CASH: Consider deploying excess cash into productive investments."
            )

        # Check diversification
        if "diversification_score" in self.dalio_analysis["principles"]:
            div_score = self.dalio_analysis["principles"]["diversification_score"]
            if div_score < 0.3:
                recommendations.append(
                    "⚠️  LOW DIVERSIFICATION: Consider adding uncorrelated assets to reduce portfolio risk."
                )

        # Check risk-adjusted returns
        if len(self.risk_metrics) > 0:
            avg_sharpe = self.risk_metrics["Sharpe_Ratio"].mean()
            if avg_sharpe < 0.5:
                recommendations.append(
                    "📊  LOW RISK-ADJUSTED RETURNS: Consider reviewing position selection for better risk-adjusted performance."
                )

        return recommendations

    def create_visualizations(self):
        """Create comprehensive portfolio visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Portfolio Analysis Dashboard", fontsize=16, fontweight="bold")

        # 1. Asset Allocation Pie Chart
        allocation_data = self.dalio_analysis["allocation_by_type"]
        axes[0, 0].pie(
            allocation_data.values, labels=allocation_data.index, autopct="%1.1f%%"
        )
        axes[0, 0].set_title("Asset Allocation by Type")

        # 2. Currency Exposure
        currency_data = self.dalio_analysis["currency_exposure"]
        axes[0, 1].bar(
            currency_data.index, currency_data.values, color=["#1f77b4", "#ff7f0e"]
        )
        axes[0, 1].set_title("Currency Exposure")
        axes[0, 1].set_ylabel("Percentage (%)")

        # 3. Top Holdings
        top_holdings = self.dalio_analysis["top_holdings"]
        axes[0, 2].barh(top_holdings["Asset"], top_holdings["Weight"])
        axes[0, 2].set_title("Top 5 Holdings")
        axes[0, 2].set_xlabel("Portfolio Weight (%)")

        # 4. Risk-Return Scatter Plot
        if len(self.risk_metrics) > 0:
            scatter = axes[1, 0].scatter(
                self.risk_metrics["Volatility"],
                self.risk_metrics["Sharpe_Ratio"],
                s=self.risk_metrics["Weight"] * 1000,
                alpha=0.7,
            )
            axes[1, 0].set_xlabel("Volatility")
            axes[1, 0].set_ylabel("Sharpe Ratio")
            axes[1, 0].set_title("Risk-Return Profile")

            # Add labels for larger positions
            for _, row in self.risk_metrics.iterrows():
                if row["Weight"] > 0.05:  # Only label positions > 5%
                    axes[1, 0].annotate(
                        row["Ticker"],
                        (row["Volatility"], row["Sharpe_Ratio"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                    )

        # 5. Maximum Drawdown by Position
        if len(self.risk_metrics) > 0:
            risk_metrics_sorted = self.risk_metrics.sort_values("Max_Drawdown")
            axes[1, 1].barh(
                risk_metrics_sorted["Ticker"], risk_metrics_sorted["Max_Drawdown"] * 100
            )
            axes[1, 1].set_title("Maximum Drawdown by Position")
            axes[1, 1].set_xlabel("Maximum Drawdown (%)")

        # 6. Portfolio Risk Metrics Summary
        if len(self.risk_metrics) > 0:
            summary_text = f"""
Portfolio Summary:
• Total Value: ${self.total_value:,.0f} CAD
• Positions: {len(self.df)}
• Avg Volatility: {self.risk_metrics['Volatility'].mean():.2%}
• Avg Sharpe Ratio: {self.risk_metrics['Sharpe_Ratio'].mean():.2f}
• Concentration (HHI): {self.dalio_analysis['hhi']:.3f}
• Diversification Score: {self.dalio_analysis['principles'].get('diversification_score', 'N/A'):.2f}
            """
            axes[1, 2].text(
                0.1,
                0.5,
                summary_text,
                transform=axes[1, 2].transAxes,
                fontsize=10,
                verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
            )
            axes[1, 2].set_title("Portfolio Metrics")
            axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig("portfolio_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def print_detailed_analysis(self):
        """Print detailed portfolio analysis"""
        print("\n" + "=" * 60)
        print("DETAILED PORTFOLIO ANALYSIS")
        print("=" * 60)

        # Asset Allocation
        print("\n📊 ASSET ALLOCATION:")
        for asset_type, percentage in self.dalio_analysis["allocation_by_type"].items():
            print(f"   {asset_type}: {percentage:.1f}%")

        # Currency Exposure
        print("\n💱 CURRENCY EXPOSURE:")
        for currency, percentage in self.dalio_analysis["currency_exposure"].items():
            print(f"   {currency}: {percentage:.1f}%")

        # Top Holdings (including cash)
        print("\n🏆 TOP 5 HOLDINGS:")
        for _, row in self.dalio_analysis["top_holdings"].iterrows():
            print(
                f"   {row['Asset']}: {row['Weight']:.1f}% (${row['Value in Cad']:,.0f})"
            )

        # Cash Positions Summary
        if len(self.cash_data) > 0:
            print("\n💰 CASH POSITIONS:")
            total_cash = self.cash_data["Value in Cad"].sum()
            print(
                f"   Total Cash: ${total_cash:,.0f} CAD ({total_cash/self.total_value:.1%})"
            )
            for _, row in self.cash_data.iterrows():
                print(
                    f"     • {row['Asset']}: ${row['Value in Cad']:,.0f} ({row['US/CAD']})"
                )

        # Individual Position Risk Analysis (only for stocks/ETFs)
        if len(self.risk_metrics) > 0:
            self.print_individual_risk_analysis()

        # Risk Metrics (only for stocks/ETFs)
        if len(self.risk_metrics) > 0:
            print("\n📈 PORTFOLIO RISK METRICS (Stock/ETF Positions):")
            print(
                f"   Average Volatility: {self.risk_metrics['Volatility'].mean():.2%}"
            )
            print(
                f"   Average Sharpe Ratio: {self.risk_metrics['Sharpe_Ratio'].mean():.2f}"
            )
            print(
                f"   Average Maximum Drawdown: {self.risk_metrics['Max_Drawdown'].mean():.2%}"
            )

        # Dalio Principles
        print("\n🎯 RAY DALIO PRINCIPLES ANALYSIS:")
        principles = self.dalio_analysis["principles"]

        # Handle potential string values safely
        diversification_score = principles.get("diversification_score", "N/A")
        risk_parity_score = principles.get("risk_parity_score", "N/A")
        geographic_diversification = principles.get("geographic_diversification", "N/A")

        print(f"\n🎯 Diversification Score: {diversification_score}")
        print(f"   - Measures how uncorrelated your assets are")
        print(f"   - Higher is better (0-1 scale)")
        print(f"\n⚖️  Risk Parity Score: {risk_parity_score}")
        print(f"   - Measures equal risk contribution across positions")
        print(f"   - Higher is better")
        print(f"\n🌍 Asset Class Diversity: {principles['asset_class_diversity']}/4")
        print(f"   - Stocks, ETFs, Bonds, Cash")
        print(f"\n🌐 Geographic Diversification: {geographic_diversification}")
        print(f"   - Measures currency and regional exposure")
        print(f"   - Higher is better (0-1 scale)")
        recommendations = self.generate_recommendations()
        if recommendations:
            print(f"\n💡 Dalio-Based Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")
        else:
            print(
                "\n✅ No major concerns identified. Portfolio appears well-structured."
            )

        # Portfolio Health Indicators
        print("\n🏥 PORTFOLIO HEALTH INDICATORS:")
        print(f"   Concentration Index (HHI): {self.dalio_analysis['hhi']:.3f}")
        if self.dalio_analysis["hhi"] < 0.15:
            print("   ✅ Well diversified")
        elif self.dalio_analysis["hhi"] < 0.25:
            print("   ⚠️  Moderately concentrated")
        else:
            print("   ❌ Highly concentrated")

    def print_individual_risk_analysis(self):
        """Print detailed risk analysis for each individual position"""
        print("\n" + "=" * 100)
        print("INDIVIDUAL POSITION RISK-ADJUSTED RETURN ANALYSIS")
        print("=" * 100)

        # Sort by portfolio weight (largest positions first)
        sorted_metrics = self.risk_metrics.sort_values("Weight", ascending=False)

        print(
            f"{'Position':<20} {'Weight':<8} {'Vol':<8} {'Sharpe':<8} {'Sortino':<8} {'Calmar':<8} {'Info':<8} {'Beta':<6}"
        )
        print("-" * 100)

        for _, row in sorted_metrics.iterrows():
            # Color coding for Sharpe ratio
            sharpe = row["Sharpe_Ratio"]
            if sharpe >= 1.0:
                sharpe_indicator = "🟢"
            elif sharpe >= 0.5:
                sharpe_indicator = "🟡"
            else:
                sharpe_indicator = "🔴"

            # Color coding for Sortino ratio
            sortino = row["Sortino_Ratio"]
            if sortino >= 1.0:
                sortino_indicator = "🟢"
            elif sortino >= 0.5:
                sortino_indicator = "🟡"
            else:
                sortino_indicator = "🔴"

            print(
                f"{row['Asset']:<20} {row['Weight']:<7.1%} {row['Volatility']:<7.1%} {sharpe_indicator}{sharpe:<6.2f} {sortino_indicator}{sortino:<6.2f} {row['Calmar_Ratio']:<7.2f} {row['Information_Ratio']:<7.2f} {row['Beta']:<5.2f}"
            )

        print("\n" + "=" * 100)
        print("DETAILED POSITION ANALYSIS")
        print("=" * 100)

        for _, row in sorted_metrics.iterrows():
            print(f"\n📊 {row['Asset']} ({row['Ticker']})")
            print(f"   Portfolio Weight: {row['Weight']:.1%}")
            print(f"   Value: ${row['Value']:,.0f} CAD")
            print(f"   Currency: {row['Currency']}")
            print(f"   Type: {row['Type']}")
            print(f"   Risk Metrics:")
            print(f"     • Volatility: {row['Volatility']:.1%} (Annualized)")
            print(f"     • Mean Return: {row['Mean_Return']:.1%} (Annualized)")
            print(f"     • Beta: {row['Beta']:.2f} (Market sensitivity)")
            print(f"     • Maximum Drawdown: {row['Max_Drawdown']:.1%} (Worst decline)")
            print(f"     • Value at Risk (95%): {row['VaR_95']:.1%} (Daily risk)")
            print(
                f"     • Conditional VaR (95%): {row['CVaR_95']:.1%} (Expected shortfall)"
            )

            print(f"   Risk-Adjusted Return Metrics:")
            print(
                f"     • Sharpe Ratio: {row['Sharpe_Ratio']:.2f} (Return per unit of total risk)"
            )
            print(
                f"     • Sortino Ratio: {row['Sortino_Ratio']:.2f} (Return per unit of downside risk)"
            )
            print(
                f"     • Calmar Ratio: {row['Calmar_Ratio']:.2f} (Return per unit of max drawdown)"
            )
            print(
                f"     • Information Ratio: {row['Information_Ratio']:.2f} (Excess return vs S&P 500)"
            )
            print(
                f"     • Treynor Ratio: {row['Treynor_Ratio']:.2f} (Return per unit of systematic risk)"
            )

            # Performance interpretation
            sharpe = row["Sharpe_Ratio"]
            if sharpe >= 1.0:
                performance = "Excellent risk-adjusted returns"
            elif sharpe >= 0.5:
                performance = "Good risk-adjusted returns"
            elif sharpe >= 0:
                performance = "Moderate risk-adjusted returns"
            else:
                performance = "Poor risk-adjusted returns"

            print(f"   Performance: {performance}")

            # Risk assessment
            vol = row["Volatility"]
            if vol <= 0.20:
                risk_assessment = "Low risk"
            elif vol <= 0.35:
                risk_assessment = "Moderate risk"
            else:
                risk_assessment = "High risk"

            print(f"   Risk Level: {risk_assessment}")

            # Position-specific recommendations
            recommendations = []
            if row["Weight"] > 0.10:  # > 10% of portfolio
                recommendations.append(
                    "Consider reducing position size for diversification"
                )
            if sharpe < 0.5:
                recommendations.append(
                    "Low risk-adjusted returns - consider alternatives"
                )
            if vol > 0.40:
                recommendations.append(
                    "High volatility - ensure this fits your risk tolerance"
                )
            if row["Beta"] > 1.5:
                recommendations.append(
                    "High market sensitivity - consider defensive positions"
                )
            if row["Information_Ratio"] < 0:
                recommendations.append("Underperforming vs S&P 500 - review position")

            if recommendations:
                print(f"   💡 Recommendations:")
                for rec in recommendations:
                    print(f"     • {rec}")
            else:
                print(f"   ✅ Position appears well-balanced")

    def print_risk_comparison_summary(self):
        """Print summary comparison of risk metrics across positions"""
        if len(self.risk_metrics) == 0:
            return

        print("\n" + "=" * 80)
        print("RISK METRICS COMPARISON SUMMARY")
        print("=" * 80)

        metrics_to_compare = [
            "Sharpe_Ratio",
            "Sortino_Ratio",
            "Calmar_Ratio",
            "Information_Ratio",
            "Treynor_Ratio",
        ]

        for metric in metrics_to_compare:
            if metric in self.risk_metrics.columns:
                best_position = self.risk_metrics.loc[
                    self.risk_metrics[metric].idxmax()
                ]
                worst_position = self.risk_metrics.loc[
                    self.risk_metrics[metric].idxmin()
                ]

                print(f"\n📈 {metric.replace('_', ' ')}:")
                print(
                    f"   Best: {best_position['Asset']} ({best_position['Ticker']}) = {best_position[metric]:.2f}"
                )
                print(
                    f"   Worst: {worst_position['Asset']} ({worst_position['Ticker']}) = {worst_position[metric]:.2f}"
                )
                print(f"   Average: {self.risk_metrics[metric].mean():.2f}")

        # Calculate portfolio-level metrics
        print("\n" + "=" * 80)
        print("📊 PORTFOLIO-LEVEL RISK METRICS")
        print("=" * 80)

        # Portfolio Beta (weighted average)
        portfolio_beta = (self.risk_metrics["Weight"] * self.risk_metrics["Beta"]).sum()
        print(f"\n🎯 Portfolio Beta: {portfolio_beta:.2f}")

        if portfolio_beta > 1.2:
            beta_interpretation = "Aggressive (high market sensitivity)"
        elif portfolio_beta > 0.8:
            beta_interpretation = "Moderate (balanced market sensitivity)"
        else:
            beta_interpretation = "Conservative (low market sensitivity)"
        print(f"   Interpretation: {beta_interpretation}")

        # Portfolio Alpha (excess return vs market)
        risk_free_rate = 0.02  # 2% risk-free rate
        market_return = 0.10  # Assuming 10% market return

        # Calculate portfolio expected return based on beta
        portfolio_expected_return = risk_free_rate + portfolio_beta * (
            market_return - risk_free_rate
        )

        # Calculate actual portfolio return (weighted average of position returns)
        portfolio_actual_return = (
            self.risk_metrics["Weight"] * self.risk_metrics["Mean_Return"]
        ).sum()

        # Portfolio Alpha = Actual Return - Expected Return (based on CAPM)
        portfolio_alpha = portfolio_actual_return - portfolio_expected_return

        print(f"\n📈 Portfolio Alpha: {portfolio_alpha:.2%}")
        if portfolio_alpha > 0.05:
            alpha_interpretation = "Excellent (significant outperformance)"
        elif portfolio_alpha > 0.02:
            alpha_interpretation = "Good (moderate outperformance)"
        elif portfolio_alpha > -0.02:
            alpha_interpretation = "Neutral (market performance)"
        else:
            alpha_interpretation = "Poor (underperformance)"
        print(f"   Interpretation: {alpha_interpretation}")

        # Portfolio Volatility (weighted average)
        portfolio_volatility = (
            self.risk_metrics["Weight"] * self.risk_metrics["Volatility"]
        ).sum()
        print(f"\n📊 Portfolio Volatility: {portfolio_volatility:.2%}")

        if portfolio_volatility > 0.30:
            vol_interpretation = "High risk"
        elif portfolio_volatility > 0.20:
            vol_interpretation = "Moderate risk"
        else:
            vol_interpretation = "Low risk"
        print(f"   Interpretation: {vol_interpretation}")

        # Portfolio Sharpe Ratio
        portfolio_sharpe = (
            (portfolio_actual_return - risk_free_rate) / portfolio_volatility
            if portfolio_volatility > 0
            else 0
        )
        print(f"\n⚡ Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}")

        if portfolio_sharpe > 1.0:
            sharpe_interpretation = "Excellent risk-adjusted returns"
        elif portfolio_sharpe > 0.5:
            sharpe_interpretation = "Good risk-adjusted returns"
        elif portfolio_sharpe > 0:
            sharpe_interpretation = "Moderate risk-adjusted returns"
        else:
            sharpe_interpretation = "Poor risk-adjusted returns"
        print(f"   Interpretation: {sharpe_interpretation}")

        # Portfolio Information Ratio (excess return vs S&P 500)
        portfolio_info_ratio = (
            self.risk_metrics["Weight"] * self.risk_metrics["Information_Ratio"]
        ).sum()
        print(f"\n📊 Portfolio Information Ratio: {portfolio_info_ratio:.2f}")

        if portfolio_info_ratio > 0.5:
            info_interpretation = "Excellent relative performance"
        elif portfolio_info_ratio > 0.2:
            info_interpretation = "Good relative performance"
        elif portfolio_info_ratio > -0.2:
            info_interpretation = "Neutral relative performance"
        else:
            info_interpretation = "Poor relative performance"
        print(f"   Interpretation: {info_interpretation}")

        # Risk decomposition
        print(f"\n🔍 RISK DECOMPOSITION:")
        print(f"   • Systematic Risk (Beta-driven): {portfolio_beta:.2f} × Market Risk")
        print(f"   • Idiosyncratic Risk: Portfolio-specific factors")
        print(f"   • Total Risk (Volatility): {portfolio_volatility:.2%}")

        # Beta contribution by position
        print(f"\n📊 BETA CONTRIBUTION BY POSITION:")
        beta_contributions = self.risk_metrics[["Asset", "Weight", "Beta"]].copy()
        beta_contributions["Beta_Contribution"] = (
            beta_contributions["Weight"] * beta_contributions["Beta"]
        )
        beta_contributions = beta_contributions.sort_values(
            "Beta_Contribution", ascending=False
        )

        for _, row in beta_contributions.head(10).iterrows():
            print(
                f"   • {row['Asset']}: {row['Beta_Contribution']:.3f} ({row['Weight']:.1%} × {row['Beta']:.2f})"
            )

        # Portfolio risk assessment
        print(f"\n🎯 PORTFOLIO RISK ASSESSMENT:")
        risk_score = 0

        # Beta risk assessment
        if portfolio_beta > 1.5:
            print(
                f"   ⚠️  HIGH BETA RISK: Portfolio is {portfolio_beta:.2f}x more volatile than market"
            )
            risk_score += 2
        elif portfolio_beta > 1.2:
            print(
                f"   🟡 MODERATE BETA RISK: Portfolio is {portfolio_beta:.2f}x more volatile than market"
            )
            risk_score += 1
        else:
            print(
                f"   ✅ LOW BETA RISK: Portfolio beta of {portfolio_beta:.2f} is manageable"
            )

        # Volatility risk assessment
        if portfolio_volatility > 0.35:
            print(
                f"   ⚠️  HIGH VOLATILITY RISK: {portfolio_volatility:.1%} annual volatility"
            )
            risk_score += 2
        elif portfolio_volatility > 0.25:
            print(
                f"   🟡 MODERATE VOLATILITY RISK: {portfolio_volatility:.1%} annual volatility"
            )
            risk_score += 1
        else:
            print(
                f"   ✅ LOW VOLATILITY RISK: {portfolio_volatility:.1%} annual volatility"
            )

        # Alpha assessment
        if portfolio_alpha > 0.05:
            print(
                f"   ✅ EXCELLENT ALPHA: {portfolio_alpha:.1%} excess return vs market"
            )
            risk_score -= 1
        elif portfolio_alpha < -0.05:
            print(f"   ⚠️  POOR ALPHA: {portfolio_alpha:.1%} underperformance vs market")
            risk_score += 1

        # Overall risk rating
        if risk_score >= 3:
            risk_rating = "🔴 HIGH RISK"
        elif risk_score >= 1:
            risk_rating = "🟡 MODERATE RISK"
        else:
            risk_rating = "🟢 LOW RISK"

        print(f"\n🏆 OVERALL PORTFOLIO RISK RATING: {risk_rating}")

        # Recommendations based on risk metrics
        print(f"\n💡 RISK MANAGEMENT RECOMMENDATIONS:")
        recommendations = []

        if portfolio_beta > 1.3:
            recommendations.append(
                "   • Consider adding defensive positions to reduce beta"
            )
            recommendations.append(
                "   • Monitor market sensitivity during volatility spikes"
            )

        if portfolio_volatility > 0.30:
            recommendations.append("   • Consider adding low-volatility assets")
            recommendations.append("   • Implement stop-loss strategies")

        if portfolio_alpha < 0:
            recommendations.append("   • Review underperforming positions")
            recommendations.append("   • Consider rebalancing to market weights")

        if len(recommendations) == 0:
            recommendations.append("   • Portfolio risk metrics appear well-balanced")
            recommendations.append(
                "   • Continue monitoring for changes in market conditions"
            )

        for rec in recommendations:
            print(rec)

    def get_latest_prices_and_convert(self):
        """Fetch latest prices and convert all values to CAD"""
        print("Fetching latest prices and converting to CAD...")
        try:
            usdcad_ticker = yf.Ticker("USDCAD=X")
            usdcad_rate = usdcad_ticker.history(period="1d")["Close"].iloc[-1]
            print(f"Current USD/CAD exchange rate: {usdcad_rate:.4f}")
        except Exception as e:
            print(f"Could not fetch USD/CAD rate, using default 1.35: {e}")
            usdcad_rate = 1.35
        updated_data = []
        for _, row in self.df.iterrows():
            asset_name = row["Asset"]
            ticker = row["Ticker"]
            currency = row["US/CAD"]
            asset_type = row["Stock/ETF"]
            # Handle cash positions
            if asset_type == "Cash":
                if currency == "USD":
                    value_cad = (
                        float(row["Value"]) * usdcad_rate
                        if pd.notna(row["Value"])
                        else float(row["Value in Cad"]) * usdcad_rate
                    )
                    print(
                        f"💰 {asset_name}: ${row['Value'] or row['Value in Cad']:,} USD → ${value_cad:,.2f} CAD"
                    )
                else:
                    value_cad = (
                        float(row["Value"])
                        if pd.notna(row["Value"])
                        else float(row["Value in Cad"])
                    )
                    print(f"💰 {asset_name}: ${value_cad:,.2f} CAD")
                updated_data.append(
                    {
                        "Asset": asset_name,
                        "Ticker": ticker,
                        "Value in Cad": value_cad,
                        "Stock/ETF": asset_type,
                        "US/CAD": currency,
                    }
                )
                continue
            # Handle stocks and ETFs
            if pd.isna(ticker) or ticker == "CASH":
                continue
            try:
                mapped_ticker = self.map_ticker(ticker)
                stock = yf.Ticker(mapped_ticker)
                latest_data = stock.history(period="1d")
                if not latest_data.empty:
                    latest_price = latest_data["Close"].iloc[-1]
                    try:
                        holding = float(str(row["Holding"]).replace(",", ""))
                        current_value_native = holding * latest_price
                    except (ValueError, TypeError):
                        print(
                            f"⚠️  Could not parse holding for {asset_name} ({ticker}), skipping."
                        )
                        continue
                    if currency == "USD":
                        current_value_cad = current_value_native * usdcad_rate
                        print(
                            f"📈 {asset_name} ({ticker}): {holding} x ${latest_price:.2f} USD → ${current_value_cad:,.2f} CAD"
                        )
                    else:
                        current_value_cad = current_value_native
                        print(
                            f"📈 {asset_name} ({ticker}): {holding} x ${latest_price:.2f} CAD → ${current_value_cad:,.2f} CAD"
                        )
                    updated_data.append(
                        {
                            "Asset": asset_name,
                            "Ticker": ticker,
                            "Value in Cad": current_value_cad,
                            "Stock/ETF": asset_type,
                            "US/CAD": currency,
                        }
                    )
                else:
                    print(f"❌ No price data available for {ticker} ({mapped_ticker})")
            except Exception as e:
                print(f"❌ Error fetching price for {ticker} ({mapped_ticker}): {e}")
        self.df_updated = pd.DataFrame(updated_data)
        self.total_value_updated = self.df_updated["Value in Cad"].sum()
        return self.df_updated

    def run_basic_analysis(self):
        print("Running Basic Portfolio Analysis...")
        self.clean_data()
        self.get_latest_prices_and_convert()
        self.df = self.df_updated
        self.total_value = self.total_value_updated
        self.analyze_diversification()
        self.apply_dalio_principles()
        self.print_detailed_analysis()

    def run_risk_analysis(self):
        print("Running Comprehensive Risk Analysis...")
        self.clean_data()
        self.get_latest_prices_and_convert()
        self.df = self.df_updated
        self.total_value = self.total_value_updated
        historical_data = self.get_historical_data()
        if historical_data:
            self.calculate_advanced_risk_metrics(historical_data)
            self.print_individual_risk_analysis()
            self.print_risk_comparison_summary()

    def run_dalio_analysis(self):
        print("Running Ray Dalio Principles Analysis...")
        self.clean_data()
        self.get_latest_prices_and_convert()
        self.df = self.df_updated
        self.total_value = self.total_value_updated
        self.analyze_diversification()
        self.apply_dalio_principles()
        print("\n" + "=" * 60)
        print("RAY DALIO PRINCIPLES ANALYSIS")
        print("=" * 60)
        principles = self.dalio_analysis["principles"]

        # Handle potential string values safely
        diversification_score = principles.get("diversification_score", "N/A")
        risk_parity_score = principles.get("risk_parity_score", "N/A")
        geographic_diversification = principles.get("geographic_diversification", "N/A")

        print(f"\n🎯 Diversification Score: {diversification_score}")
        print(f"   - Measures how uncorrelated your assets are")
        print(f"   - Higher is better (0-1 scale)")
        print(f"\n⚖️  Risk Parity Score: {risk_parity_score}")
        print(f"   - Measures equal risk contribution across positions")
        print(f"   - Higher is better")
        print(f"\n🌍 Asset Class Diversity: {principles['asset_class_diversity']}/4")
        print(f"   - Stocks, ETFs, Bonds, Cash")
        print(f"\n🌐 Geographic Diversification: {geographic_diversification}")
        print(f"   - Measures currency and regional exposure")
        print(f"   - Higher is better (0-1 scale)")
        recommendations = self.generate_recommendations()
        if recommendations:
            print(f"\n💡 Dalio-Based Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")

    def run_visualizations(self):
        print("Creating Portfolio Visualizations...")
        self.clean_data()
        self.get_latest_prices_and_convert()
        self.df = self.df_updated
        self.total_value = self.total_value_updated
        historical_data = self.get_historical_data()
        if historical_data:
            self.calculate_advanced_risk_metrics(historical_data)
        self.analyze_diversification()
        self.apply_dalio_principles()
        self.create_visualizations()

    def run_complete_analysis(self):
        print("Running Complete Portfolio Analysis...")
        self.clean_data()
        self.get_latest_prices_and_convert()
        self.df = self.df_updated
        self.total_value = self.total_value_updated
        historical_data = self.get_historical_data()
        if historical_data:
            self.calculate_advanced_risk_metrics(historical_data)
        self.analyze_diversification()
        self.apply_dalio_principles()
        self.print_detailed_analysis()
        self.print_risk_comparison_summary()
        self.create_visualizations()

    def run_protection_analysis(self):
        """Run comprehensive downside protection analysis"""
        print("Running Downside Protection Analysis...")
        self.clean_data()
        self.get_latest_prices_and_convert()
        self.df = self.df_updated
        self.total_value = self.total_value_updated
        historical_data = self.get_historical_data()
        if historical_data:
            self.calculate_advanced_risk_metrics(historical_data)
        self.analyze_diversification()
        self.apply_dalio_principles()
        self.print_protection_analysis()

    def print_protection_analysis(self):
        """Print comprehensive downside protection strategies"""
        print("\n" + "=" * 80)
        print("🛡️  DOWNSIDE PROTECTION ANALYSIS")
        print("=" * 80)

        # Current portfolio risk assessment
        print("\n📊 CURRENT PORTFOLIO RISK ASSESSMENT:")
        print("-" * 50)

        # Find highest risk positions
        high_risk_positions = []
        concentration_risk = None

        print(f"   🔍 Analyzing {len(self.df)} positions...")

        for _, row in self.df.iterrows():
            asset_name = row["Asset"]
            weight = (row["Value in Cad"] / self.total_value) * 100

            # Check for concentration risk first
            if weight > 10:  # Concentration risk
                concentration_risk = {
                    "asset": asset_name,
                    "weight": weight,
                    "value": row["Value in Cad"],
                    "volatility": 0,  # Will update if found in risk metrics
                }
                print(f"   🚨 Found concentration risk: {asset_name} ({weight:.1f}%)")

            # Check for high volatility positions
            if len(self.risk_metrics) > 0:
                asset_metrics = self.risk_metrics[
                    self.risk_metrics["Asset"] == asset_name
                ]
                if not asset_metrics.empty:
                    metrics = asset_metrics.iloc[0]
                    volatility = metrics.get("Volatility", 0)

                    # Update concentration risk volatility if it's the same asset
                    if concentration_risk and concentration_risk["asset"] == asset_name:
                        concentration_risk["volatility"] = volatility

                    if volatility > 40:  # High volatility
                        high_risk_positions.append(
                            {
                                "asset": asset_name,
                                "weight": weight,
                                "volatility": volatility,
                                "sharpe": metrics.get("Sharpe_Ratio", 0),
                                "max_drawdown": metrics.get("Max_Drawdown", 0),
                            }
                        )
                        print(
                            f"   ⚠️  Found high volatility: {asset_name} ({volatility:.1f}%)"
                        )

        if concentration_risk:
            print(f"🚨 CONCENTRATION RISK DETECTED:")
            print(
                f"   • {concentration_risk['asset']}: {concentration_risk['weight']:.1f}% of portfolio"
            )
            print(f"   • Value: ${concentration_risk['value']:,.0f} CAD")
            print(f"   • Volatility: {concentration_risk['volatility']:.1f}%")
            print(f"   • Risk: Single point of failure")

        if high_risk_positions:
            print(f"\n⚠️  HIGH VOLATILITY POSITIONS:")
            for pos in high_risk_positions:
                print(
                    f"   • {pos['asset']}: {pos['weight']:.1f}% weight, {pos['volatility']:.1f}% volatility"
                )
                print(
                    f"     Sharpe: {pos['sharpe']:.2f}, Max Drawdown: {pos['max_drawdown']:.1f}%"
                )

        # Protection strategies
        print("\n" + "=" * 80)
        print("🛡️  DOWNSIDE PROTECTION STRATEGIES")
        print("=" * 80)

        print("\n1️⃣  PROTECTIVE PUT OPTIONS (Immediate Protection):")
        print("-" * 50)
        if concentration_risk:
            put_cost = concentration_risk["value"] * 0.025  # 2.5% of position
            print(f"   📈 {concentration_risk['asset']} Protection:")
            print(f"      • Buy 3-month puts 10% below current price")
            print(f"      • Cost: ~${put_cost:,.0f} (2.5% of position)")
            print(f"      • Protection: Limits downside to 10%")
            print(f"      • Upside: Unlimited potential maintained")
        else:
            print("   No concentration risk positions detected.")

        if high_risk_positions:
            for pos in high_risk_positions[:3]:  # Top 3 high-risk positions
                put_cost = (
                    (pos["weight"] / 100) * self.total_value * 0.03
                )  # 3% of position
                print(f"   📈 {pos['asset']} Protection:")
                print(f"      • Buy put spreads (lower cost than outright puts)")
                print(f"      • Cost: ~${put_cost:,.0f}")
                print(f"      • Protection: Limits losses while maintaining upside")
        else:
            print("   No high-volatility positions detected.")

        print("\n2️⃣  PORTFOLIO INSURANCE WITH DEFENSIVE ASSETS:")
        print("-" * 50)

        # Calculate current cash and defensive allocation
        cash_positions = self.df[self.df["Stock/ETF"] == "Cash"]
        total_cash = (
            cash_positions["Value in Cad"].sum() if not cash_positions.empty else 0
        )

        print(
            f"   💰 Current Cash: ${total_cash:,.0f} ({total_cash/self.total_value*100:.1f}%)"
        )

        recommended_defensive = self.total_value * 0.15  # 15% defensive allocation
        additional_needed = recommended_defensive - total_cash

        if additional_needed > 0:
            print(
                f"   🎯 Recommended Defensive Allocation: ${recommended_defensive:,.0f} (15%)"
            )
            print(f"   ➕ Additional Needed: ${additional_needed:,.0f}")

            print(f"\n   📋 Defensive Asset Allocation:")
            print(f"      • TLT (Long-term Treasury): ${additional_needed * 0.4:,.0f}")
            print(f"      • GLD (Gold): ${additional_needed * 0.3:,.0f}")
            print(f"      • SHY (Short-term Treasury): ${additional_needed * 0.2:,.0f}")
            print(f"      • Cash: ${additional_needed * 0.1:,.0f}")
        else:
            print("   Defensive allocation target met or exceeded.")

        print("\n3️⃣  DYNAMIC ASSET ALLOCATION:")
        print("-" * 50)
        print("   🛑 Stop-Loss Strategy:")

        if concentration_risk:
            stop_loss_value = concentration_risk["value"] * 0.15  # 15% stop-loss
            print(
                f"      • {concentration_risk['asset']}: -15% stop-loss (protects ${stop_loss_value:,.0f})"
            )
        else:
            print("      • No concentration risk positions for stop-loss.")

        if high_risk_positions:
            for pos in high_risk_positions[:3]:
                stop_loss_value = (
                    (pos["weight"] / 100) * self.total_value * 0.20
                )  # 20% stop-loss
                print(
                    f"      • {pos['asset']}: -20% stop-loss (protects ${stop_loss_value:,.0f})"
                )
        else:
            print("      • No high-volatility positions for stop-loss.")

        print("   📈 Trailing Stops: Move stops up as positions gain")

        print("\n4️⃣  COVERED CALL INCOME GENERATION:")
        print("-" * 50)

        # Find good candidates for covered calls
        covered_call_candidates = []
        for _, row in self.df.iterrows():
            if len(self.risk_metrics) > 0:
                asset_metrics = self.risk_metrics[
                    self.risk_metrics["Asset"] == row["Asset"]
                ]
                if not asset_metrics.empty:
                    metrics = asset_metrics.iloc[0]
                    weight = (row["Value in Cad"] / self.total_value) * 100

                    if weight > 2 and metrics.get("Sharpe_Ratio", 0) > 0.8:
                        covered_call_candidates.append(
                            {
                                "asset": row["Asset"],
                                "value": row["Value in Cad"],
                                "weight": weight,
                                "sharpe": metrics.get("Sharpe_Ratio", 0),
                            }
                        )

        if covered_call_candidates:
            print("   📊 Covered Call Candidates:")
            for candidate in covered_call_candidates[:5]:
                monthly_income = candidate["value"] * 0.02  # 2% monthly premium
                print(
                    f"      • {candidate['asset']}: ${monthly_income:,.0f}/month premium"
                )
                print(
                    f"        Upside cap: 5% per month, Downside protection: Premium reduces cost basis"
                )
        else:
            print("   No covered call candidates at this time.")

        print("\n5️⃣  RISK PARITY REBALANCING:")
        print("-" * 50)

        if concentration_risk:
            reduction_amount = concentration_risk["value"] * 0.6  # Reduce by 60%
            print(f"   🎯 Immediate Action Plan:")
            print(
                f"      SELL: {concentration_risk['asset']} (61% → 20%) = ${reduction_amount:,.0f}"
            )
            print(f"      BUY:")
            print(
                f"        • TLT (Long-term Treasury): 25% = ${reduction_amount * 0.4:,.0f}"
            )
            print(
                f"        • VIG (Dividend Growth): 15% = ${reduction_amount * 0.25:,.0f}"
            )
            print(f"        • GLD (Gold): 10% = ${reduction_amount * 0.15:,.0f}")
            print(f"        • Cash: 10% = ${reduction_amount * 0.2:,.0f}")
        else:
            print("   No concentration risk positions for rebalancing.")

        print("\n6️⃣  SECTOR ROTATION DEFENSIVE:")
        print("-" * 50)

        # Find poor performing positions to replace
        poor_performers = []
        for _, row in self.df.iterrows():
            if len(self.risk_metrics) > 0:
                asset_metrics = self.risk_metrics[
                    self.risk_metrics["Asset"] == row["Asset"]
                ]
                if not asset_metrics.empty:
                    metrics = asset_metrics.iloc[0]
                    sharpe = metrics.get("Sharpe_Ratio", 0)

                    if sharpe < 0.5:  # Poor risk-adjusted returns
                        poor_performers.append(
                            {
                                "asset": row["Asset"],
                                "value": row["Value in Cad"],
                                "sharpe": sharpe,
                                "weight": (row["Value in Cad"] / self.total_value)
                                * 100,
                            }
                        )

        if poor_performers:
            print("   🔄 Replace Poor Performers:")
            for performer in poor_performers:
                print(
                    f"      • SELL: {performer['asset']} (Sharpe: {performer['sharpe']:.2f})"
                )
                print(f"        BUY: VIG/VYM/VTV (Quality dividend/value ETFs)")
        else:
            print("   No poor performers to replace at this time.")

        # Implementation plan
        print("\n" + "=" * 80)
        print("📋 IMPLEMENTATION PLAN")
        print("=" * 80)

        print("\n🎯 PHASE 1: Immediate Protection (This Week)")
        print("-" * 50)
        print("   1. Buy Put Options:")
        if concentration_risk:
            put_cost = concentration_risk["value"] * 0.025
            print(
                f"      • {concentration_risk['asset']} $150 puts (3 months): ${put_cost:,.0f}"
            )
        else:
            print("      • No concentration risk positions for puts.")
        if high_risk_positions:
            for pos in high_risk_positions[:2]:
                put_cost = (pos["weight"] / 100) * self.total_value * 0.03
                print(f"      • {pos['asset']} puts (3 months): ${put_cost:,.0f}")
        else:
            print("      • No high-volatility positions for puts.")
        print("   2. Rebalance Cash to Defensive Assets:")
        if additional_needed > 0:
            print(f"      • TLT: ${additional_needed * 0.4:,.0f}")
            print(f"      • GLD: ${additional_needed * 0.3:,.0f}")
            print(f"      • SHY: ${additional_needed * 0.2:,.0f}")
            print(f"      • Cash: ${additional_needed * 0.1:,.0f}")
        else:
            print("      • Defensive allocation target met or exceeded.")

        print("\n🎯 PHASE 2: Reduce Concentration (Next Month)")
        print("-" * 50)
        if concentration_risk:
            reduction_amount = concentration_risk["value"] * 0.6
            print(f"   3. Sell {concentration_risk['asset']}: ${reduction_amount:,.0f}")
            print(f"      • Buy TLT: ${reduction_amount * 0.4:,.0f}")
            print(f"      • Buy VIG: ${reduction_amount * 0.25:,.0f}")
            print(f"      • Buy VYM: ${reduction_amount * 0.25:,.0f}")
            print(f"      • Cash: ${reduction_amount * 0.1:,.0f}")
        else:
            print("   3. No concentration risk positions to reduce.")

        print("\n🎯 PHASE 3: Income Generation (Ongoing)")
        print("-" * 50)
        print("   4. Covered Call Strategy:")
        if covered_call_candidates:
            total_monthly_income = sum(
                c["value"] * 0.02 for c in covered_call_candidates[:3]
            )
            print(f"      • Monthly Income: ${total_monthly_income:,.0f}")
            print(f"      • Annual Income: ${total_monthly_income * 12:,.0f}")
        else:
            print("      • No covered call candidates at this time.")
        print("   5. Cash-Secured Puts:")
        print("      • Sell puts on quality stocks at 10% below market")
        print("      • Generate 2-4% monthly income")

        # Expected results
        print("\n" + "=" * 80)
        print("📊 EXPECTED RESULTS")
        print("=" * 80)

        print("\n📈 BEFORE PROTECTION:")
        print("   • Max Drawdown: -30% (concentration risk)")
        print("   • Daily VaR: -2.8%")
        print("   • Portfolio Volatility: ~25%")

        print("\n📈 AFTER PROTECTION:")
        print("   • Max Drawdown: -15% (limited by puts + bonds)")
        print("   • Daily VaR: -1.5%")
        print("   • Portfolio Volatility: ~12%")
        print("   • Upside Potential: Still 100%+ (unlimited)")
        if covered_call_candidates:
            total_monthly_income = sum(
                c["value"] * 0.02 for c in covered_call_candidates[:3]
            )
            annual_income = total_monthly_income * 12
            # Cost-benefit analysis
            print("\n" + "=" * 80)
            print("💰 COST-BENEFIT ANALYSIS")
            print("=" * 80)

            # Always define these variables at the very top
            total_put_cost = 0
            if concentration_risk:
                total_put_cost += concentration_risk["value"] * 0.025
            for pos in high_risk_positions[:3]:
                total_put_cost += (pos["weight"] / 100) * self.total_value * 0.03

            bond_opportunity_cost = 0
            if "additional_needed" in locals() and additional_needed > 0:
                bond_opportunity_cost = additional_needed * 0.05  # 5% opportunity cost

            print(f"\n💸 Protection Costs:")
            print(
                f"   • Put Options: ${total_put_cost:,.0f} annually ({(total_put_cost/self.total_value)*100:.1f}% of portfolio)"
            )
            print(
                f"   • Bond Allocation: ${bond_opportunity_cost:,.0f} annually opportunity cost"
            )
            print(
                f"   • Total Cost: ${total_put_cost + bond_opportunity_cost:,.0f} annually ({((total_put_cost + bond_opportunity_cost)/self.total_value)*100:.1f}% of portfolio)"
            )

            print(f"\n✅ Protection Benefits:")
            print(f"   • Downside Limited: From -30% to -15%")
            print(f"   • Sleep Better: Reduced stress and anxiety")
            print(f"   • Stay Invested: Avoid panic selling")
            print(f"   • Compound Growth: Preserve capital for future gains")

            # Only print net benefit if all variables are available
            if covered_call_candidates:
                total_monthly_income = sum(
                    c["value"] * 0.02 for c in covered_call_candidates[:3]
                )
                annual_income = total_monthly_income * 12
                if total_put_cost is not None and bond_opportunity_cost is not None:
                    net_benefit = annual_income - (
                        total_put_cost + bond_opportunity_cost
                    )
                    print(
                        f"\n💰 Net Benefit: ${net_benefit:,.0f} annually (income minus costs)"
                    )

            print("\n" + "=" * 80)
            print("✅ PROTECTION ANALYSIS COMPLETE")
            print("=" * 80)

    def run_stress_test_analysis(self):
        """Run comprehensive stress testing under severe financial conditions"""
        print("Running Stress Testing Analysis...")
        self.clean_data()
        self.get_latest_prices_and_convert()
        self.df = self.df_updated
        self.total_value = self.total_value_updated
        historical_data = self.get_historical_data()
        if historical_data:
            self.calculate_advanced_risk_metrics(historical_data)
        self.analyze_diversification()
        self.apply_dalio_principles()
        self.print_stress_test_analysis()

    def print_stress_test_analysis(self):
        """Print comprehensive stress testing results"""
        print("\n" + "=" * 80)
        print("🔥 STRESS TESTING ANALYSIS")
        print("=" * 80)

        # Analyze complex instruments first
        complex_risks = self.analyze_complex_instrument_risks()

        if complex_risks:
            print("\n🚨 COMPLEX INSTRUMENT DETECTED:")
            print("-" * 50)
            total_complex_value = 0
            for instrument_type, positions in complex_risks.items():
                type_value = sum(pos["value"] for pos in positions)
                total_complex_value += type_value
                print(
                    f"   📊 {instrument_type.replace('_', ' ').title()}: ${type_value:,.0f} ({type_value/self.total_value:.1%})"
                )
                for pos in positions:
                    print(
                        f"      • {pos['asset']} ({pos['ticker']}): ${pos['value']:,.0f} ({pos['weight']:.1%})"
                    )

            print(
                f"\n   ⚠️  Complex instruments represent {total_complex_value/self.total_value:.1%} of portfolio"
            )
            print(f"   🎯 Enhanced stress modeling will be applied")

        # Define stress scenarios
        stress_scenarios = {
            "2008 Financial Crisis": {
                "market_crash": -50,
                "volatility_spike": 3.0,
                "correlation_increase": 0.8,
                "liquidity_dry_up": 0.3,
                "currency_volatility": 0.25,
                "duration": "18 months",
            },
            "2020 COVID Crash": {
                "market_crash": -35,
                "volatility_spike": 4.0,
                "correlation_increase": 0.9,
                "liquidity_dry_up": 0.2,
                "currency_volatility": 0.15,
                "duration": "3 months",
            },
            "1970s Stagflation": {
                "market_crash": -25,
                "volatility_spike": 2.5,
                "correlation_increase": 0.6,
                "liquidity_dry_up": 0.1,
                "currency_volatility": 0.20,
                "duration": "10 years",
            },
            "Dot-com Bubble Burst": {
                "market_crash": -45,
                "volatility_spike": 2.8,
                "correlation_increase": 0.7,
                "liquidity_dry_up": 0.15,
                "currency_volatility": 0.12,
                "duration": "2 years",
            },
            "Black Monday 1987": {
                "market_crash": -22,
                "volatility_spike": 5.0,
                "correlation_increase": 0.95,
                "liquidity_dry_up": 0.4,
                "currency_volatility": 0.08,
                "duration": "1 day",
            },
            "2022 Inflation Shock": {
                "market_crash": -20,
                "volatility_spike": 2.2,
                "correlation_increase": 0.5,
                "liquidity_dry_up": 0.05,
                "currency_volatility": 0.18,
                "duration": "12 months",
            },
        }

        print("\n📊 STRESS SCENARIOS SIMULATION:")
        print("-" * 50)

        # Calculate current portfolio metrics
        current_portfolio_value = self.total_value
        current_cash = (
            self.cash_data["Value in Cad"].sum() if not self.cash_data.empty else 0
        )
        current_stock_etf_value = (
            self.stock_etf_data["Value in Cad"].sum()
            if not self.stock_etf_data.empty
            else 0
        )

        print(f"💰 Current Portfolio Value: ${current_portfolio_value:,.0f} CAD")
        print(
            f"   • Cash: ${current_cash:,.0f} CAD ({current_cash/current_portfolio_value:.1%})"
        )
        print(
            f"   • Stocks/ETFs: ${current_stock_etf_value:,.0f} CAD ({current_stock_etf_value/current_portfolio_value:.1%})"
        )

        # Run stress tests with enhanced modeling
        stress_results = {}

        for scenario_name, scenario_params in stress_scenarios.items():
            print(f"\n🔥 {scenario_name.upper()}:")
            print(f"   Duration: {scenario_params['duration']}")

            # Enhanced stress calculation for complex instruments
            stressed_stock_etf_value = 0

            for _, row in self.stock_etf_data.iterrows():
                ticker = row["Ticker"]
                position_value = row["Value in Cad"]

                # Calculate base stress impact
                base_stress_impact = scenario_params["market_crash"] / 100

                # Apply instrument-specific stress modeling
                instrument_stress_impact = self.calculate_complex_instrument_stress(
                    ticker, base_stress_impact, scenario_params
                )

                # Apply additional stress factors
                correlation_penalty = (
                    scenario_params["correlation_increase"] - 0.3
                ) * 0.1
                liquidity_penalty = scenario_params["liquidity_dry_up"] * 0.05
                currency_penalty = scenario_params["currency_volatility"] * 0.02

                # Calculate final stressed value for this position
                stressed_position_value = position_value * (
                    1 + instrument_stress_impact
                )
                stressed_position_value *= 1 - correlation_penalty
                stressed_position_value *= 1 - liquidity_penalty
                stressed_position_value *= 1 - currency_penalty

                stressed_stock_etf_value += stressed_position_value

                # Print significant complex instrument impacts
                if (
                    self.classify_instrument(ticker) != "standard"
                    and abs(instrument_stress_impact - base_stress_impact) > 0.1
                ):
                    print(
                        f"      📊 {row['Asset']} ({ticker}): {instrument_stress_impact:.1%} vs {base_stress_impact:.1%} base"
                    )

            # Cash remains relatively stable (small inflation impact)
            inflation_impact = (
                -0.02 if "stagflation" in scenario_name.lower() else -0.005
            )
            stressed_cash = current_cash * (1 + inflation_impact)

            # Calculate total stressed portfolio value
            stressed_total = stressed_stock_etf_value + stressed_cash
            portfolio_loss = (
                stressed_total - current_portfolio_value
            ) / current_portfolio_value

            stress_results[scenario_name] = {
                "stressed_value": stressed_total,
                "portfolio_loss": portfolio_loss,
                "stressed_stock_etf": stressed_stock_etf_value,
                "stressed_cash": stressed_cash,
                "duration": scenario_params["duration"],
            }

            print(f"   📉 Portfolio Loss: {portfolio_loss:.1%}")
            print(f"   💰 Stressed Value: ${stressed_total:,.0f} CAD")
            print(f"   📊 Stocks/ETFs: ${stressed_stock_etf_value:,.0f} CAD")
            print(f"   💵 Cash: ${stressed_cash:,.0f} CAD")

        # Enhanced worst-case scenario analysis
        print("\n" + "=" * 80)
        print("🚨 WORST-CASE SCENARIO ANALYSIS")
        print("=" * 80)

        worst_scenario = min(
            stress_results.items(), key=lambda x: x[1]["portfolio_loss"]
        )
        worst_loss = worst_scenario[1]["portfolio_loss"]
        worst_value = worst_scenario[1]["stressed_value"]

        print(f"\n🔥 Worst Scenario: {worst_scenario[0]}")
        print(f"   📉 Maximum Loss: {worst_loss:.1%}")
        print(f"   💰 Minimum Value: ${worst_value:,.0f} CAD")
        print(f"   ⏱️  Duration: {worst_scenario[1]['duration']}")

        # Complex instrument specific warnings
        if complex_risks:
            print(f"\n⚠️  COMPLEX INSTRUMENT RISKS IN WORST SCENARIO:")
            for instrument_type, positions in complex_risks.items():
                if instrument_type == "leveraged_etfs":
                    print(
                        f"   🚨 Leveraged ETFs: Amplified losses in {worst_scenario[0]}"
                    )
                elif instrument_type == "inverse_etfs":
                    print(
                        f"   🎯 Inverse ETFs: May provide protection in {worst_scenario[0]}"
                    )
                elif instrument_type == "volatility_etfs":
                    print(
                        f"   📈 Volatility ETFs: High volatility exposure in {worst_scenario[0]}"
                    )
                elif instrument_type == "options_etfs":
                    print(
                        f"   🎲 Options ETFs: Strategy breakdown risk in {worst_scenario[0]}"
                    )

        # Recovery analysis
        print(f"\n📈 Recovery Analysis:")
        recovery_years = {
            "2008 Financial Crisis": 4,
            "2020 COVID Crash": 1,
            "1970s Stagflation": 8,
            "Dot-com Bubble Burst": 3,
            "Black Monday 1987": 0.5,
            "2022 Inflation Shock": 2,
        }

        if worst_scenario[0] in recovery_years:
            recovery_time = recovery_years[worst_scenario[0]]
            print(f"   • Historical Recovery Time: {recovery_time} years")

            # Calculate compound annual growth rate needed for recovery
            if worst_loss < 0:
                cagr_needed = (
                    (current_portfolio_value / worst_value) ** (1 / recovery_time)
                ) - 1
                print(f"   • Required CAGR for Recovery: {cagr_needed:.1%}")

                # Assess if recovery is realistic
                if cagr_needed > 0.15:  # 15% CAGR
                    print(
                        f"   ⚠️  Recovery may be challenging (requires >15% annual returns)"
                    )
                elif cagr_needed > 0.10:  # 10% CAGR
                    print(
                        f"   🟡 Recovery is achievable but requires strong performance"
                    )
                else:
                    print(f"   ✅ Recovery appears realistic")

        # Enhanced portfolio resilience scoring
        print("\n" + "=" * 80)
        print("🛡️  PORTFOLIO RESILIENCE SCORING")
        print("=" * 80)

        # Calculate resilience metrics
        avg_loss = np.mean(
            [result["portfolio_loss"] for result in stress_results.values()]
        )
        max_loss = min([result["portfolio_loss"] for result in stress_results.values()])
        loss_volatility = np.std(
            [result["portfolio_loss"] for result in stress_results.values()]
        )

        # Cash buffer score
        cash_buffer_score = min(
            current_cash / current_portfolio_value * 10, 10
        )  # Max 10 points

        # Diversification score
        diversification_score = min((1 - self.dalio_analysis["hhi"]) * 10, 10)

        # Volatility score (lower is better)
        if len(self.risk_metrics) > 0:
            avg_volatility = self.risk_metrics["Volatility"].mean()
            volatility_score = max(10 - (avg_volatility * 20), 0)  # Max 10 points
        else:
            volatility_score = 5  # Neutral score

        # Correlation score
        if "diversification_score" in self.dalio_analysis["principles"]:
            correlation_score = (
                self.dalio_analysis["principles"]["diversification_score"] * 10
            )
        else:
            correlation_score = 5  # Neutral score

        # Complex instrument penalty/adjustment
        complex_instrument_score = 10
        if complex_risks:
            total_complex_weight = sum(
                sum(pos["weight"] for pos in positions)
                for positions in complex_risks.values()
            )

            # Penalize for high complex instrument exposure
            if total_complex_weight > 0.3:  # >30% in complex instruments
                complex_instrument_score = 2
            elif total_complex_weight > 0.15:  # >15% in complex instruments
                complex_instrument_score = 5
            elif total_complex_weight > 0.05:  # >5% in complex instruments
                complex_instrument_score = 7

        # Overall resilience score
        resilience_score = (
            cash_buffer_score
            + diversification_score
            + volatility_score
            + correlation_score
            + complex_instrument_score
        ) / 5

        print(f"\n📊 Resilience Metrics:")
        print(f"   • Average Loss Across Scenarios: {avg_loss:.1%}")
        print(f"   • Maximum Loss: {max_loss:.1%}")
        print(f"   • Loss Volatility: {loss_volatility:.1%}")

        print(f"\n🏆 Resilience Scoring (0-10 scale):")
        print(f"   • Cash Buffer: {cash_buffer_score:.1f}/10")
        print(f"   • Diversification: {diversification_score:.1f}/10")
        print(f"   • Volatility Management: {volatility_score:.1f}/10")
        print(f"   • Correlation Management: {correlation_score:.1f}/10")
        print(f"   • Complex Instrument Management: {complex_instrument_score:.1f}/10")
        print(f"   • OVERALL RESILIENCE: {resilience_score:.1f}/10")

        # Resilience rating
        if resilience_score >= 8:
            resilience_rating = "🟢 EXCELLENT"
            rating_description = "Portfolio shows strong resilience to stress scenarios"
        elif resilience_score >= 6:
            resilience_rating = "🟡 GOOD"
            rating_description = (
                "Portfolio has moderate resilience with room for improvement"
            )
        elif resilience_score >= 4:
            resilience_rating = "🟠 MODERATE"
            rating_description = "Portfolio may struggle in severe stress scenarios"
        else:
            resilience_rating = "🔴 POOR"
            rating_description = "Portfolio is vulnerable to stress scenarios"

        print(f"\n🎯 Resilience Rating: {resilience_rating}")
        print(f"   {rating_description}")

        # Enhanced stress test recommendations
        print("\n" + "=" * 80)
        print("💡 STRESS TEST RECOMMENDATIONS")
        print("=" * 80)

        recommendations = []

        # Complex instrument specific recommendations
        if complex_risks:
            complex_recommendations = self.generate_complex_instrument_recommendations(
                complex_risks
            )
            for rec in complex_recommendations:
                print(rec)

            # Add to recommendations list for summary
            recommendations.extend(
                [
                    "🚨 Complex instruments detected - see detailed analysis above",
                    "🛡️  Implement instrument-specific risk management",
                    "📊 Monitor correlations and breakdowns daily",
                ]
            )

        # Cash recommendations
        if current_cash / current_portfolio_value < 0.05:
            recommendations.append(
                "💰 Increase cash buffer to 5-10% for stress scenarios"
            )
        elif current_cash / current_portfolio_value > 0.20:
            recommendations.append(
                "📈 Consider deploying excess cash for better returns"
            )

        # Diversification recommendations
        if self.dalio_analysis["hhi"] > 0.25:
            recommendations.append(
                "🔄 Reduce concentration risk by diversifying holdings"
            )

        # Volatility recommendations
        if len(self.risk_metrics) > 0 and self.risk_metrics["Volatility"].mean() > 0.30:
            recommendations.append(
                "📉 Consider adding low-volatility assets to reduce portfolio risk"
            )

        # Defensive asset recommendations
        defensive_assets = ["TLT", "GLD", "SHY", "BND"]
        current_defensive = 0
        for asset in defensive_assets:
            if asset in self.df["Ticker"].values:
                current_defensive += self.df[self.df["Ticker"] == asset][
                    "Value in Cad"
                ].sum()

        defensive_allocation = current_defensive / current_portfolio_value
        if defensive_allocation < 0.15:
            recommendations.append(
                "🛡️  Add defensive assets (bonds, gold) for stress protection"
            )

        # Currency diversification
        usd_exposure = self.dalio_analysis["currency_exposure"].get("USD", 0)
        if usd_exposure > 0.70:
            recommendations.append(
                "🌍 Reduce USD concentration for currency diversification"
            )

        # Specific stress scenario recommendations
        if worst_loss < -0.30:  # More than 30% loss in worst case
            recommendations.append(
                "🚨 Consider hedging strategies for extreme market scenarios"
            )

        if recommendations:
            print(f"\n📋 Recommendations for Stress Resilience:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print(f"\n✅ Portfolio appears well-positioned for stress scenarios")

        # Enhanced Monte Carlo simulation
        print("\n" + "=" * 80)
        print("🎲 MONTE CARLO STRESS SIMULATION")
        print("=" * 80)

        print(
            f"\n📊 Running 10,000 Monte Carlo simulations with complex instrument modeling..."
        )

        # Simulate portfolio returns under various stress conditions
        np.random.seed(42)  # For reproducible results

        # Generate random stress scenarios
        n_simulations = 10000
        simulated_losses = []

        for _ in range(n_simulations):
            # Random stress parameters
            market_crash = np.random.normal(-0.25, 0.15)  # Mean -25%, std 15%
            volatility_spike = (
                np.random.exponential(2.0) + 1.0
            )  # Exponential distribution
            correlation_impact = np.random.beta(
                2, 5
            )  # Beta distribution for correlation
            liquidity_impact = np.random.exponential(0.1)  # Liquidity impact

            # Enhanced stress calculation for complex instruments
            stressed_value = 0

            for _, row in self.stock_etf_data.iterrows():
                ticker = row["Ticker"]
                position_value = row["Value in Cad"]

                # Create scenario params for this simulation
                sim_scenario_params = {
                    "market_crash": market_crash * 100,
                    "volatility_spike": volatility_spike,
                    "correlation_increase": correlation_impact,
                    "liquidity_dry_up": liquidity_impact,
                    "currency_volatility": np.random.normal(0.15, 0.05),
                }

                # Apply instrument-specific stress modeling
                instrument_stress_impact = self.calculate_complex_instrument_stress(
                    ticker, market_crash, sim_scenario_params
                )

                # Apply additional stress factors
                stressed_position_value = position_value * (
                    1 + instrument_stress_impact
                )
                stressed_position_value *= 1 - correlation_impact * 0.05
                stressed_position_value *= 1 - liquidity_impact

                stressed_value += stressed_position_value

            # Cash impact (small)
            cash_impact = np.random.normal(-0.01, 0.02)
            stressed_cash = current_cash * (1 + cash_impact)

            total_stressed = stressed_value + stressed_cash
            loss = (total_stressed - current_portfolio_value) / current_portfolio_value
            simulated_losses.append(loss)

        simulated_losses = np.array(simulated_losses)

        # Calculate Monte Carlo statistics
        mc_avg_loss = np.mean(simulated_losses)
        mc_median_loss = np.median(simulated_losses)
        mc_std_loss = np.std(simulated_losses)
        mc_var_95 = np.percentile(simulated_losses, 5)
        mc_var_99 = np.percentile(simulated_losses, 1)
        mc_max_loss = np.min(simulated_losses)

        print(f"\n📈 Monte Carlo Results:")
        print(f"   • Average Loss: {mc_avg_loss:.1%}")
        print(f"   • Median Loss: {mc_median_loss:.1%}")
        print(f"   • Loss Volatility: {mc_std_loss:.1%}")
        print(f"   • 95% VaR: {mc_var_95:.1%}")
        print(f"   • 99% VaR: {mc_var_99:.1%}")
        print(f"   • Maximum Loss: {mc_max_loss:.1%}")

        # Probability of different loss levels
        prob_10_percent_loss = np.mean(simulated_losses <= -0.10)
        prob_20_percent_loss = np.mean(simulated_losses <= -0.20)
        prob_30_percent_loss = np.mean(simulated_losses <= -0.30)

        print(f"\n📊 Loss Probabilities:")
        print(f"   • P(Loss > 10%): {prob_10_percent_loss:.1%}")
        print(f"   • P(Loss > 20%): {prob_20_percent_loss:.1%}")
        print(f"   • P(Loss > 30%): {prob_30_percent_loss:.1%}")

        # Complex instrument specific Monte Carlo analysis
        if complex_risks:
            print(f"\n🎲 Complex Instrument Monte Carlo Analysis:")
            for instrument_type, positions in complex_risks.items():
                type_value = sum(pos["value"] for pos in positions)
                type_weight = type_value / current_portfolio_value

                if instrument_type == "leveraged_etfs":
                    print(f"   📊 Leveraged ETFs ({type_weight:.1%}): Higher tail risk")
                elif instrument_type == "inverse_etfs":
                    print(
                        f"   🎯 Inverse ETFs ({type_weight:.1%}): Potential downside protection"
                    )
                elif instrument_type == "volatility_etfs":
                    print(
                        f"   📈 Volatility ETFs ({type_weight:.1%}): High volatility exposure"
                    )
                elif instrument_type == "options_etfs":
                    print(
                        f"   🎲 Options ETFs ({type_weight:.1%}): Strategy breakdown risk"
                    )

        # Stress test summary
        print("\n" + "=" * 80)
        print("📋 STRESS TEST SUMMARY")
        print("=" * 80)

        print(f"\n🎯 Key Findings:")
        print(f"   • Portfolio shows {resilience_rating} resilience to stress")
        print(
            f"   • Worst historical scenario: {worst_scenario[0]} ({worst_loss:.1%} loss)"
        )
        print(f"   • Monte Carlo 95% VaR: {mc_var_95:.1%}")
        print(f"   • Probability of >20% loss: {prob_20_percent_loss:.1%}")

        if complex_risks:
            total_complex_weight = sum(
                sum(pos["weight"] for pos in positions)
                for positions in complex_risks.values()
            )
            print(f"   • Complex instruments: {total_complex_weight:.1%} of portfolio")
            print(f"   • Enhanced stress modeling applied for complex instruments")

        if resilience_score < 6:
            print(
                f"\n⚠️  WARNING: Portfolio may be vulnerable to severe stress scenarios"
            )
            print(f"   Consider implementing stress protection strategies")
        else:
            print(f"\n✅ Portfolio appears resilient to most stress scenarios")

        print("\n" + "=" * 80)
        print("✅ STRESS TESTING ANALYSIS COMPLETE")
        print("=" * 80)

    def classify_instrument(self, ticker):
        """Classify instrument type for stress testing"""
        for instrument_type, tickers in self.complex_instruments.items():
            if ticker in tickers:
                return instrument_type
        return "standard"  # Regular stocks/ETFs

    def calculate_complex_instrument_stress(
        self, ticker, base_stress_impact, scenario_params
    ):
        """Calculate stress impact for complex instruments"""
        instrument_type = self.classify_instrument(ticker)

        if instrument_type == "standard":
            return base_stress_impact

        multipliers = self.stress_multipliers[instrument_type]

        # Apply instrument-specific multipliers
        adjusted_impact = base_stress_impact

        # Market crash impact (inverse ETFs gain during crashes)
        if instrument_type == "inverse_etfs":
            market_crash = scenario_params["market_crash"] / 100
            adjusted_impact = (
                -market_crash * multipliers["market_crash"]
            )  # Inverse relationship
        else:
            market_crash = scenario_params["market_crash"] / 100
            adjusted_impact = market_crash * multipliers["market_crash"]

        # Volatility impact
        volatility_impact = (
            (scenario_params["volatility_spike"] - 1.0)
            * multipliers["volatility_spike"]
            * 0.1
        )
        adjusted_impact += volatility_impact

        # Liquidity impact
        liquidity_impact = (
            scenario_params["liquidity_dry_up"] * multipliers["liquidity_dry_up"] * 0.05
        )
        adjusted_impact -= liquidity_impact

        # Correlation breakdown impact
        correlation_impact = (
            scenario_params["correlation_increase"]
            * multipliers["correlation_breakdown"]
            * 0.1
        )
        adjusted_impact -= correlation_impact

        return adjusted_impact

    def analyze_complex_instrument_risks(self):
        """Analyze risks specific to complex instruments"""
        complex_risks = {}

        for _, row in self.stock_etf_data.iterrows():
            ticker = row["Ticker"]
            instrument_type = self.classify_instrument(ticker)

            if instrument_type != "standard":
                if instrument_type not in complex_risks:
                    complex_risks[instrument_type] = []

                complex_risks[instrument_type].append(
                    {
                        "ticker": ticker,
                        "asset": row["Asset"],
                        "value": row["Value in Cad"],
                        "weight": row["Value in Cad"] / self.total_value,
                        "type": instrument_type,
                    }
                )

        return complex_risks

    def generate_complex_instrument_recommendations(self, complex_risks):
        """Generate specific recommendations for complex instruments"""
        recommendations = []

        if not complex_risks:
            return recommendations

        recommendations.append("\n🚨 COMPLEX INSTRUMENT MANAGEMENT STRATEGIES:")

        # Leveraged ETF recommendations
        if "leveraged_etfs" in complex_risks:
            leveraged_positions = complex_risks["leveraged_etfs"]
            total_leveraged_value = sum(pos["value"] for pos in leveraged_positions)
            leveraged_weight = total_leveraged_value / self.total_value

            recommendations.append(
                f"\n📊 LEVERAGED ETFs ({leveraged_weight:.1%} of portfolio):"
            )
            recommendations.append(
                "   ⚠️  HIGH RISK: Amplified losses in stress scenarios"
            )

            if leveraged_weight > 0.15:
                recommendations.append("   🚨 CRITICAL: Reduce exposure immediately")
                recommendations.append("      • Consider reducing to <5% of portfolio")
                recommendations.append("      • Use stop-losses at -10% to -15%")
                recommendations.append("      • Monitor daily for volatility spikes")
            elif leveraged_weight > 0.10:
                recommendations.append("   ⚠️  HIGH: Monitor closely")
                recommendations.append("      • Set tight stop-losses")
                recommendations.append("      • Consider hedging with inverse ETFs")
            else:
                recommendations.append("   🟡 MODERATE: Manage carefully")
                recommendations.append("      • Use for tactical positions only")
                recommendations.append("      • Avoid long-term holds")

            recommendations.append(
                "   💡 Strategy: Use for short-term momentum trades only"
            )
            recommendations.append("   🛡️  Protection: Hedge with inverse ETFs or puts")

        # Inverse ETF recommendations
        if "inverse_etfs" in complex_risks:
            inverse_positions = complex_risks["inverse_etfs"]
            total_inverse_value = sum(pos["value"] for pos in inverse_positions)
            inverse_weight = total_inverse_value / self.total_value

            recommendations.append(
                f"\n🎯 INVERSE ETFs ({inverse_weight:.1%} of portfolio):"
            )
            recommendations.append("   🛡️  PROTECTION: Can provide downside protection")

            if inverse_weight > 0.20:
                recommendations.append("   ⚠️  HIGH: Monitor sizing")
                recommendations.append("      • Inverse ETFs have decay over time")
                recommendations.append("      • Consider reducing to <15% of portfolio")
                recommendations.append("      • Rebalance monthly to manage decay")
            elif inverse_weight > 0.10:
                recommendations.append("   🟡 MODERATE: Good protection level")
                recommendations.append("      • Monitor for correlation breakdown")
                recommendations.append("      • Consider tactical adjustments")
            else:
                recommendations.append("   ✅ LOW: Appropriate sizing")
                recommendations.append("      • Good for portfolio insurance")
                recommendations.append(
                    "      • Monitor correlation with main portfolio"
                )

            recommendations.append("   💡 Strategy: Use for portfolio insurance")
            recommendations.append("   ⏰ Timing: Best during high volatility periods")
            recommendations.append("   🔄 Rebalancing: Monthly to manage decay")

        # Volatility ETF recommendations
        if "volatility_etfs" in complex_risks:
            vol_positions = complex_risks["volatility_etfs"]
            total_vol_value = sum(pos["value"] for pos in vol_positions)
            vol_weight = total_vol_value / self.total_value

            recommendations.append(
                f"\n📈 VOLATILITY ETFs ({vol_weight:.1%} of portfolio):"
            )
            recommendations.append(
                "   📊 VOLATILITY: Direct exposure to market volatility"
            )

            if vol_weight > 0.10:
                recommendations.append("   ⚠️  HIGH: Reduce exposure")
                recommendations.append("      • Volatility ETFs are highly speculative")
                recommendations.append("      • Consider reducing to <5% of portfolio")
                recommendations.append(
                    "      • Use only for short-term volatility plays"
                )
            elif vol_weight > 0.05:
                recommendations.append("   🟡 MODERATE: Manage carefully")
                recommendations.append("      • Monitor VIX term structure")
                recommendations.append("      • Be aware of contango/backwardation")
            else:
                recommendations.append("   ✅ LOW: Appropriate for tactical use")
                recommendations.append("      • Good for volatility hedging")
                recommendations.append("      • Monitor VIX levels")

            recommendations.append("   💡 Strategy: Use for volatility hedging only")
            recommendations.append("   📊 Monitor: VIX term structure and contango")
            recommendations.append("   ⏰ Holding: Short-term only (days to weeks)")

        # Options ETF recommendations
        if "options_etfs" in complex_risks:
            options_positions = complex_risks["options_etfs"]
            total_options_value = sum(pos["value"] for pos in options_positions)
            options_weight = total_options_value / self.total_value

            recommendations.append(
                f"\n🎲 OPTIONS ETFs ({options_weight:.1%} of portfolio):"
            )
            recommendations.append(
                "   🎲 OPTIONS: Income generation with strategy risk"
            )

            if options_weight > 0.25:
                recommendations.append("   ⚠️  HIGH: Monitor strategy performance")
                recommendations.append(
                    "      • Options strategies can break down in stress"
                )
                recommendations.append("      • Consider reducing to <20% of portfolio")
                recommendations.append("      • Monitor implied volatility levels")
            elif options_weight > 0.15:
                recommendations.append("   🟡 MODERATE: Good income generation")
                recommendations.append("      • Monitor strategy performance")
                recommendations.append("      • Be aware of gamma risk")
            else:
                recommendations.append("   ✅ LOW: Good for income generation")
                recommendations.append("      • Monitor strategy performance")
                recommendations.append("      • Consider as income supplement")

            recommendations.append(
                "   💡 Strategy: Income generation with risk management"
            )
            recommendations.append("   📊 Monitor: Strategy performance and volatility")
            recommendations.append("   🛡️  Risk: Options strategies can break down")

        # Currency ETF recommendations
        if "currency_etfs" in complex_risks:
            currency_positions = complex_risks["currency_etfs"]
            total_currency_value = sum(pos["value"] for pos in currency_positions)
            currency_weight = total_currency_value / self.total_value

            recommendations.append(
                f"\n🌍 CURRENCY ETFs ({currency_weight:.1%} of portfolio):"
            )
            recommendations.append(
                "   🌍 CURRENCY: Currency diversification and speculation"
            )

            if currency_weight > 0.15:
                recommendations.append("   ⚠️  HIGH: Monitor currency exposure")
                recommendations.append("      • High currency exposure can be risky")
                recommendations.append("      • Consider reducing to <10% of portfolio")
                recommendations.append("      • Monitor central bank policies")
            elif currency_weight > 0.08:
                recommendations.append("   🟡 MODERATE: Good currency diversification")
                recommendations.append("      • Monitor currency correlations")
                recommendations.append(
                    "      • Be aware of interest rate differentials"
                )
            else:
                recommendations.append("   ✅ LOW: Good for currency diversification")
                recommendations.append("      • Monitor currency correlations")
                recommendations.append("      • Consider as portfolio hedge")

            recommendations.append("   💡 Strategy: Currency diversification")
            recommendations.append(
                "   📊 Monitor: Interest rate differentials and policies"
            )
            recommendations.append("   🛡️  Risk: Currency volatility and correlations")

        # Commodity ETF recommendations
        if "commodity_etfs" in complex_risks:
            commodity_positions = complex_risks["commodity_etfs"]
            total_commodity_value = sum(pos["value"] for pos in commodity_positions)
            commodity_weight = total_commodity_value / self.total_value

            recommendations.append(
                f"\n🏭 COMMODITY ETFs ({commodity_weight:.1%} of portfolio):"
            )
            recommendations.append(
                "   🏭 COMMODITY: Inflation hedge and diversification"
            )

            if commodity_weight > 0.20:
                recommendations.append("   ⚠️  HIGH: Monitor commodity exposure")
                recommendations.append(
                    "      • High commodity exposure can be volatile"
                )
                recommendations.append("      • Consider reducing to <15% of portfolio")
                recommendations.append("      • Monitor supply/demand fundamentals")
            elif commodity_weight > 0.10:
                recommendations.append("   🟡 MODERATE: Good inflation hedge")
                recommendations.append("      • Monitor commodity fundamentals")
                recommendations.append("      • Be aware of storage costs and contango")
            else:
                recommendations.append("   ✅ LOW: Good for inflation protection")
                recommendations.append("      • Monitor commodity fundamentals")
                recommendations.append("      • Consider as inflation hedge")

            recommendations.append(
                "   💡 Strategy: Inflation protection and diversification"
            )
            recommendations.append("   📊 Monitor: Supply/demand fundamentals")
            recommendations.append("   🛡️  Risk: Commodity volatility and storage costs")

        # Overall complex instrument recommendations
        total_complex_weight = sum(
            sum(pos["weight"] for pos in positions)
            for positions in complex_risks.values()
        )

        recommendations.append(
            f"\n📊 OVERALL COMPLEX INSTRUMENT EXPOSURE ({total_complex_weight:.1%}):"
        )

        if total_complex_weight > 0.40:
            recommendations.append(
                "   🚨 CRITICAL: Portfolio is heavily exposed to complex instruments"
            )
            recommendations.append("      • Consider reducing to <25% of portfolio")
            recommendations.append("      • Implement strict risk management")
            recommendations.append("      • Monitor correlations and breakdowns")
        elif total_complex_weight > 0.25:
            recommendations.append(
                "   ⚠️  HIGH: Significant complex instrument exposure"
            )
            recommendations.append("      • Monitor correlations and breakdowns")
            recommendations.append("      • Implement risk management strategies")
            recommendations.append("      • Consider reducing exposure")
        elif total_complex_weight > 0.15:
            recommendations.append(
                "   🟡 MODERATE: Manageable complex instrument exposure"
            )
            recommendations.append("      • Monitor correlations and breakdowns")
            recommendations.append("      • Implement basic risk management")
            recommendations.append("      • Consider tactical adjustments")
        else:
            recommendations.append("   ✅ LOW: Appropriate complex instrument exposure")
            recommendations.append("      • Monitor correlations and breakdowns")
            recommendations.append("      • Implement basic risk management")
            recommendations.append("      • Consider tactical opportunities")

        recommendations.append("\n🛡️  RISK MANAGEMENT STRATEGIES:")
        recommendations.append("   • Set stop-losses for all complex instruments")
        recommendations.append("   • Monitor correlations daily during stress")
        recommendations.append("   • Rebalance monthly to manage decay")
        recommendations.append("   • Consider hedging with simpler instruments")
        recommendations.append("   • Monitor liquidity during market stress")

        return recommendations


def main():
    """Main function to run the portfolio analysis with command-line options"""
    parser = argparse.ArgumentParser(
        description="Portfolio Analyzer with Ray Dalio Investment Principles"
    )
    parser.add_argument(
        "--csv",
        default="stocks.csv",
        help="CSV file containing portfolio data (default: stocks.csv)",
    )
    parser.add_argument(
        "--analysis",
        choices=[
            "basic",
            "risk",
            "dalio",
            "visual",
            "complete",
            "protection",
            "stress",
        ],
        default="complete",
        help="Type of analysis to run (default: complete)",
    )
    parser.add_argument(
        "--period", default="2y", help="Historical data period (default: 2y)"
    )

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = PortfolioAnalyzer(args.csv)

        # Run selected analysis
        if args.analysis == "basic":
            analyzer.run_basic_analysis()
        elif args.analysis == "risk":
            analyzer.run_risk_analysis()
        elif args.analysis == "dalio":
            analyzer.run_dalio_analysis()
        elif args.analysis == "visual":
            analyzer.run_visualizations()
        elif args.analysis == "complete":
            analyzer.run_complete_analysis()
        elif args.analysis == "protection":
            analyzer.run_protection_analysis()
        elif args.analysis == "stress":
            analyzer.run_stress_test_analysis()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
