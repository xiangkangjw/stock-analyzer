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
            "ETF": "MFC.TO",  # Manulife Financial Corp
            "SHV": "SHV",
            "VOO": "VOO",
            "AAPL": "AAPL",
            "MSFT": "MSFT",
            "AVGO": "AVGO",
            "GOOG": "GOOG",
            "NVDA": "NVDA",
            "TSLA": "TSLA",
            "AMD": "AMD",
        }

    def map_ticker(self, ticker):
        """Map ticker symbol to correct format for yfinance"""
        return self.ticker_mapper.get(ticker, ticker)

    def clean_data(self):
        """Clean and prepare portfolio data"""
        # Remove rows with missing tickers or values
        self.df = self.df.dropna(subset=["Ticker", "Value in Cad"])

        # Convert Value in Cad to numeric, handling any formatting issues
        self.df["Value in Cad"] = pd.to_numeric(
            self.df["Value in Cad"], errors="coerce"
        )
        self.df = self.df.dropna(subset=["Value in Cad"])

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
        tickers = self.stock_etf_data["Ticker"].unique()
        historical_data = {}

        print("Fetching historical data for risk analysis...")
        for ticker in tickers:
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

        # Top Holdings
        print("\n🏆 TOP 5 HOLDINGS:")
        for _, row in self.dalio_analysis["top_holdings"].iterrows():
            print(
                f"   {row['Asset']}: {row['Weight']:.1f}% (${row['Value in Cad']:,.0f})"
            )

        # Individual Position Risk Analysis
        if len(self.risk_metrics) > 0:
            self.print_individual_risk_analysis()

        # Risk Metrics
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
        if "diversification_score" in principles:
            print(
                f"   Diversification Score: {principles['diversification_score']:.2f}"
            )
        if "risk_parity_score" in principles:
            print(f"   Risk Parity Score: {principles['risk_parity_score']:.2f}")
        print(f"   Asset Class Diversity: {principles['asset_class_diversity']}/4")
        print(
            f"   Geographic Diversification: {principles['geographic_diversification']:.2f}"
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

        # Recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   {rec}")
        else:
            print(
                "\n✅ No major concerns identified. Portfolio appears well-structured."
            )

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

    def run_basic_analysis(self):
        """Run basic portfolio analysis without risk metrics"""
        print("Running Basic Portfolio Analysis...")
        self.clean_data()
        self.analyze_diversification()
        self.apply_dalio_principles()
        self.print_detailed_analysis()

    def run_risk_analysis(self):
        """Run comprehensive risk analysis"""
        print("Running Comprehensive Risk Analysis...")
        self.clean_data()
        historical_data = self.get_historical_data()
        if historical_data:
            self.calculate_advanced_risk_metrics(historical_data)
            self.print_individual_risk_analysis()
            self.print_risk_comparison_summary()
        else:
            print("No historical data available for risk analysis")

    def run_dalio_analysis(self):
        """Run Ray Dalio principles analysis"""
        print("Running Ray Dalio Principles Analysis...")
        self.clean_data()
        self.analyze_diversification()
        self.apply_dalio_principles()

        print("\n" + "=" * 60)
        print("RAY DALIO PRINCIPLES ANALYSIS")
        print("=" * 60)

        principles = self.dalio_analysis["principles"]
        print(
            f"\n🎯 Diversification Score: {principles.get('diversification_score', 'N/A'):.2f}"
        )
        print(f"   - Measures how uncorrelated your assets are")
        print(f"   - Higher is better (0-1 scale)")

        print(
            f"\n⚖️  Risk Parity Score: {principles.get('risk_parity_score', 'N/A'):.2f}"
        )
        print(f"   - Measures equal risk contribution across positions")
        print(f"   - Higher is better")

        print(f"\n🌍 Asset Class Diversity: {principles['asset_class_diversity']}/4")
        print(f"   - Stocks, ETFs, Bonds, Cash")

        print(
            f"\n🌐 Geographic Diversification: {principles['geographic_diversification']:.2f}"
        )
        print(f"   - Measures currency and regional exposure")
        print(f"   - Higher is better (0-1 scale)")

        # Recommendations based on Dalio principles
        recommendations = self.generate_recommendations()
        if recommendations:
            print(f"\n💡 Dalio-Based Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")

    def run_visualizations(self):
        """Run portfolio visualizations"""
        print("Creating Portfolio Visualizations...")
        self.clean_data()
        historical_data = self.get_historical_data()
        if historical_data:
            self.calculate_advanced_risk_metrics(historical_data)
        self.analyze_diversification()
        self.apply_dalio_principles()
        self.create_visualizations()

    def run_complete_analysis(self):
        """Run complete portfolio analysis"""
        print("Running Complete Portfolio Analysis...")
        self.clean_data()
        historical_data = self.get_historical_data()
        if historical_data:
            self.calculate_advanced_risk_metrics(historical_data)
        self.analyze_diversification()
        self.apply_dalio_principles()
        self.print_detailed_analysis()
        self.print_risk_comparison_summary()
        self.create_visualizations()


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
        choices=["basic", "risk", "dalio", "visual", "complete"],
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

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
