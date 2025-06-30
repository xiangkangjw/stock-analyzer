#!/usr/bin/env python3
"""
Test script to demonstrate enhanced stress testing with complex instruments
"""

import pandas as pd
import tempfile
import os


def create_test_portfolio_with_complex_instruments():
    """Create a test portfolio with various complex instruments"""

    # Test portfolio with complex instruments
    test_data = {
        "Asset": [
            "S&P 500 ETF",
            "Inverse S&P 500",
            "3x Leveraged Tech",
            "Volatility ETF",
            "Covered Call ETF",
            "Gold ETF",
            "US Dollar ETF",
            "Cash Position",
        ],
        "Ticker": ["VOO", "SH", "TQQQ", "VXX", "XYLD", "GLD", "UUP", "CASH"],
        "Group": ["VOO", "SH", "TQQQ", "VXX", "XYLD", "GLD", "UUP", "CASH"],
        "Price": [450.0, 25.0, 45.0, 15.0, 35.0, 180.0, 28.0, 1.0],
        "Holding": [100, 200, 50, 100, 150, 50, 100, 10000],
        "Value": [45000, 5000, 2250, 1500, 5250, 9000, 2800, 10000],
        "US/CAD": ["USD", "USD", "USD", "USD", "USD", "USD", "USD", "CAD"],
        "Stock/ETF": ["ETF", "ETF", "ETF", "ETF", "ETF", "ETF", "ETF", "Cash"],
        "Value in Cad": [60750, 6750, 3037.5, 2025, 7087.5, 12150, 3780, 10000],
    }

    return pd.DataFrame(test_data)


def main():
    """Run the enhanced stress testing analysis"""

    # Create test portfolio
    test_df = create_test_portfolio_with_complex_instruments()

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        test_df.to_csv(f.name, index=False)
        csv_file = f.name

    try:
        # Import and run analysis
        from portfolio_analyzer import PortfolioAnalyzer

        print("🧪 TESTING ENHANCED STRESS TESTING WITH COMPLEX INSTRUMENTS")
        print("=" * 80)

        # Initialize analyzer
        analyzer = PortfolioAnalyzer(csv_file)

        # Run stress testing analysis
        analyzer.run_stress_test_analysis()

        print("\n" + "=" * 80)
        print("✅ TEST COMPLETE")
        print("=" * 80)
        print("\nKey improvements demonstrated:")
        print("• Complex instrument detection and classification")
        print("• Instrument-specific stress modeling")
        print("• Enhanced risk assessment for inverse/leveraged ETFs")
        print("• Detailed recommendations for complex instruments")
        print("• Improved Monte Carlo simulation with instrument-specific factors")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up temporary file
        if os.path.exists(csv_file):
            os.unlink(csv_file)


if __name__ == "__main__":
    main()
