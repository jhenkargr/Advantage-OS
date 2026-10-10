"""
Test script for Demand Prediction + Stockout Risk models
=========================================================

Loads the trained .pkl models and runs representative scenarios
to verify that inference works correctly end-to-end.
"""

import sys
import os

# Ensure imports resolve when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.demand_model import predict_future_demand
from src.stockout_model import predict_stockout

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEPARATOR = "=" * 60


def test_demand_predictions():
    """Test the demand regressor with high-demand and low-demand inputs."""
    print(SEPARATOR)
    print("DEMAND PREDICTION TESTS")
    print(SEPARATOR)
    print()

    # --- HIGH DEMAND SCENARIO ---
    high = predict_future_demand(
        current_stock=20,
        avg_daily_sales_7d=10.0,
        sales_std_7d=2.5,
        trend_score=0.8,
    )
    print("HIGH DEMAND SCENARIO")
    print(f"  Inputs:  stock=20, avg_sales=10.0, std=2.5, trend=0.8")
    print(f"  Predicted 7-day demand: {high} units")
    print()

    # --- LOW DEMAND SCENARIO ---
    low = predict_future_demand(
        current_stock=100,
        avg_daily_sales_7d=2.0,
        sales_std_7d=0.5,
        trend_score=0.0,
    )
    print("LOW DEMAND SCENARIO")
    print(f"  Inputs:  stock=100, avg_sales=2.0, std=0.5, trend=0.0")
    print(f"  Predicted 7-day demand: {low} units")
    print()

    # Basic sanity
    assert high >= 0, "Demand prediction should be non-negative"
    assert low >= 0, "Demand prediction should be non-negative"
    assert isinstance(high, int), "Demand prediction should be an integer"
    assert isinstance(low, int), "Demand prediction should be an integer"
    print("  All demand sanity checks PASSED")
    print()


def test_stockout_predictions():
    """Test the stockout classifier with high-risk and low-risk inputs."""
    print(SEPARATOR)
    print("STOCKOUT RISK TESTS")
    print(SEPARATOR)
    print()

    # --- HIGH RISK (low stock, high sales) ---
    high_risk = predict_stockout(
        current_stock=5,
        avg_daily_sales_7d=12.0,
        sales_std_7d=3.0,
        trend_score=0.8,
    )
    print("HIGH RISK SCENARIO")
    print(f"  Inputs:  stock=5, avg_sales=12.0, std=3.0, trend=0.8")
    print(f"  Prediction: {high_risk['stockout_prediction']}")
    print(f"  Stockout probability: {high_risk['stockout_probability']}")
    print()

    # --- LOW RISK (high stock, low sales) ---
    low_risk = predict_stockout(
        current_stock=150,
        avg_daily_sales_7d=2.0,
        sales_std_7d=0.5,
        trend_score=0.0,
    )
    print("LOW RISK SCENARIO")
    print(f"  Inputs:  stock=150, avg_sales=2.0, std=0.5, trend=0.0")
    print(f"  Prediction: {low_risk['stockout_prediction']}")
    print(f"  Stockout probability: {low_risk['stockout_probability']}")
    print()

    # Basic sanity
    assert high_risk["stockout_prediction"] in (0, 1)
    assert low_risk["stockout_prediction"] in (0, 1)
    assert 0.0 <= high_risk["stockout_probability"] <= 1.0
    assert 0.0 <= low_risk["stockout_probability"] <= 1.0
    print("  All stockout sanity checks PASSED")
    print()


def main():
    print()
    print(SEPARATOR)
    print("  INVENTORY MODULE -- MODEL TEST SUITE")
    print(SEPARATOR)
    print()

    test_demand_predictions()
    test_stockout_predictions()

    print(SEPARATOR)
    print("ALL TESTS PASSED SUCCESSFULLY")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
