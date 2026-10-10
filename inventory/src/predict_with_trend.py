"""
CLI & Interactive Tool: Demand Forecasting & Stock Requirement Engine with Custom Trends.

Allows specifying custom trend scores per product or across the entire catalog,
computes 7-day predicted demand and required stock levels to maintain,
and surfaces comprehensive machine learning evaluation metrics (loss, accuracy, MAE, R2).
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# Allow relative imports if run from root or inventory directory
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from inventory_service import (
    get_inventory_prediction,
    get_all_inventory_predictions,
    EXPANDED_MODEL_METRICS,
)


def print_ml_metrics_card():
    """Print a structured card of the machine learning model performance and evaluation metrics."""
    dm = EXPANDED_MODEL_METRICS["demand_model"]
    sm = EXPANDED_MODEL_METRICS["stockout_model"]

    print("\n" + "=" * 90)
    print("                    MACHINE LEARNING MODEL EVALUATION & METRICS")
    print("=" * 90)
    print(f" 1. DEMAND REGRESSION MODEL  : {dm['model_name']}")
    print(f"    - Architecture           : {dm['architecture']}")
    print(f"    - Training Loss Function : {dm['loss_function']}")
    print(f"    - Forecast Horizon       : {dm['horizon_days']} days (unconstrained potential demand)")
    print(f"    - Test MAE (Mean Abs Err): {dm['test_mae']:.2f} units  (Average unit prediction error: +/- 15 units)")
    print(f"    - Test RMSE (Root MSE)   : {dm['test_rmse']:.2f} units")
    print(f"    - Test R-Squared (R^2)   : {dm['test_r2']:.4f}     (Explains ~43% variance; beats naive baseline 0.098)")
    print(f"    - Validation MAE / R^2   : {dm['val_mae']:.2f} units / {dm['val_r2']:.4f}")
    print("-" * 90)
    print(f" 2. STOCKOUT RISK CLASSIFIER : {sm['model_name']}")
    print(f"    - Loss Function          : {sm['loss_function']}")
    print(f"    - Prediction Horizon     : {sm['horizon_days']} days (binary risk: stock reaches 0)")
    print(f"    - Test Accuracy          : {sm['test_accuracy'] * 100:.2f}%")
    print(f"    - Test Precision         : {sm['test_precision'] * 100:.2f}%")
    print(f"    - Test Recall (Sens.)    : {sm['test_recall'] * 100:.2f}%  (Catches ~89% of true stockout events!)")
    print(f"    - Test F1-Score          : {sm['test_f1'] * 100:.2f}%")
    print(f"    - Validation Accuracy/F1 : {sm['val_accuracy'] * 100:.2f}% / {sm['val_f1'] * 100:.2f}%")
    print("=" * 90 + "\n")


def print_prediction_table(predictions: list[dict]):
    """Format and display predictions for each product in a clear tabular layout."""
    header = (
        f"{'SKU':<8} | {'Product Name':<28} | {'Stock':<6} | {'Trend':<6} | "
        f"{'Demand(7d)':<10} | {'Safety':<6} | {'Req.Stock':<10} | {'Reorder':<8} | {'Risk':<7} | {'Alert':<6}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for p in predictions:
        p_name = p["product_name"][:28]
        cur_stock = p["current_stock"]
        trend = f"{p['trend_score']:.2f}"
        demand = f"{p['predicted_7_day_demand']} u"
        safety = f"{p['safety_stock']} u"
        req_stock = f"{p['required_stock_to_maintain']} u"
        reorder = f"{p['recommended_order_quantity']} u"
        risk = f"{p['stockout_probability'] * 100:.1f}%"
        alert = p["alert_level"]

        print(
            f"{p['product_id']:<8} | {p_name:<28} | {cur_stock:>6} | {trend:>6} | "
            f"{demand:>10} | {safety:>6} | {req_stock:>10} | {reorder:>8} | {risk:>7} | {alert:<6}"
        )
    print(separator)


def run_pipeline(
    sku: str | None = None,
    trend_score: float | None = None,
    export_csv: str | None = None,
):
    """Execute demand and inventory predictions and print results."""
    print("\n" + "=" * 90)
    print("       ADVANTAGE OS -- INVENTORY DEMAND & STOCK REQUIREMENT FORECASTER")
    print("=" * 90)

    if sku:
        print(f"[*] Running prediction for single SKU: {sku}")
        if trend_score is not None:
            print(f"[*] Applied custom trend score: {trend_score:.2f}")
        else:
            print("[*] Using historical trend score from sales log")

        pred = get_inventory_prediction(sku, trend_score=trend_score)
        predictions = [pred]
        print_prediction_table(predictions)

        print("\n[+] Detailed Breakdown for", pred["product_name"], f"({sku}):")
        print(f"    - Current Stock on Hand     : {pred['current_stock']} units")
        print(f"    - Trend Score Input         : {pred['trend_score']}")
        print(f"    - Average Daily Sales (7d)  : {pred['avg_daily_sales_7d']} units/day")
        print(f"    - Predicted 7-Day Demand    : {pred['predicted_7_day_demand']} units "
              f"(Confidence Range: {pred['confidence_interval_demand']['min_demand']} - {pred['confidence_interval_demand']['max_demand']} units)")
        print(f"    - Safety Buffer Required    : {pred['safety_stock']} units")
        print(f"    - Required Stock to Maintain: {pred['required_stock_to_maintain']} units (Demand + Safety Buffer)")
        print(f"    - Recommended Reorder Order : {pred['recommended_order_quantity']} units")
        print(f"    - 3-Day Stockout Probability: {pred['stockout_probability'] * 100:.1f}% -> Alert: {pred['alert_level']}")

    else:
        trend_label = f"Custom Trend Score ({trend_score:.2f})" if trend_score is not None else "Historical Log Trends"
        print(f"[*] Running predictions for ALL catalog products using: {trend_label}")

        summary = get_all_inventory_predictions(default_trend_score=trend_score)
        predictions = summary["predictions"]
        print_prediction_table(predictions)

        print("\n" + "=" * 90)
        print("                              PORTFOLIO SUMMARY")
        print("=" * 90)
        print(f"  Total Catalog SKUs Evaluated  : {summary['total_skus']}")
        print(f"  Total Current Stock on Hand   : {summary['total_units_in_stock']:,} units")
        print(f"  Total 7-Day Predicted Demand  : {summary['total_predicted_7_day_demand']:,} units")
        print(f"  Total Required Stock Level    : {summary['total_required_stock']:,} units (Demand + Safety Buffers)")
        print(f"  Total Recommended Reorders    : {summary['total_reorder_units']:,} units")
        print(f"  High Stockout Risk Products   : {summary['high_risk_count']}")
        print(f"  Medium Risk Products          : {summary['medium_risk_count']}")
        print("=" * 90)

    # Display ML evaluation metrics card
    print_ml_metrics_card()

    # Optional CSV export
    if export_csv:
        df_export = pd.DataFrame([
            {
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "category": p["category"],
                "current_stock": p["current_stock"],
                "trend_score": p["trend_score"],
                "predicted_7_day_demand": p["predicted_7_day_demand"],
                "safety_stock": p["safety_stock"],
                "required_stock_to_maintain": p["required_stock_to_maintain"],
                "recommended_order_quantity": p["recommended_order_quantity"],
                "stockout_probability": p["stockout_probability"],
                "alert_level": p["alert_level"],
            }
            for p in predictions
        ])
        out_path = Path(export_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_export.to_csv(out_path, index=False)
        print(f"[SUCCESS] Forecast results successfully exported to: {out_path.resolve()}\n")


def interactive_mode():
    """Prompt user interactively for SKU and trend score."""
    print("\n=== AdVantage OS Interactive Trend & Demand Forecaster ===")
    print("Leave SKU empty to forecast across ALL 40 products.")
    sku_in = input("Enter Product SKU (e.g., SKU001 or press Enter for ALL): ").strip()
    sku = sku_in if sku_in else None

    trend_in = input("Enter Trend Score (e.g. 0.8 for viral, 0.0 for baseline, or Enter for default): ").strip()
    trend = float(trend_in) if trend_in else None

    run_pipeline(sku=sku, trend_score=trend)


def main():
    parser = argparse.ArgumentParser(
        description="Forecast product demand and required stock levels using ML models with custom trend inputs."
    )
    parser.add_argument(
        "--sku",
        type=str,
        default=None,
        help="Optional SKU ID (e.g. SKU001). If omitted, evaluates all 40 products.",
    )
    parser.add_argument(
        "--trend",
        type=float,
        default=None,
        help="Custom trend score override (e.g., 0.8 for viral spike, 0.0 for normal, 1.2 for hyper-trend).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal prompt to input SKU and trend values.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to export the forecast results as a CSV file.",
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        run_pipeline(sku=args.sku, trend_score=args.trend, export_csv=args.csv)


if __name__ == "__main__":
    main()
