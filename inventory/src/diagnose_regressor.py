"""
Diagnostic script for demand regressor prediction compression.
Checks whether the MLP is compressing predictions toward the mean.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Allow imports from inventory/src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.demand_model import build_features, FEATURE_COLS, SALES_PATH, MODEL_SAVE_PATH

SEP = "=" * 60


def run_diagnostics():
    # ------------------------------------------------------------------
    # 1. Load data and model
    # ------------------------------------------------------------------
    sales = pd.read_csv(SALES_PATH)
    sales["date"] = pd.to_datetime(sales["date"])
    model = joblib.load(MODEL_SAVE_PATH)

    df = build_features(sales)
    X = df[FEATURE_COLS]
    y_true = df["future_demand_7d"].values
    y_pred = model.predict(X)

    # ------------------------------------------------------------------
    # 2. Distribution diagnostics
    # ------------------------------------------------------------------
    print(SEP)
    print("PREDICTION COMPRESSION DIAGNOSTIC")
    print(SEP)
    print()

    true_std = np.std(y_true)
    pred_std = np.std(y_pred)
    compression_ratio = pred_std / true_std

    print("TRUE TARGET (future_demand_7d)")
    print(f"  min:  {np.min(y_true):.1f}")
    print(f"  max:  {np.max(y_true):.1f}")
    print(f"  mean: {np.mean(y_true):.1f}")
    print(f"  std:  {true_std:.3f}")
    print()

    print("PREDICTED TARGET")
    print(f"  min:  {np.min(y_pred):.1f}")
    print(f"  max:  {np.max(y_pred):.1f}")
    print(f"  mean: {np.mean(y_pred):.1f}")
    print(f"  std:  {pred_std:.3f}")
    print()

    print(f"Prediction compression ratio (pred_std / true_std): {compression_ratio:.3f}")
    print()

    # ------------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------------
    avg_sales = df["avg_daily_sales_7d"].values
    stock = df["current_stock"].values

    corr_pred_sales = np.corrcoef(y_pred, avg_sales)[0, 1]
    corr_true_sales = np.corrcoef(y_true, avg_sales)[0, 1]
    corr_pred_stock = np.corrcoef(y_pred, stock)[0, 1]
    corr_true_stock = np.corrcoef(y_true, stock)[0, 1]

    print("CORRELATIONS")
    print(f"  corr(prediction, avg_daily_sales_7d): {corr_pred_sales:.3f}")
    print(f"  corr(true_target, avg_daily_sales_7d): {corr_true_sales:.3f}")
    print(f"  corr(prediction, current_stock):       {corr_pred_stock:.3f}")
    print(f"  corr(true_target, current_stock):       {corr_true_stock:.3f}")
    print()

    # ------------------------------------------------------------------
    # 3. Naive baseline: avg_daily_sales_7d * 7
    # ------------------------------------------------------------------
    baseline = avg_sales * 7

    b_mae = mean_absolute_error(y_true, baseline)
    b_rmse = root_mean_squared_error(y_true, baseline)
    b_r2 = r2_score(y_true, baseline)

    m_mae = mean_absolute_error(y_true, y_pred)
    m_rmse = root_mean_squared_error(y_true, y_pred)
    m_r2 = r2_score(y_true, y_pred)

    print(SEP)
    print("NAIVE BASELINE vs CURRENT MLP")
    print(SEP)
    print()
    print(f"{'Metric':<8} {'Naive baseline':>16} {'Current MLP':>16}")
    print("-" * 44)
    print(f"{'MAE':<8} {b_mae:>16.4f} {m_mae:>16.4f}")
    print(f"{'RMSE':<8} {b_rmse:>16.4f} {m_rmse:>16.4f}")
    print(f"{'R2':<8} {b_r2:>16.4f} {m_r2:>16.4f}")
    print()

    # ------------------------------------------------------------------
    # 4. Extreme scenario diagnostic
    # ------------------------------------------------------------------
    print(SEP)
    print("EXTREME SCENARIO DIAGNOSTIC")
    print(SEP)
    print()

    scenarios = [
        ("HIGH DEMAND", {"avg_daily_sales_7d": 10.0, "sales_std_7d": 2.5,
                         "current_stock": 20, "trend_score": 0.8}),
        ("LOW DEMAND",  {"avg_daily_sales_7d": 2.0,  "sales_std_7d": 0.5,
                         "current_stock": 100, "trend_score": 0.0}),
    ]

    for label, inputs in scenarios:
        features = pd.DataFrame([inputs], columns=FEATURE_COLS)
        pred = model.predict(features)[0]
        naive = inputs["avg_daily_sales_7d"] * 7

        print(f"{label} SCENARIO")
        print(f"  Inputs: stock={inputs['current_stock']}, "
              f"avg_sales={inputs['avg_daily_sales_7d']}, "
              f"std={inputs['sales_std_7d']}, "
              f"trend={inputs['trend_score']}")
        print(f"  MLP prediction:     {pred:.1f} units")
        print(f"  Naive baseline:     {naive:.1f} units")
        print()

    print(SEP)
    print("DIAGNOSTIC COMPLETE")
    print(SEP)


if __name__ == "__main__":
    run_diagnostics()
