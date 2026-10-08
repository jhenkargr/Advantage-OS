"""
Demand Model Comparison — Model A vs Model B
=============================================

Compares two MLP configurations using a strict chronological
70/15/15 train/validation/test split.

Model selection uses VALIDATION metrics only.
Test set is evaluated ONCE for the selected model.

Does NOT modify demand_regressor.pkl.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Allow imports from inventory/src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.demand_model import build_features, FEATURE_COLS, SALES_PATH

SEP = "=" * 56
LINE = "-" * 56


def eval_metrics(y_true, y_pred):
    """Return dict of MAE, RMSE, R2."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def print_metrics(label, m):
    print(f"  {label}:")
    print(f"    MAE  : {m['MAE']:.4f}")
    print(f"    RMSE : {m['RMSE']:.4f}")
    print(f"    R2   : {m['R2']:.4f}")


def compression_stats(y_true, y_pred, label):
    true_std = np.std(y_true)
    pred_std = np.std(y_pred)
    ratio = pred_std / true_std if true_std > 0 else float("nan")
    print(f"  {label}:")
    print(f"    True std       : {true_std:.3f}")
    print(f"    Predicted std  : {pred_std:.3f}")
    print(f"    Compression    : {ratio:.3f}")
    return ratio


def main():
    # ------------------------------------------------------------------
    # Load data and build features
    # ------------------------------------------------------------------
    sales = pd.read_csv(SALES_PATH)
    sales["date"] = pd.to_datetime(sales["date"])
    df = build_features(sales)

    # Sort by date for chronological split
    df = df.sort_values("date").reset_index(drop=True)

    X = df[FEATURE_COLS]
    y = df["future_demand_7d"].values
    dates = df["date"].values

    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    # remainder goes to test

    X_train, y_train = X.iloc[:n_train], y[:n_train]
    X_val, y_val = X.iloc[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val:], y[n_train + n_val:]

    print()
    print(SEP)
    print("  DEMAND MODEL COMPARISON")
    print(SEP)
    print()
    print("Dataset:")
    print(f"  Total feature rows : {n}")
    print(f"  Train              : {len(X_train)}")
    print(f"  Validation         : {len(X_val)}")
    print(f"  Test               : {len(X_test)}")
    print(f"  Train dates        : {pd.Timestamp(dates[0]).strftime('%Y-%m-%d')}"
          f" - {pd.Timestamp(dates[n_train - 1]).strftime('%Y-%m-%d')}")
    print(f"  Val dates          : {pd.Timestamp(dates[n_train]).strftime('%Y-%m-%d')}"
          f" - {pd.Timestamp(dates[n_train + n_val - 1]).strftime('%Y-%m-%d')}")
    print(f"  Test dates         : {pd.Timestamp(dates[n_train + n_val]).strftime('%Y-%m-%d')}"
          f" - {pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}")
    print()

    # ------------------------------------------------------------------
    # Define models
    # ------------------------------------------------------------------
    configs = {
        "A": {
            "label": "Model A: (32,16) + early_stopping=True",
            "mlp": MLPRegressor(
                hidden_layer_sizes=(32, 16),
                max_iter=2000,
                early_stopping=True,
                random_state=42,
            ),
        },
        "B": {
            "label": "Model B: (64,32) + early_stopping=False",
            "mlp": MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=5000,
                alpha=0.0001,
                early_stopping=False,
                random_state=42,
            ),
        },
    }

    results = {}

    for key, cfg in configs.items():
        print(LINE)
        print(f"  {cfg['label']}")
        print(LINE)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", cfg["mlp"]),
        ])
        pipe.fit(X_train, y_train)

        pred_train = pipe.predict(X_train)
        pred_val = pipe.predict(X_val)
        pred_test = pipe.predict(X_test)

        m_train = eval_metrics(y_train, pred_train)
        m_val = eval_metrics(y_val, pred_val)
        m_test = eval_metrics(y_test, pred_test)

        print_metrics("Train", m_train)
        print_metrics("Validation", m_val)
        print_metrics("Test", m_test)
        print()

        compression_stats(y_val, pred_val, "Validation compression")
        compression_stats(y_test, pred_test, "Test compression")
        print()

        results[key] = {
            "pipe": pipe,
            "val": m_val,
            "test": m_test,
            "pred_val": pred_val,
            "pred_test": pred_test,
        }

    # ------------------------------------------------------------------
    # Naive baseline
    # ------------------------------------------------------------------
    print(LINE)
    print("  NAIVE BASELINE (avg_daily_sales_7d * 7)")
    print(LINE)

    naive_val = X_val["avg_daily_sales_7d"].values * 7
    naive_test = X_test["avg_daily_sales_7d"].values * 7

    m_naive_val = eval_metrics(y_val, naive_val)
    m_naive_test = eval_metrics(y_test, naive_test)

    print_metrics("Validation", m_naive_val)
    print_metrics("Test", m_naive_test)
    print()

    # ------------------------------------------------------------------
    # Model selection on VALIDATION only
    # ------------------------------------------------------------------
    print(LINE)
    print("  MODEL SELECTION (based on Validation MAE)")
    print(LINE)

    val_mae_a = results["A"]["val"]["MAE"]
    val_mae_b = results["B"]["val"]["MAE"]
    val_r2_a = results["A"]["val"]["R2"]
    val_r2_b = results["B"]["val"]["R2"]

    print(f"  Model A  -  Val MAE: {val_mae_a:.4f}  Val R2: {val_r2_a:.4f}")
    print(f"  Model B  -  Val MAE: {val_mae_b:.4f}  Val R2: {val_r2_b:.4f}")
    print()

    if val_mae_a < val_mae_b:
        selected = "A"
        reason = "Lower validation MAE"
    elif val_mae_b < val_mae_a:
        selected = "B"
        reason = "Lower validation MAE"
    else:
        # Tie-break on R2
        if val_r2_a >= val_r2_b:
            selected = "A"
            reason = "Tied MAE; higher validation R2"
        else:
            selected = "B"
            reason = "Tied MAE; higher validation R2"

    sel_label = configs[selected]["label"]
    print(f"  Selected : {sel_label}")
    print(f"  Reason   : {reason}")
    print()

    # ------------------------------------------------------------------
    # Final test evaluation — selected model only
    # ------------------------------------------------------------------
    print(LINE)
    print("  FINAL TEST RESULT")
    print(LINE)

    sel_test = results[selected]["test"]
    print(f"  Selected model: {sel_label}")
    print(f"  Test MAE  : {sel_test['MAE']:.4f}")
    print(f"  Test RMSE : {sel_test['RMSE']:.4f}")
    print(f"  Test R2   : {sel_test['R2']:.4f}")
    print()

    # ------------------------------------------------------------------
    # Sanity check — extreme scenarios
    # ------------------------------------------------------------------
    print(LINE)
    print("  SANITY CHECK - EXTREME SCENARIOS")
    print(LINE)

    selected_pipe = results[selected]["pipe"]

    scenarios = [
        ("HIGH DEMAND", {"avg_daily_sales_7d": 10.0, "sales_std_7d": 2.5,
                         "current_stock": 20, "trend_score": 0.8}),
        ("LOW DEMAND",  {"avg_daily_sales_7d": 2.0,  "sales_std_7d": 0.5,
                         "current_stock": 100, "trend_score": 0.0}),
    ]

    for label, inputs in scenarios:
        feat = pd.DataFrame([inputs], columns=FEATURE_COLS)
        pred = selected_pipe.predict(feat)[0]
        naive = inputs["avg_daily_sales_7d"] * 7
        print(f"  {label}:")
        print(f"    MLP prediction     : {pred:.1f} units")
        print(f"    Naive 7-day est.   : {naive:.1f} units")
        print()

    print(SEP)
    print("  COMPARISON COMPLETE")
    print(SEP)
    print()
    print("  NOTE: demand_regressor.pkl was NOT modified.")
    print()


if __name__ == "__main__":
    main()
