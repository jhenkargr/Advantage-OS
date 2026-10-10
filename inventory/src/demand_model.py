"""
Demand Prediction Regressor — AdvantAge OS Inventory ML Pipeline
=================================================================
Model:
    Ridge Regressor with StandardScaler pipeline and L2 regularization.
    Trained on uncensored potential_demand over rolling 7 calendar days.
    Calibrated with Split Conformal Prediction for empirical interval coverage (~80%).

Features (16 leakage-safe features):
    - Units sold lags (1d, 7d)
    - Shifted rolling statistics (3d, 7d, 14d mean; 7d, 14d std)
    - Calendar flags (day_of_week, is_weekend)
    - Inventory dynamics (current_stock, days_of_cover, stockout_frac_7d, days_since_last_stockout)
    - Trend dynamics (trend_score, trend_change_3d)
    - Catalog price
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from features import compute_features, FEATURE_COLS, compute_latest_sku_features

SALES_PATH = "inventory/data/sales_log.csv"
CATALOG_PATH = "inventory/data/product_catalog.csv"
MODEL_SAVE_PATH = "inventory/models/demand_regressor.pkl"


def train_demand_model():
    """Trains the calibrated Ridge demand regression pipeline."""
    sales = pd.read_csv(SALES_PATH)
    catalog = pd.read_csv(CATALOG_PATH)

    df_full = compute_features(sales, catalog)
    df_valid = df_full.dropna(subset=["rolling_mean_14d", "future_demand_7d"]).copy()
    df_valid.sort_values("date", inplace=True)
    df_valid.reset_index(drop=True, inplace=True)

    unique_dates = sorted(df_valid["date"].unique())
    # 14 days test, 14 days val, 7-day purge, rest train
    test_dates = unique_dates[-14:]
    train_val_dates = unique_dates[:-14]
    val_dates = train_val_dates[-14:]
    tr_dates = train_val_dates[:-21]  # 7-day purge gap

    tr_df = df_valid[df_valid["date"].isin(tr_dates)]
    va_df = df_valid[df_valid["date"].isin(val_dates)]
    te_df = df_valid[df_valid["date"].isin(test_dates)]

    # Fit Ridge pipeline
    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0, random_state=42)),
    ])
    reg.fit(tr_df[FEATURE_COLS], tr_df["future_demand_7d"])

    # Conformal calibration on validation fold (80% target coverage)
    va_preds = np.clip(reg.predict(va_df[FEATURE_COLS]), 0, None)
    va_actual = va_df["future_demand_7d"].values
    va_res = np.abs(va_actual - va_preds)
    va_scale = np.sqrt(va_preds + 1.0)
    q_conformal = float(np.quantile(va_res / va_scale, 0.80 * (1.0 + 1.0 / len(va_res))))

    # Single evaluation on untouched test set
    te_preds = np.clip(reg.predict(te_df[FEATURE_COLS]), 0, None)
    te_actual = te_df["future_demand_7d"].values

    mae = float(mean_absolute_error(te_actual, te_preds))
    rmse = float(root_mean_squared_error(te_actual, te_preds))
    wape = float(np.sum(np.abs(te_actual - te_preds)) / np.sum(te_actual))
    r2 = float(r2_score(te_actual, te_preds))

    te_scale = np.sqrt(te_preds + 1.0)
    lower_ci = np.clip(te_preds - q_conformal * te_scale, 0, None)
    upper_ci = te_preds + q_conformal * te_scale
    coverage = float(((te_actual >= lower_ci) & (te_actual <= upper_ci)).mean())
    avg_width = float((upper_ci - lower_ci).mean())

    metrics = {
        "test_mae": round(mae, 3),
        "test_rmse": round(rmse, 3),
        "test_wape": round(wape, 3),
        "test_r2": round(r2, 3),
        "conformal_q": round(q_conformal, 3),
        "conformal_coverage": round(coverage, 3),
        "conformal_avg_width": round(avg_width, 2),
    }

    # Attach metadata to pipeline
    reg.features_ = FEATURE_COLS
    reg.q_conformal_ = q_conformal
    reg.metrics_ = metrics

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(reg, MODEL_SAVE_PATH)
    print(f"Demand Model saved -> {MODEL_SAVE_PATH}")
    print(f"Test MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f} | Coverage: {coverage:.1%}")
    return reg


def predict_future_demand(
    current_stock: float,
    avg_daily_sales_7d: float,
    sales_std_7d: float,
    trend_score: float,
    full_features: pd.DataFrame = None
) -> int:
    """
    Predicts 7-day demand using the trained model.
    Accepts full 16-feature row if available; otherwise falls back to a padded row.
    """
    model = joblib.load(MODEL_SAVE_PATH)
    if full_features is not None:
        raw_pred = model.predict(full_features[FEATURE_COLS])[0]
    else:
        # Fallback padded observation
        dummy = pd.DataFrame([{
            "lag_1": avg_daily_sales_7d,
            "lag_7": avg_daily_sales_7d,
            "rolling_mean_3d": avg_daily_sales_7d,
            "rolling_mean_7d": avg_daily_sales_7d,
            "rolling_mean_14d": avg_daily_sales_7d,
            "rolling_std_7d": sales_std_7d,
            "rolling_std_14d": sales_std_7d,
            "day_of_week": 2,
            "is_weekend": 0,
            "current_stock": current_stock,
            "days_of_cover": min(30.0, current_stock / (avg_daily_sales_7d + 1e-6)),
            "stockout_frac_7d": 0.0,
            "days_since_last_stockout": 30,
            "trend_score": trend_score,
            "trend_change_3d": 0.0,
            "price": 50.0,
            "sku_base_rate": avg_daily_sales_7d,
        }])[FEATURE_COLS]
        raw_pred = model.predict(dummy)[0]

    return max(0, int(round(raw_pred)))


if __name__ == "__main__":
    train_demand_model()
