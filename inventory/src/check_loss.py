import os
import sys
from pathlib import Path
from datetime import timedelta

os.environ["OMP_NUM_THREADS"] = "1"

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
PROJ_ROOT = ROOT_DIR.parent

sys.path.insert(0, str(CUR_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

from features import compute_features, FEATURE_COLS

SALES_PATH = ROOT_DIR / "data" / "sales_log.csv"
CATALOG_PATH = ROOT_DIR / "data" / "product_catalog.csv"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def pinball_loss(y_true, y_pred, q=0.74):
    diff = y_true - y_pred
    return np.mean(np.maximum(q * diff, (q - 1.0) * diff))


def business_cost_loss(y_true, y_pred, c_under=2.00, c_over=0.70):
    diff = y_true - y_pred
    under = np.maximum(0.0, diff)
    over = np.maximum(0.0, -diff)
    return np.mean(c_under * under + c_over * over)


def main():
    print("=" * 80)
    print(" ADVANTAGE OS - LOSS FUNCTION DIAGNOSTICS & VERIFICATION")
    print("=" * 80)

    catalog = pd.read_csv(CATALOG_PATH)
    sales = pd.read_csv(SALES_PATH)
    sales["date"] = pd.to_datetime(sales["date"])

    np.random.seed(42)
    start_date = sales["date"].min()
    dgp_records = []
    for _, row in catalog.iterrows():
        pid = row["product_id"]
        bd = np.random.uniform(2, 15)
        tbd = np.random.randint(30, 70)
        stock = int(row["current_stock"]) + 50
        for day in range(90):
            date = start_date + timedelta(days=day)
            wf = 1.3 if date.weekday() >= 5 else 1.0
            tf = 1.8 if tbd <= day <= tbd + 10 else 1.0
            exp_daily = bd * wf * tf
            pd_val = int(np.random.poisson(exp_daily))
            daily_sales = min(pd_val, stock)
            stock -= daily_sales
            if stock < 10 and np.random.random() < 0.3:
                stock += np.random.randint(40, 80)
            dgp_records.append({
                "product_id": pid,
                "date": date,
                "expected_daily_demand": exp_daily,
            })
    dgp_df = pd.DataFrame(dgp_records)
    sales = sales.merge(dgp_df, on=["product_id", "date"], how="left")

    df_full = compute_features(sales, catalog)
    df_valid = df_full.dropna(subset=["rolling_mean_14d", "future_demand_7d", "stockout_within_3d"]).copy()
    df_valid.sort_values("date", inplace=True)
    df_valid.reset_index(drop=True, inplace=True)

    unique_dates = sorted(df_valid["date"].unique())
    # Exclude test window (last 14 days)
    dev_dates = unique_dates[:-14]
    dev_df = df_valid[df_valid["date"].isin(dev_dates)].copy()

    # 5-fold purged TimeSeriesSplit with gap = 280 (7 days x 40 SKUs)
    tscv = TimeSeriesSplit(n_splits=5, gap=280)

    # -------------------------------------------------------------------------
    # 1. RIDGE (ALPHA=10) PER-FOLD METRICS
    # -------------------------------------------------------------------------
    print("\n--- RIDGE (alpha=10) PER-FOLD STATS ---")
    ridge_pipe = Pipeline([
        ("s", StandardScaler()),
        ("m", Ridge(alpha=10.0, random_state=42))
    ])

    fold_tr_mse, fold_tr_mae = [], []
    fold_va_mse, fold_va_mae = [], []
    fold_va_bias = []

    # Losses for the same Ridge predictions
    loss_mse, loss_mae, loss_pinball, loss_biz = [], [], [] ,[]

    for k, (tr_i, va_i) in enumerate(tscv.split(dev_df)):
        X_tr = dev_df[FEATURE_COLS].iloc[tr_i]
        y_tr = dev_df["future_demand_7d"].iloc[tr_i]
        X_va = dev_df[FEATURE_COLS].iloc[va_i]
        y_va = dev_df["future_demand_7d"].iloc[va_i]

        ridge_pipe.fit(X_tr, y_tr)
        p_tr = np.clip(ridge_pipe.predict(X_tr), 0, None)
        p_va = np.clip(ridge_pipe.predict(X_va), 0, None)

        tr_mse = mean_squared_error(y_tr, p_tr)
        tr_mae = mean_absolute_error(y_tr, p_tr)
        va_mse = mean_squared_error(y_va, p_va)
        va_mae = mean_absolute_error(y_va, p_va)
        va_bias = float(np.mean(y_va) - np.mean(p_va))

        fold_tr_mse.append(tr_mse)
        fold_tr_mae.append(tr_mae)
        fold_va_mse.append(va_mse)
        fold_va_mae.append(va_mae)
        fold_va_bias.append(va_bias)

        pb = pinball_loss(y_va.values, p_va, q=0.74)
        biz = business_cost_loss(y_va.values, p_va, c_under=2.00, c_over=0.70)

        loss_mse.append(va_mse)
        loss_mae.append(va_mae)
        loss_pinball.append(pb)
        loss_biz.append(biz)

        print(f"Fold {k+1}: Train MSE={tr_mse:6.2f}, Train MAE={tr_mae:5.2f} | Val MSE={va_mse:6.2f}, Val MAE={va_mae:5.2f} | Val Bias={va_bias:+6.2f}")

    print("\nRidge Overall 5-Fold Averages:")
    print(f"  Train MSE: {np.mean(fold_tr_mse):.2f} +/- {np.std(fold_tr_mse):.2f}")
    print(f"  Train MAE: {np.mean(fold_tr_mae):.2f} +/- {np.std(fold_tr_mae):.2f}")
    print(f"  Val MSE  : {np.mean(fold_va_mse):.2f} +/- {np.std(fold_va_mse):.2f}")
    print(f"  Val MAE  : {np.mean(fold_va_mae):.2f} +/- {np.std(fold_va_mae):.2f}")
    print(f"  Val Bias : {np.mean(fold_va_bias):+.2f} +/- {np.std(fold_va_bias):.2f}")

    print("\n--- RIDGE PREDICTIONS JUDGED ON FOUR EVALUATION LOSSES ---")
    print(f"  1. MSE                 : {np.mean(loss_mse):.2f} +/- {np.std(loss_mse):.2f}")
    print(f"  2. MAE                 : {np.mean(loss_mae):.2f} +/- {np.std(loss_mae):.2f}")
    print(f"  3. Pinball (q=0.74)    : {np.mean(loss_pinball):.2f} +/- {np.std(loss_pinball):.2f}")
    print(f"  4. Business Cost ($)   : {np.mean(loss_biz):.2f} +/- {np.std(loss_biz):.2f}")

    # -------------------------------------------------------------------------
    # 2. MLP (64, 32) PER-FOLD CONVERGENCE & LOSS CURVE PLOT
    # -------------------------------------------------------------------------
    print("\n--- MLP (64, 32) CONVERGENCE & FINAL TRAINING LOSS PER FOLD ---")
    plt.figure(figsize=(8, 4.5))
    mlp_final_losses = []
    mlp_iters = []

    for k, (tr_i, va_i) in enumerate(tscv.split(dev_df)):
        X_tr = dev_df[FEATURE_COLS].iloc[tr_i]
        y_tr = dev_df["future_demand_7d"].iloc[tr_i]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)

        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            alpha=1.0,
            max_iter=2000,
            random_state=42 + k
        )
        mlp.fit(X_tr_s, y_tr)

        mlp_final_losses.append(mlp.loss_)
        mlp_iters.append(mlp.n_iter_)

        plt.plot(mlp.loss_curve_, label=f"Fold {k+1} (final={mlp.loss_:.2f}, n_iter={mlp.n_iter_})")
        print(f"Fold {k+1}: Final Loss={mlp.loss_:.4f}, Iterations={mlp.n_iter_}")

    plt.xlabel("Iteration")
    plt.ylabel("Training Loss (MSE)")
    plt.title("MLP (64, 32) Training Loss Curves across 5 Purged CV Folds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = FIGURES_DIR / "loss_check_mlp.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved MLP loss curve plot to {plot_path}")


if __name__ == "__main__":
    main()
