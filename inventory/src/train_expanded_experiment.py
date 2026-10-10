"""
Expanded-dataset retraining experiment (does NOT overwrite production models).

Saves:
    inventory/models/demand_regressor_expanded.pkl
    inventory/models/stockout_model_expanded.pkl

Writes:
    inventory/model_comparison_expanded.txt
"""

from __future__ import annotations

import hashlib
import os
import sys
import types
from datetime import datetime

import numpy as np
from scipy.special import expit


def _install_wdac_sklearn_stubs():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    def fake(name, attrs=None):
        if name in sys.modules:
            return sys.modules[name]
        mod = types.ModuleType(name)
        for key, value in (attrs or {}).items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    fake("sklearn.utils._openmp_helpers", {
        "_openmp_parallelism_enabled": lambda: False,
        "_openmp_effective_n_threads": lambda n=None: 1,
    })

    class CyLossFunction:
        pass

    class CyHalfBinomialLoss(CyLossFunction):
        # sklearn HalfBinomialLoss: log(1+exp(raw)) - y*raw
        def loss(self, y_true, raw_prediction, sample_weight, loss_out, n_threads=1):
            raw = np.asarray(raw_prediction, dtype=np.float64)
            y = np.asarray(y_true, dtype=np.float64)
            pos = raw >= 0
            sp = np.empty_like(raw)
            sp[pos] = raw[pos] + np.log1p(np.exp(-raw[pos]))
            sp[~pos] = np.log1p(np.exp(raw[~pos]))
            out = sp - y * raw
            if sample_weight is not None:
                out = out * np.asarray(sample_weight, dtype=np.float64)
            loss_out[:] = out
            return loss_out

        def gradient(self, y_true, raw_prediction, sample_weight, gradient_out, n_threads=1):
            raw = np.asarray(raw_prediction, dtype=np.float64)
            y = np.asarray(y_true, dtype=np.float64)
            g = expit(raw) - y
            if sample_weight is not None:
                g = g * np.asarray(sample_weight, dtype=np.float64)
            gradient_out[:] = g
            return gradient_out

        def loss_gradient(
            self, y_true, raw_prediction, sample_weight, loss_out, gradient_out, n_threads=1
        ):
            self.loss(y_true, raw_prediction, sample_weight, loss_out, n_threads)
            self.gradient(y_true, raw_prediction, sample_weight, gradient_out, n_threads)
            return loss_out, gradient_out

        def gradient_hessian(
            self, y_true, raw_prediction, sample_weight, gradient_out, hessian_out, n_threads=1
        ):
            raw = np.asarray(raw_prediction, dtype=np.float64)
            y = np.asarray(y_true, dtype=np.float64)
            p = expit(raw)
            g = p - y
            h = p * (1.0 - p)
            if sample_weight is not None:
                w = np.asarray(sample_weight, dtype=np.float64)
                g = g * w
                h = h * w
            gradient_out[:] = g
            hessian_out[:] = h
            return gradient_out, hessian_out

    class _Dummy(CyLossFunction):
        pass

    fake("sklearn._loss._loss", {
        "CyLossFunction": CyLossFunction,
        "CyAbsoluteError": _Dummy,
        "CyExponentialLoss": _Dummy,
        "CyHalfBinomialLoss": CyHalfBinomialLoss,
        "CyHalfGammaLoss": _Dummy,
        "CyHalfMultinomialLoss": _Dummy,
        "CyHalfPoissonLoss": _Dummy,
        "CyHalfSquaredError": _Dummy,
        "CyHalfTweedieLoss": _Dummy,
        "CyHalfTweedieLossIdentity": _Dummy,
        "CyHuberLoss": _Dummy,
        "CyPinballLoss": _Dummy,
    })
    fake("sklearn.linear_model._sag_fast", {"sag32": None, "sag64": None})
    fake(
        "sklearn.linear_model._sag",
        {"sag_solver": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sag blocked"))},
    )
    fake("sklearn.linear_model._sgd_fast", {
        "EpsilonInsensitive": type("EpsilonInsensitive", (), {}),
        "Hinge": type("Hinge", (), {}),
        "ModifiedHuber": type("ModifiedHuber", (), {}),
        "SquaredEpsilonInsensitive": type("SquaredEpsilonInsensitive", (), {}),
        "SquaredHinge": type("SquaredHinge", (), {}),
        "_plain_sgd32": None,
        "_plain_sgd64": None,
    })


_install_wdac_sklearn_stubs()

import joblib
import pandas as pd
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.demand_model import FEATURE_COLS, build_features as build_demand_features
from src.stockout_model import build_features as build_stockout_features

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SALES_PATH = os.path.join(PROJECT_ROOT, "inventory", "data", "sales_log.csv")
CATALOG_PATH = os.path.join(PROJECT_ROOT, "inventory", "data", "product_catalog.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "inventory", "models")
PROD_DEMAND_PATH = os.path.join(MODELS_DIR, "demand_regressor.pkl")
PROD_STOCKOUT_PATH = os.path.join(MODELS_DIR, "stockout_model.pkl")
EXP_DEMAND_PATH = os.path.join(MODELS_DIR, "demand_regressor_expanded.pkl")
EXP_STOCKOUT_PATH = os.path.join(MODELS_DIR, "stockout_model_expanded.pkl")
REPORT_PATH = os.path.join(PROJECT_ROOT, "inventory", "model_comparison_expanded.txt")

# Previous production metrics (15 SKU / 1,350-row dataset)
OLD_DEMAND = {"MAE": 17.01, "RMSE": 22.45, "R2": 0.4373}
OLD_NAIVE = {"MAE": 21.03, "R2": 0.1083}
OLD_STOCKOUT = {"Precision": 0.5327, "Recall": 0.8143, "F1": 0.6441}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_reg(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def fmt_metrics(m: dict) -> str:
    return f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}"


def chronological_split(df: pd.DataFrame, purge_days: int):
    """Chronological 70/15/15 split with target-horizon gaps. No shuffle."""
    unique_dates = sorted(df["date"].unique())
    n = len(unique_dates)
    retained_n = n - 2 * purge_days
    if purge_days < 0 or retained_n < 3:
        raise ValueError(
            f"Not enough unique dates ({n}) for 70/15/15 split with "
            f"{purge_days}-day purge gaps."
        )

    # Allocate the 70/15/15 proportions across retained dates; reserve a
    # target-horizon-sized gap between folds so labels cannot cross folds.
    n_train = int(retained_n * 0.70)
    n_val = int(retained_n * 0.15)
    n_test = retained_n - n_train - n_val
    if min(n_train, n_val, n_test) == 0:
        raise ValueError(
            f"Not enough retained dates ({retained_n}) for non-empty "
            "train, validation, and test folds."
        )

    train_end = n_train
    val_start = train_end + purge_days
    val_end = val_start + n_val
    test_start = val_end + purge_days
    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[val_start:val_end])
    test_dates = set(unique_dates[test_start:])
    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    bounds = {
        "n_unique_dates": n,
        "purge_days": purge_days,
        "n_purged_dates": 2 * purge_days,
        "train_n": len(train_df),
        "val_n": len(val_df),
        "test_n": len(test_df),
        "train_range": (
            pd.Timestamp(min(train_dates)).strftime("%Y-%m-%d"),
            pd.Timestamp(max(train_dates)).strftime("%Y-%m-%d"),
        ),
        "val_range": (
            pd.Timestamp(min(val_dates)).strftime("%Y-%m-%d"),
            pd.Timestamp(max(val_dates)).strftime("%Y-%m-%d"),
        ),
        "test_range": (
            pd.Timestamp(min(test_dates)).strftime("%Y-%m-%d"),
            pd.Timestamp(max(test_dates)).strftime("%Y-%m-%d"),
        ),
        "n_train_dates": len(train_dates),
        "n_val_dates": len(val_dates),
        "n_test_dates": n_test,
    }
    return train_df, val_df, test_df, bounds


def verify_demand_target_source(sales: pd.DataFrame, sample_df: pd.DataFrame) -> dict:
    """Confirm future_demand_7d is the next 7 days of potential_demand, not units_sold."""
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    sales = sales.sort_values(["product_id", "date"]).reset_index(drop=True)

    mismatches_pd = 0
    mismatches_sold_would_match = 0
    checked = 0
    for _, row in sample_df.head(200).iterrows():
        grp = sales[sales["product_id"] == row["product_id"]].reset_index(drop=True)
        idx = grp.index[grp["date"] == row["date"]]
        if len(idx) != 1:
            continue
        i = int(idx[0])
        future_pd = grp["potential_demand"].iloc[i + 1: i + 8]
        future_sold = grp["units_sold"].iloc[i + 1: i + 8]
        if len(future_pd) < 7:
            continue
        checked += 1
        expected_pd = int(future_pd.sum())
        expected_sold = int(future_sold.sum())
        actual = int(row["future_demand_7d"])
        if actual != expected_pd:
            mismatches_pd += 1
        if actual == expected_sold and expected_sold != expected_pd:
            mismatches_sold_would_match += 1

    censored_rate = float((sales["potential_demand"] > sales["units_sold"]).mean())
    return {
        "checked": checked,
        "mismatches_vs_potential_demand": mismatches_pd,
        "target_is_potential_demand": mismatches_pd == 0 and checked > 0,
        "rows_where_sold_sum_equals_target_but_pd_differs": mismatches_sold_would_match,
        "pct_potential_gt_units_sold": censored_rate,
    }


def main():
    output_paths = [EXP_DEMAND_PATH, EXP_STOCKOUT_PATH, REPORT_PATH]
    existing_outputs = [path for path in output_paths if os.path.exists(path)]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing experiment output(s):\n"
            + "\n".join(existing_outputs)
        )

    prod_hashes_before = {
        "demand": file_sha256(PROD_DEMAND_PATH),
        "stockout": file_sha256(PROD_STOCKOUT_PATH),
    }

    catalog = pd.read_csv(CATALOG_PATH)
    sales = pd.read_csv(SALES_PATH)
    sales["date"] = pd.to_datetime(sales["date"])

    catalog_cols = list(catalog.columns)
    sales_cols = list(sales.columns)
    expected_catalog = [
        "product_id", "product_name", "category",
        "current_stock", "price",
    ]
    expected_sales = [
        "product_id", "date", "potential_demand",
        "units_sold", "stock_after", "trend_score",
    ]

    rows_per_product = sales.groupby("product_id").size()
    censored_pct = float((sales["potential_demand"] > sales["units_sold"]).mean() * 100)

    print("=" * 64)
    print("DATASET SUMMARY")
    print("=" * 64)
    print(f"  Catalog products : {len(catalog)}")
    print(f"  Sales rows       : {len(sales)}")
    print(f"  Unique SKUs      : {sales['product_id'].nunique()}")
    print(f"  Date range       : {sales['date'].min().date()} to {sales['date'].max().date()}")
    print(f"  Unique dates     : {sales['date'].nunique()}")
    print(f"  Rows per product : min={rows_per_product.min()} max={rows_per_product.max()}")
    print(f"  Catalog nulls    : {int(catalog.isna().sum().sum())}")
    print(f"  Sales nulls      : {int(sales.isna().sum().sum())}")
    print(f"  Catalog columns  : {catalog_cols}")
    print(f"  Sales columns    : {sales_cols}")
    print(f"  potential_demand > units_sold : {censored_pct:.2f}%")
    print()

    assert catalog_cols == expected_catalog
    assert sales_cols == expected_sales
    assert len(catalog) == 40
    assert len(sales) == 3600
    assert sales["product_id"].nunique() == 40
    assert sales["date"].nunique() == 90
    assert (rows_per_product == 90).all()
    assert sales.isna().sum().sum() == 0
    assert catalog.isna().sum().sum() == 0
    assert (sales["units_sold"] <= sales["potential_demand"]).all()
    assert (sales["potential_demand"] >= 0).all()
    assert (sales["units_sold"] >= 0).all()
    assert (sales["stock_after"] >= 0).all()

    demand_df = build_demand_features(sales)
    target_check = verify_demand_target_source(sales, demand_df)
    print("TARGET VALIDATION (future_demand_7d)")
    print(f"  Samples checked vs next-7 potential_demand : {target_check['checked']}")
    print(f"  Mismatches                                 : {target_check['mismatches_vs_potential_demand']}")
    print(f"  Target is future potential_demand          : {target_check['target_is_potential_demand']}")
    print()
    if not target_check["target_is_potential_demand"]:
        raise RuntimeError("future_demand_7d is not future potential_demand; aborting.")

    feat_ranges = demand_df[FEATURE_COLS + ["future_demand_7d"]].describe()
    print("FEATURE / TARGET RANGES (demand feature table)")
    print(feat_ranges.to_string())
    print()

    train_d, val_d, test_d, d_bounds = chronological_split(demand_df, purge_days=7)
    print("DEMAND CHRONOLOGICAL SPLIT (70/15/15 retained dates, 7-day purges)")
    print(f"  Usable rows : {len(demand_df)}")
    print(f"  Purged dates between folds: {d_bounds['n_purged_dates']} "
          f"({d_bounds['purge_days']} at each boundary)")
    print(f"  Train : {d_bounds['train_n']} rows | {d_bounds['n_train_dates']} dates | {d_bounds['train_range'][0]} to {d_bounds['train_range'][1]}")
    print(f"  Val   : {d_bounds['val_n']} rows | {d_bounds['n_val_dates']} dates | {d_bounds['val_range'][0]} to {d_bounds['val_range'][1]}")
    print(f"  Test  : {d_bounds['test_n']} rows | {d_bounds['n_test_dates']} dates | {d_bounds['test_range'][0]} to {d_bounds['test_range'][1]}")
    print()

    X_train = train_d[FEATURE_COLS]
    y_train = train_d["future_demand_7d"]
    X_val = val_d[FEATURE_COLS]
    y_val = val_d["future_demand_7d"]
    X_test = test_d[FEATURE_COLS]
    y_test = test_d["future_demand_7d"]

    configs = {
        "A": {
            "label": "Model A: MLPRegressor hidden=(32,16), early_stopping=True, max_iter=2000",
            "mlp_kwargs": dict(
                hidden_layer_sizes=(32, 16),
                max_iter=2000,
                early_stopping=True,
                random_state=42,
            ),
        },
        "B": {
            "label": "Model B: MLPRegressor hidden=(64,32), early_stopping=False, max_iter=5000, alpha=0.0001",
            "mlp_kwargs": dict(
                hidden_layer_sizes=(64, 32),
                max_iter=5000,
                alpha=0.0001,
                early_stopping=False,
                random_state=42,
            ),
        },
    }

    val_results = {}
    fitted = {}
    for key, cfg in configs.items():
        print("-" * 64)
        print(cfg["label"])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(**cfg["mlp_kwargs"])),
        ])
        pipe.fit(X_train, y_train)
        pred_val = pipe.predict(X_val)
        m_val = eval_reg(y_val, pred_val)
        print(f"  Validation: {fmt_metrics(m_val)}")
        val_results[key] = m_val
        fitted[key] = pipe

    mae_a, mae_b = val_results["A"]["MAE"], val_results["B"]["MAE"]
    r2_a, r2_b = val_results["A"]["R2"], val_results["B"]["R2"]
    if mae_a < mae_b:
        selected, reason = "A", "Lowest validation MAE"
    elif mae_b < mae_a:
        selected, reason = "B", "Lowest validation MAE"
    elif r2_a >= r2_b:
        selected, reason = "A", "Tied validation MAE; higher validation R2"
    else:
        selected, reason = "B", "Tied validation MAE; higher validation R2"

    print("-" * 64)
    print("MODEL SELECTION (validation only; test not used)")
    print(f"  Selected: {configs[selected]['label']}")
    print(f"  Reason  : {reason}")
    print()

    selected_pipe = fitted[selected]
    pred_test = selected_pipe.predict(X_test)
    test_metrics = eval_reg(y_test, pred_test)
    naive_test = eval_reg(y_test, X_test["avg_daily_sales_7d"].values * 7)
    naive_val = eval_reg(y_val, X_val["avg_daily_sales_7d"].values * 7)

    print("FINAL TEST (selected demand model only)")
    print(f"  {fmt_metrics(test_metrics)}")
    print("NAIVE BASELINE TEST (avg_daily_sales_7d * 7)")
    print(f"  {fmt_metrics(naive_test)}")
    print()

    # In-distribution sanity check from TRAIN feature ranges
    q = X_train.quantile([0.10, 0.50, 0.90])
    high_inputs = {
        "avg_daily_sales_7d": float(q.loc[0.90, "avg_daily_sales_7d"]),
        "sales_std_7d": float(q.loc[0.50, "sales_std_7d"]),
        "current_stock": float(q.loc[0.50, "current_stock"]),
        "trend_score": float(q.loc[0.90, "trend_score"]),
    }
    low_inputs = {
        "avg_daily_sales_7d": float(q.loc[0.10, "avg_daily_sales_7d"]),
        "sales_std_7d": float(q.loc[0.50, "sales_std_7d"]),
        "current_stock": float(q.loc[0.50, "current_stock"]),
        "trend_score": float(q.loc[0.10, "trend_score"]),
    }
    train_min = X_train.min()
    train_max = X_train.max()
    for name, inputs in [("HIGH", high_inputs), ("LOW", low_inputs)]:
        for col, val in inputs.items():
            if not (train_min[col] <= val <= train_max[col]):
                raise RuntimeError(f"{name} sanity input {col}={val} outside train range")

    high_pred = float(selected_pipe.predict(pd.DataFrame([high_inputs], columns=FEATURE_COLS))[0])
    low_pred = float(selected_pipe.predict(pd.DataFrame([low_inputs], columns=FEATURE_COLS))[0])
    high_naive = high_inputs["avg_daily_sales_7d"] * 7
    low_naive = low_inputs["avg_daily_sales_7d"] * 7
    high_gt_low = high_pred > low_pred

    print("SANITY CHECK (in-distribution, from training quantiles)")
    print(f"  HIGH inputs: {high_inputs}")
    print(f"  HIGH pred={high_pred:.2f}  naive={high_naive:.2f}")
    print(f"  LOW  inputs: {low_inputs}")
    print(f"  LOW  pred={low_pred:.2f}  naive={low_naive:.2f}")
    print(f"  High > Low : {high_gt_low}")
    print()

    # Stockout experiment
    stock_df = build_stockout_features(sales)
    train_s, val_s, test_s, s_bounds = chronological_split(stock_df, purge_days=3)
    print("STOCKOUT CHRONOLOGICAL SPLIT (70/15/15 retained dates, 3-day purges)")
    print(f"  Usable rows : {len(stock_df)}")
    print(f"  Purged dates between folds: {s_bounds['n_purged_dates']} "
          f"({s_bounds['purge_days']} at each boundary)")
    print(f"  Train : {s_bounds['train_n']} | {s_bounds['train_range'][0]} to {s_bounds['train_range'][1]}")
    print(f"  Val   : {s_bounds['val_n']} | {s_bounds['val_range'][0]} to {s_bounds['val_range'][1]}")
    print(f"  Test  : {s_bounds['test_n']} | {s_bounds['test_range'][0]} to {s_bounds['test_range'][1]}")
    print(f"  Class balance (all usable):")
    print(stock_df["stockout_within_3d"].value_counts().to_string())
    print()

    Xs_train = train_s[FEATURE_COLS]
    ys_train = train_s["stockout_within_3d"]
    Xs_val = val_s[FEATURE_COLS]
    ys_val = val_s["stockout_within_3d"]
    Xs_test = test_s[FEATURE_COLS]
    ys_test = test_s["stockout_within_3d"]

    stock_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    stock_clf.fit(Xs_train, ys_train)

    def class_metrics(y_true, y_pred):
        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
            "Confusion": confusion_matrix(y_true, y_pred).tolist(),
            "Report": classification_report(y_true, y_pred, zero_division=0),
        }

    pred_s_val = stock_clf.predict(Xs_val)
    pred_s_test = stock_clf.predict(Xs_test)
    stock_val = class_metrics(ys_val, pred_s_val)
    stock_test = class_metrics(ys_test, pred_s_test)

    print("STOCKOUT VALIDATION")
    print(f"  Acc={stock_val['Accuracy']:.4f} Prec={stock_val['Precision']:.4f} "
          f"Rec={stock_val['Recall']:.4f} F1={stock_val['F1']:.4f}")
    print("STOCKOUT TEST")
    print(f"  Acc={stock_test['Accuracy']:.4f} Prec={stock_test['Precision']:.4f} "
          f"Rec={stock_test['Recall']:.4f} F1={stock_test['F1']:.4f}")
    print("  Confusion matrix (test) [tn,fp; fn,tp]:", stock_test["Confusion"])
    print()

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(selected_pipe, EXP_DEMAND_PATH)
    joblib.dump(stock_clf, EXP_STOCKOUT_PATH)
    print(f"Saved {EXP_DEMAND_PATH}")
    print(f"Saved {EXP_STOCKOUT_PATH}")

    prod_hashes_after = {
        "demand": file_sha256(PROD_DEMAND_PATH),
        "stockout": file_sha256(PROD_STOCKOUT_PATH),
    }
    prod_unchanged = prod_hashes_before == prod_hashes_after
    if not prod_unchanged:
        raise RuntimeError("Production model files changed unexpectedly.")

    improved_demand_mae = test_metrics["MAE"] < OLD_DEMAND["MAE"]
    improved_demand_r2 = test_metrics["R2"] > OLD_DEMAND["R2"]
    improved_stock_recall = stock_test["Recall"] > OLD_STOCKOUT["Recall"]
    beats_naive = test_metrics["MAE"] < naive_test["MAE"]

    report_lines = [
        "AdvantAge OS — Inventory + Demand Prediction",
        "Expanded-dataset retraining experiment",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "NOTE: Production models were NOT overwritten.",
        "  Baseline: inventory/models/demand_regressor.pkl",
        "  Baseline: inventory/models/stockout_model.pkl",
        "  Experiment: inventory/models/demand_regressor_expanded.pkl",
        "  Experiment: inventory/models/stockout_model_expanded.pkl",
        "",
        "=" * 72,
        "1. DATASET SUMMARY",
        "=" * 72,
        f"  Products (catalog)     : {len(catalog)}",
        f"  Unique SKUs (sales)    : {sales['product_id'].nunique()}",
        f"  Sales-log rows         : {len(sales)}",
        f"  Unique dates           : {sales['date'].nunique()}",
        f"  Date range             : {sales['date'].min().date()} to {sales['date'].max().date()}",
        f"  Rows per product       : {int(rows_per_product.min())} (min) / {int(rows_per_product.max())} (max)",
        f"  potential_demand > units_sold : {censored_pct:.2f}% of rows (stock-censored sales still present)",
        "",
        "=" * 72,
        "2. DATA VALIDATION",
        "=" * 72,
        f"  Catalog columns        : {catalog_cols}",
        f"  Sales columns          : {sales_cols}",
        f"  Catalog schema match   : {catalog_cols == expected_catalog}",
        f"  Sales schema match     : {sales_cols == expected_sales}",
        f"  Catalog nulls          : {int(catalog.isna().sum().sum())}",
        f"  Sales nulls            : {int(sales.isna().sum().sum())}",
        f"  units_sold <= potential_demand : {(sales['units_sold'] <= sales['potential_demand']).all()}",
        f"  Non-negative demand/sold/stock : "
        f"{bool((sales['potential_demand'] >= 0).all() and (sales['units_sold'] >= 0).all() and (sales['stock_after'] >= 0).all())}",
        f"  Target future_demand_7d uses FUTURE potential_demand (not units_sold): "
        f"{target_check['target_is_potential_demand']} ({target_check['checked']} samples checked)",
        f"  Feature columns        : {FEATURE_COLS}",
        f"  No extra features added.",
        "",
        "  Demand feature-table ranges:",
        feat_ranges.to_string(),
        "",
        "  Broader-range note vs previous 15-SKU dataset:",
        "  The previous raw sales_log is no longer available for a direct numeric",
        "  range comparison. Prior diagnostics treated avg_daily_sales_7d=10.0,",
        "  sales_std_7d=2.5, current_stock=20, trend_score=0.8 as in-distribution.",
        f"  On the expanded training split, avg_daily_sales_7d range is "
        f"{X_train['avg_daily_sales_7d'].min():.3f} to {X_train['avg_daily_sales_7d'].max():.3f};",
        f"  sales_std_7d {X_train['sales_std_7d'].min():.3f} to {X_train['sales_std_7d'].max():.3f};",
        f"  current_stock {X_train['current_stock'].min():.3f} to {X_train['current_stock'].max():.3f};",
        f"  trend_score {X_train['trend_score'].min():.3f} to {X_train['trend_score'].max():.3f}.",
        "  trend_score remains {0.0, 0.8} because generation logic was unchanged.",
        "",
        "=" * 72,
        "3. DEMAND MODEL CANDIDATE RESULTS (VALIDATION ONLY)",
        "=" * 72,
        f"  Split: chronological unique dates, 70/15/15 of retained dates, "
        f"{d_bounds['purge_days']}-day purge at each fold boundary (no shuffle).",
        f"  Train dates : {d_bounds['n_train_dates']} ({d_bounds['train_range'][0]} to {d_bounds['train_range'][1]}), rows={d_bounds['train_n']}",
        f"  Val dates   : {d_bounds['n_val_dates']} ({d_bounds['val_range'][0]} to {d_bounds['val_range'][1]}), rows={d_bounds['val_n']}",
        f"  Test dates  : {d_bounds['n_test_dates']} ({d_bounds['test_range'][0]} to {d_bounds['test_range'][1]}), rows={d_bounds['test_n']}",
        f"  Purged dates between folds: {d_bounds['n_purged_dates']} "
        f"({d_bounds['purge_days']} at each boundary)",
        f"  Usable demand samples after rolling/horizon drop: {len(demand_df)}",
        "",
        "  Candidate                         MAE       RMSE        R2",
        "  ------------------------------- -------- ---------- ----------",
        f"  Model A (32,16) early_stop=True  {val_results['A']['MAE']:8.4f} {val_results['A']['RMSE']:10.4f} {val_results['A']['R2']:10.4f}",
        f"  Model B (64,32) early_stop=False {val_results['B']['MAE']:8.4f} {val_results['B']['RMSE']:10.4f} {val_results['B']['R2']:10.4f}",
        f"  Naive (avg*7)                    {naive_val['MAE']:8.4f} {naive_val['RMSE']:10.4f} {naive_val['R2']:10.4f}",
        "",
        "  Test metrics were NOT computed for both candidates and were NOT used",
        "  for selection.",
        "",
        "=" * 72,
        "4. SELECTED DEMAND MODEL",
        "=" * 72,
        f"  Selected : {configs[selected]['label']}",
        f"  Reason   : {reason}",
        f"  Val MAE  : {val_results[selected]['MAE']:.4f}",
        f"  Val RMSE : {val_results[selected]['RMSE']:.4f}",
        f"  Val R2   : {val_results[selected]['R2']:.4f}",
        "",
        "=" * 72,
        "5. FINAL DEMAND TEST METRICS (selected model, one evaluation)",
        "=" * 72,
        f"  Test MAE  : {test_metrics['MAE']:.4f}",
        f"  Test RMSE : {test_metrics['RMSE']:.4f}",
        f"  Test R2   : {test_metrics['R2']:.4f}",
        "",
        "=" * 72,
        "6. NAIVE BASELINE METRICS (avg_daily_sales_7d * 7)",
        "=" * 72,
        f"  Val MAE   : {naive_val['MAE']:.4f}",
        f"  Val RMSE  : {naive_val['RMSE']:.4f}",
        f"  Val R2    : {naive_val['R2']:.4f}",
        f"  Test MAE  : {naive_test['MAE']:.4f}",
        f"  Test RMSE : {naive_test['RMSE']:.4f}",
        f"  Test R2   : {naive_test['R2']:.4f}",
        f"  Selected ML beats naive on test MAE : {beats_naive}",
        "",
        "=" * 72,
        "7. STOCKOUT MODEL METRICS",
        "=" * 72,
        "  Model: LogisticRegression + StandardScaler, class_weight=balanced,",
        "  default 0.5 threshold, same 4 features, target=stockout_within_3d.",
        f"  Train dates : {s_bounds['n_train_dates']} ({s_bounds['train_range'][0]} to {s_bounds['train_range'][1]}), rows={s_bounds['train_n']}",
        f"  Val dates   : {s_bounds['n_val_dates']} ({s_bounds['val_range'][0]} to {s_bounds['val_range'][1]}), rows={s_bounds['val_n']}",
        f"  Test dates  : {s_bounds['n_test_dates']} ({s_bounds['test_range'][0]} to {s_bounds['test_range'][1]}), rows={s_bounds['test_n']}",
        f"  Purged dates between folds: {s_bounds['n_purged_dates']} "
        f"({s_bounds['purge_days']} at each boundary)",
        "",
        "  Split        Accuracy  Precision    Recall        F1",
        "  ---------- ---------- ---------- ---------- ----------",
        f"  Validation {stock_val['Accuracy']:10.4f} {stock_val['Precision']:10.4f} {stock_val['Recall']:10.4f} {stock_val['F1']:10.4f}",
        f"  Test       {stock_test['Accuracy']:10.4f} {stock_test['Precision']:10.4f} {stock_test['Recall']:10.4f} {stock_test['F1']:10.4f}",
        "",
        f"  Test confusion matrix [[TN, FP], [FN, TP]]: {stock_test['Confusion']}",
        "",
        "  Test classification report:",
        stock_test["Report"],
        "",
        "=" * 72,
        "8. SANITY-CHECK RESULTS",
        "=" * 72,
        "  Inputs taken from training-split 10th/50th/90th percentiles",
        "  (inside observed training ranges).",
        "",
        f"  HIGH DEMAND inputs: {high_inputs}",
        f"  HIGH prediction   : {high_pred:.2f}",
        f"  HIGH naive (avg*7): {high_naive:.2f}",
        "",
        f"  LOW DEMAND inputs : {low_inputs}",
        f"  LOW prediction    : {low_pred:.2f}",
        f"  LOW naive (avg*7) : {low_naive:.2f}",
        "",
        f"  High > Low        : {high_gt_low}",
        "",
        "=" * 72,
        "9. COMPARISON WITH OLD PRODUCTION METRICS",
        "=" * 72,
        "  Old numbers come from the 15-SKU / 1,350-row experiment",
        "  (approximately 80/20 chronological for production scripts).",
        "  New numbers use 40 SKU / 3,600 rows and 70/15/15 unique-date split.",
        "  These are NOT a perfectly matched A/B of the same holdout.",
        "",
        "  Demand (test)",
        "  Source                         MAE       RMSE        R2",
        "  ----------------------- ---------- ---------- ----------",
        f"  Old production (approx) {OLD_DEMAND['MAE']:10.2f} {OLD_DEMAND['RMSE']:10.2f} {OLD_DEMAND['R2']:10.4f}",
        f"  New selected (this run) {test_metrics['MAE']:10.4f} {test_metrics['RMSE']:10.4f} {test_metrics['R2']:10.4f}",
        f"  Old naive (approx)      {OLD_NAIVE['MAE']:10.2f} {'n/a':>10} {OLD_NAIVE['R2']:10.4f}",
        f"  New naive (this run)    {naive_test['MAE']:10.4f} {naive_test['RMSE']:10.4f} {naive_test['R2']:10.4f}",
        "",
        "  Stockout class 1 (test)",
        "  Source                    Precision     Recall         F1",
        "  ----------------------- ---------- ---------- ----------",
        f"  Old production (approx) {OLD_STOCKOUT['Precision']:10.4f} {OLD_STOCKOUT['Recall']:10.4f} {OLD_STOCKOUT['F1']:10.4f}",
        f"  New (this run)          {stock_test['Precision']:10.4f} {stock_test['Recall']:10.4f} {stock_test['F1']:10.4f}",
        "",
        "=" * 72,
        "10. DID THE EXPANDED DATASET IMPROVE THE MODELS?",
        "=" * 72,
        f"  Demand test MAE lower than old reported MAE : {improved_demand_mae}",
        f"  Demand test R2  higher than old reported R2 : {improved_demand_r2}",
        f"  Demand selected model beats new naive MAE   : {beats_naive}",
        f"  Stockout test recall higher than old recall : {improved_stock_recall}",
        "",
        "  Interpretation must be cautious: different SKU mix, different split",
        "  ratios, and a later date window mean metric movement is not a pure",
        "  sample-size effect. Production files were left unchanged pending review.",
        "",
        "=" * 72,
        "11. LIMITATIONS",
        "=" * 72,
        "  - Data remain synthetic (Poisson + weekday/trend factors).",
        "  - Features still use units_sold (stock-censored) while the demand",
        "    target uses potential_demand; this mismatch is intentional but",
        "    caps how well recent sales can explain unconstrained demand.",
        "  - trend_score is still a binary-like boost indicator (0.0 or 0.8).",
        "  - No product identity/category features (by design).",
        "  - 90-day window is short; 7-day horizon plus 7-day history drop many rows.",
        "  - MLP training is stochastic aside from random_state; early_stopping",
        "    uses an internal validation split of the training fold.",
        "  - Old vs new metric comparison is across different datasets/splits.",
        "  - Orchestrator still points at production .pkl files, not these artifacts.",
        "",
        f"  Production SHA256 unchanged: {prod_unchanged}",
        f"    demand_regressor.pkl  : {prod_hashes_after['demand']}",
        f"    stockout_model.pkl    : {prod_hashes_after['stockout']}",
        "",
        "END OF REPORT",
        "",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Report written -> {REPORT_PATH}")
    print()

    print("=" * 64)
    print("DATASET")
    print("-------")
    print("Products: 40")
    print("Rows: 3600")
    print("Days: 90")
    print()
    print("DEMAND MODEL")
    print("------------")
    print(f"Selected model: {configs[selected]['label']}")
    print(f"Validation MAE: {val_results[selected]['MAE']:.4f}")
    print(f"Validation R²: {val_results[selected]['R2']:.4f}")
    print(f"Test MAE: {test_metrics['MAE']:.4f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"Test R²: {test_metrics['R2']:.4f}")
    print()
    print("NAIVE BASELINE")
    print("--------------")
    print(f"Test MAE: {naive_test['MAE']:.4f}")
    print(f"Test RMSE: {naive_test['RMSE']:.4f}")
    print(f"Test R²: {naive_test['R2']:.4f}")
    print()
    print("STOCKOUT MODEL")
    print("--------------")
    print(f"Accuracy: {stock_test['Accuracy']:.4f}")
    print(f"Precision: {stock_test['Precision']:.4f}")
    print(f"Recall: {stock_test['Recall']:.4f}")
    print(f"F1: {stock_test['F1']:.4f}")
    print()
    print("SANITY CHECK")
    print("------------")
    print(f"High-demand prediction: {high_pred:.2f}")
    print(f"Low-demand prediction: {low_pred:.2f}")
    print(f"High > Low: {'YES' if high_gt_low else 'NO'}")
    print()
    print("MODEL FILES")
    print("-----------")
    print("demand_regressor_expanded.pkl")
    print("stockout_model_expanded.pkl")
    print()
    print("Production models unchanged:")
    print("YES" if prod_unchanged else "NO")
    print("=" * 64)


if __name__ == "__main__":
    main()
