"""
Unit Test: Data Leakage Assertion
AdvantAge OS Inventory ML Module
=================================
Asserts that no feature at row (sku, t) changes when future rows (t+1, t+2, ...)
are altered or corrupted with random noise.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from features import compute_features, FEATURE_COLS


def test_no_future_leakage():
    sales = pd.read_csv("inventory/data/sales_log.csv")
    catalog = pd.read_csv("inventory/data/product_catalog.csv")
    
    # Pick 3 arbitrary SKUs
    test_skus = ["SKU001", "SKU015", "SKU040"]
    
    for pid in test_skus:
        sku_orig = sales[sales["product_id"] == pid].sort_values("date").reset_index(drop=True)
        cat_sub = catalog[catalog["product_id"] == pid]
        
        feat_orig = compute_features(sku_orig, cat_sub)
        
        # Test cutoff at day 35
        cutoff = 35
        sku_altered = sku_orig.copy()
        n_future = len(sku_altered) - (cutoff + 1)
        
        # Severely corrupt future rows
        np.random.seed(999)
        sku_altered.loc[cutoff + 1:, "units_sold"] = np.random.randint(200, 500, size=n_future)
        sku_altered.loc[cutoff + 1:, "stock_after"] = np.random.randint(1000, 3000, size=n_future)
        sku_altered.loc[cutoff + 1:, "trend_score"] = 9.99
        if "potential_demand" in sku_altered.columns:
            sku_altered.loc[cutoff + 1:, "potential_demand"] = np.random.randint(500, 1000, size=n_future)
            
        feat_altered = compute_features(sku_altered, cat_sub)
        
        # Check all features at the cutoff day row
        row_orig = feat_orig.iloc[cutoff][FEATURE_COLS]
        row_alt = feat_altered.iloc[cutoff][FEATURE_COLS]
        
        for col in FEATURE_COLS:
            v_orig = row_orig[col]
            v_alt = row_alt[col]
            assert np.isclose(v_orig, v_alt, atol=1e-7, equal_nan=True), (
                f"LEAKAGE DETECTED in {pid} at row {cutoff} for feature '{col}': "
                f"original={v_orig}, altered={v_alt}"
            )
            
    print("ALL LEAKAGE ASSERTIONS PASSED: No feature uses future data.")


def test_no_zero_variance():
    sales = pd.read_csv("inventory/data/sales_log.csv")
    catalog = pd.read_csv("inventory/data/product_catalog.csv")
    df = compute_features(sales, catalog)
    df_valid = df.dropna(subset=["rolling_mean_14d"])
    for col in FEATURE_COLS:
        std_val = float(df_valid[col].std())
        assert std_val > 1e-4, f"Feature '{col}' has zero or near-zero variance! (std={std_val})"
    print(f"ALL VARIANCE ASSERTIONS PASSED: All {len(FEATURE_COLS)} features have non-zero variance.")


if __name__ == "__main__":
    test_no_future_leakage()
    test_no_zero_variance()
