"""
Unified, Leakage-Free Feature Engineering Pipeline
AdvantAge OS Inventory ML Module
=================================================
Ensures identical feature definitions between training and online inference.
Every feature at row (sku, t) uses only data strictly up to and including day t.
Rolling windows on units_sold use shift(1) to avoid lookahead.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "rolling_mean_3d",
    "rolling_mean_7d",
    "rolling_mean_14d",
    "rolling_std_7d",
    "rolling_std_14d",
    "day_of_week",
    "is_weekend",
    "current_stock",
    "days_of_cover",
    "stockout_frac_7d",
    "days_since_last_stockout",
    "trend_score",
    "trend_change_3d",
    "price",
    "sku_base_rate",
]


def compute_features(sales_df: pd.DataFrame, catalog_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Computes leakage-safe features for each SKU-day row.
    
    Parameters
    ----------
    sales_df : pd.DataFrame
        Columns: ['product_id', 'date', 'units_sold', 'stock_after', 'trend_score', optional 'potential_demand']
    catalog_df : pd.DataFrame, optional
        Columns: ['product_id', 'price', 'category', ...]
        
    Returns
    -------
    pd.DataFrame with FEATURE_COLS + target columns if potential_demand is present.
    """
    sales = sales_df.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    sales.sort_values(["product_id", "date"], inplace=True)
    
    if catalog_df is not None:
        cat_dict = catalog_df.set_index("product_id").to_dict("index")
    else:
        cat_dict = {}

    frames = []
    has_pd = "potential_demand" in sales.columns

    for pid, grp in sales.groupby("product_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        
        # 1. Lags of units_sold
        grp["lag_1"] = grp["units_sold"].shift(1)
        grp["lag_7"] = grp["units_sold"].shift(7)
        
        # 2. Shifted rolling statistics (strictly backward-looking, shifted by 1)
        s1 = grp["units_sold"].shift(1)
        grp["rolling_mean_3d"] = s1.rolling(3, min_periods=3).mean()
        grp["rolling_mean_7d"] = s1.rolling(7, min_periods=7).mean()
        grp["rolling_mean_14d"] = s1.rolling(14, min_periods=14).mean()
        grp["rolling_std_7d"] = s1.rolling(7, min_periods=7).std().fillna(0.0)
        grp["rolling_std_14d"] = s1.rolling(14, min_periods=14).std().fillna(0.0)
        
        # 3. Calendar features
        grp["day_of_week"] = grp["date"].dt.dayofweek
        grp["is_weekend"] = (grp["day_of_week"] >= 5).astype(int)
        
        # 4. Inventory dynamics
        grp["current_stock"] = grp["stock_after"].astype(float)
        # Clipped to [0, 30] to prevent extreme/inf values when rolling_mean_7d is near 0
        raw_doc = grp["current_stock"] / (grp["rolling_mean_7d"] + 1e-6)
        grp["days_of_cover"] = np.clip(raw_doc.fillna(30.0), 0.0, 30.0)
        
        is_so = (grp["stock_after"] <= 0).astype(int)
        grp["stockout_frac_7d"] = is_so.shift(1).rolling(7, min_periods=1).mean().fillna(0.0)
        
        days_since = []
        last_so_idx = -999
        for idx, so_val in enumerate(is_so.shift(1).fillna(0)):
            if so_val == 1:
                last_so_idx = idx - 1
            if last_so_idx >= 0:
                days_since.append(min(30, idx - last_so_idx))
            else:
                days_since.append(30)
        grp["days_since_last_stockout"] = days_since
        
        # 5. Trend dynamics
        grp["trend_change_3d"] = (grp["trend_score"] - grp["trend_score"].shift(3)).fillna(0.0)
        
        # 6. Catalog attributes
        cat_info = cat_dict.get(pid, {})
        grp["price"] = float(cat_info.get("price", 50.0))
        grp["category"] = str(cat_info.get("category", "General"))
        
        # Censoring-corrected baseline feature
        c_mean = []
        for i in range(len(grp)):
            w14 = grp.iloc[max(0, i - 13): i + 1]
            non_so = w14[w14["stock_after"] > 0]["units_sold"]
            if len(non_so) > 0:
                c_mean.append(non_so.mean() * 7.0)
            else:
                rm7 = grp["rolling_mean_7d"].iloc[i]
                c_mean.append((rm7 if not np.isnan(rm7) else 0.0) * 7.0)
        grp["censoring_corrected_7d"] = c_mean
        
        # 7. Leakage-safe observable base rate & calendar lookahead features
        is_non_so = (grp["stock_after"] > 0)
        non_so_sales = grp["units_sold"].where(is_non_so)
        # Expanding mean over NON-stocked-out days up to day t (strictly shifted by 1)
        base_rate = non_so_sales.shift(1).expanding(min_periods=1).mean()
        grp["sku_base_rate"] = base_rate.fillna(0.0)
        
        # Target: future_demand_7d (uncensored sum of potential demand over next 7 days)
        if has_pd:
            future_pd = []
            for i in range(len(grp)):
                fw = grp["potential_demand"].iloc[i + 1: i + 1 + 7]
                if len(fw) < 7:
                    future_pd.append(np.nan)
                else:
                    future_pd.append(fw.sum())
            grp["future_demand_7d"] = future_pd
            
        # Target: stockout_within_3d (binary stockout flag within next 3 days)
        so_3d = []
        for i in range(len(grp)):
            fw_stock = grp["stock_after"].iloc[i + 1: i + 1 + 3]
            if len(fw_stock) < 3:
                so_3d.append(np.nan)
            elif (fw_stock <= 0).any():
                so_3d.append(1)
            else:
                so_3d.append(0)
        grp["stockout_within_3d"] = so_3d
        
        frames.append(grp)
        
    df_result = pd.concat(frames, ignore_index=True)
    return df_result


def compute_latest_sku_features(
    sku_history_df: pd.DataFrame,
    catalog_row: dict = None,
    trend_score_override: float = None
) -> pd.DataFrame:
    """
    Computes a single row of FEATURE_COLS for online inference for a single SKU.
    Uses the most recent observation in sku_history_df.
    """
    df_sku = sku_history_df.copy()
    df_sku["date"] = pd.to_datetime(df_sku["date"])
    df_sku.sort_values("date", inplace=True)
    df_sku.reset_index(drop=True, inplace=True)

    if trend_score_override is not None:
        df_sku.loc[df_sku.index[-1], "trend_score"] = float(trend_score_override)

    cat_df = pd.DataFrame([catalog_row]) if catalog_row else None
    features_full = compute_features(df_sku, cat_df)
    latest_row = features_full.iloc[[-1]][FEATURE_COLS]
    return latest_row
