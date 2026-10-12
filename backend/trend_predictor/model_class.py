#!/usr/bin/env python3
"""
AdVantage OS - TrendForecastingModel Definition
===============================================
Encapsulates the high-accuracy Hybrid Delta-Momentum Trend Forecasting Model.
Ensures seamless serialization and deserialization across scripts, FastAPI, and database.
"""

from typing import Any
import numpy as np
import pandas as pd


class TrendForecastingModel:
    """
    Production-grade Hybrid Delta Trend Forecaster.
    Implements Scikit-learn .predict(X) for drop-in backward compatibility,
    along with native multi-horizon forecasting with momentum damping.
    """

    def __init__(self, delta_regressor: Any, feature_cols: list[str], tag_list: list[str], sigma: float = 0.35):
        self.delta_regressor = delta_regressor
        self.feature_cols = feature_cols
        self.tag_list = tag_list
        self.sigma = float(sigma)
        self.model_name = "TrendForecastingModel_v2_DeltaEnsemble"

    def predict(self, X: Any) -> np.ndarray:
        """
        Scikit-learn compatible prediction.
        Returns predicted level: lag_1 + predicted_delta.
        """
        if isinstance(X, pd.DataFrame):
            X_feat = X[self.feature_cols].copy() if all(c in X.columns for c in self.feature_cols) else X
            deltas = self.delta_regressor.predict(X_feat)
            if "lag_1" in X.columns:
                levels = X["lag_1"].values + deltas
                return np.maximum(levels, 0.0)
            return deltas
        return self.delta_regressor.predict(X)

    def forecast_series(
        self,
        series: pd.Series,
        tag_name: str = "",
        horizon: int = 30,
        damping: float = 0.97,
    ) -> pd.DataFrame:
        """
        High-accuracy multi-horizon time series forecasting.
        Computes realistic momentum trajectory, natural saturation, and confidence bands.
        """
        from trend_predictor.predict import _make_features, _resolve_closest_tag

        effective_tag = _resolve_closest_tag(tag_name, self.tag_list) if tag_name else ""
        
        target_domain_mean = 14.0
        ref_mean = max(float(series.iloc[-7:].mean()), 1.0)
        scale = ref_mean / target_domain_mean if ref_mean > 22.0 else 1.0

        scaled_history = (series / scale).copy()
        preds = []
        last_date = series.index.max()
        volatility = max(float(series.iloc[-7:].std()) if len(series) >= 7 else self.sigma, 0.4)
        observed_slope = float(series.iloc[-1] - series.iloc[0]) / max(len(series) - 1, 1)

        for step in range(1, horizon + 1):
            next_date = last_date + pd.Timedelta(days=step)
            tmp = pd.concat([scaled_history, pd.Series([np.nan], index=[next_date])])
            feat = _make_features(tmp, effective_tag, self.tag_list, self.feature_cols).loc[[next_date], self.feature_cols]
            feat = feat.bfill().fillna(0.0)

            # Predict delta in model's native domain and scale back
            raw_delta_scaled = float(self.delta_regressor.predict(feat)[0])
            ml_delta = raw_delta_scaled * scale
            
            # Blend ML forecast with observed empirical momentum with damping
            combined_delta = (0.65 * ml_delta + 0.35 * observed_slope) * (damping ** (step - 1))
            
            # Add slight weekly cyclic rhythm (weekend engagement bump)
            day_of_week = next_date.dayofweek
            weekend_boost = 0.12 if day_of_week in [4, 5, 6] else -0.04
            
            prev_val = float(series.iloc[-1] if step == 1 else preds[-1]["predicted_trend_score"])
            next_val = max(round(prev_val + combined_delta + weekend_boost * (abs(combined_delta) > 0.02), 2), 0.5)
            
            scaled_history.loc[next_date] = next_val / scale

            # Expanding 95% confidence bounds
            ci_spread = round(1.96 * volatility * np.sqrt(step / 3.0), 2)
            ci_low = max(round(next_val - ci_spread, 2), 0.0)
            ci_high = round(next_val + ci_spread, 2)

            preds.append({
                "date": next_date.date(),
                "predicted_trend_score": round(next_val, 2),
                "confidence_lower": ci_low,
                "confidence_upper": ci_high,
            })

        return pd.DataFrame(preds)
