"""
predict.py
----------
Core prediction module.

Flow:
  keywords (list[str])
      ↓
  scrape_reddit()  +  scrape_x()      ← run concurrently
      ↓                  ↓
    reddit_df          x_df
          ↓ merge ↓
        merged_df
          ↓
    sentiment analysis (ensemble: VADER + DistilBERT)
          ↓
    trend score aggregation
          ↓
    ML model forecast (30-day)
          ↓
    summary dict  ← returned to FastAPI
"""

from __future__ import annotations

import concurrent.futures
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ── NLTK bootstrap ─────────────────────────────────────────────────────────────
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    print("[INFO] Downloading NLTK vader_lexicon …")
    nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()
_pipeline_cache: Any = None   # None = not loaded yet; False = failed

# ── Model path ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
MODEL_PATH = _HERE / "model_test_analysis.pkl"


# ── 1. Text preprocessing ──────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Clean raw social-post text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r'[^a-z0-9\s!?.,\'"]', " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── 2. Sentiment scorers ───────────────────────────────────────────────────────

def rule_based_sentiment(text: str) -> dict:
    return {"score": sia.polarity_scores(preprocess(text))["compound"]}


def get_transformer_pipeline():
    global _pipeline_cache
    if _pipeline_cache is None:
        try:
            from transformers import pipeline as hf_pipeline
            _pipeline_cache = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512,
            )
            print("[INFO] Transformer pipeline loaded.")
        except Exception as exc:
            print(f"[WARN] Transformer unavailable ({exc}). Falling back to VADER.")
            _pipeline_cache = False
    return _pipeline_cache if _pipeline_cache is not False else None


def transformer_sentiment(text: str) -> dict:
    pipe = get_transformer_pipeline()
    if pipe is None:
        return rule_based_sentiment(text)
    clean = preprocess(text)[:512]
    if not clean:
        return {"score": 0.0, "label": "neutral"}
    result = pipe(clean)[0]
    label = {"POSITIVE": "positive", "NEGATIVE": "negative"}.get(result["label"], "neutral")
    score = result["score"] if label == "positive" else -result["score"]
    return {"score": float(score), "label": label}


def ensemble_sentiment(text: str, use_transformer: bool = True) -> dict:
    """Weighted ensemble: 70 % transformer + 30 % VADER."""
    lex = rule_based_sentiment(text)
    if use_transformer:
        trans = transformer_sentiment(text)
        combined = 0.7 * trans["score"] + 0.3 * lex["score"]
    else:
        trans = lex
        combined = lex["score"]

    if combined > 0.15:
        label = "positive"
    elif combined < -0.15:
        label = "negative"
    else:
        label = "neutral"

    return {
        "sentiment_score": round(combined, 4),
        "sentiment_label": label,
        "lexicon_score":   round(lex["score"], 4),
        "transformer_score": round(trans.get("score", lex["score"]), 4),
    }


def analyze_dataframe_sentiment(
    df: pd.DataFrame, text_col: str = "text", use_transformer: bool = True
) -> pd.DataFrame:
    """Appends sentiment columns to *df* in-place (returns new DataFrame)."""
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")
    print(f"[INFO] Analyzing sentiment for {len(df)} rows …")
    results = df[text_col].apply(lambda x: ensemble_sentiment(x, use_transformer))
    sentiment_df = pd.DataFrame(results.tolist(), index=df.index)
    return pd.concat([df, sentiment_df], axis=1)


# ── 3. Trend-score aggregation ─────────────────────────────────────────────────

def build_daily_series(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate row-level data into a smoothed daily trend_score Series.
    Expected columns: text, likes, comments, date, score
    (matches original pipeline column naming).
    """
    df = df.copy()
    # Use the same explicit format as the original script
    df["ts"] = pd.to_datetime(
        df["date"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce"
    )
    df = df.dropna(subset=["ts"])
    df["date_only"] = df["ts"].dt.tz_localize(None).dt.normalize()

    daily = (
        df.groupby("date_only")
        .agg(
            post_count=("text", "count"),
            total_likes=("likes", "sum"),
            total_comments=("comments", "sum"),
            avg_sentiment=("score", "mean"),  # matches original 'score' column
        )
        .reset_index()
    )

    daily["trend_score"] = (
        daily["post_count"]
        * np.log1p(daily["total_likes"] + 2 * daily["total_comments"])
        * (1 + 0.5 * daily["avg_sentiment"].abs())
    )

    daily = daily.set_index("date_only").sort_index()
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily_full = daily["trend_score"].reindex(full_idx).ffill().bfill()
    smoothed = daily_full.rolling(7, min_periods=1).mean()
    return smoothed


# ── 4. Feature builder & recursive forecast ───────────────────────────────────

def _make_features(series: pd.Series, tag_name: str, tag_list: list, feature_cols: list) -> pd.DataFrame:
    d = pd.DataFrame({"trend_score": series})
    for lag in [1, 2, 3, 7, 14, 21]:
        d[f"lag_{lag}"] = d["trend_score"].shift(lag)
    d["roll_mean_14"] = d["trend_score"].shift(1).rolling(14, min_periods=1).mean()
    d["roll_mean_30"] = d["trend_score"].shift(1).rolling(30, min_periods=1).mean()
    d["roll_std_14"]  = d["trend_score"].shift(1).rolling(14, min_periods=1).std()
    d["slope_7"]      = d["trend_score"].shift(1) - d["trend_score"].shift(8)
    d["dow"]   = d.index.dayofweek
    d["month"] = d.index.month
    for t in tag_list:
        d[f"tag_{t}"] = 1.0 if t == tag_name else 0.0
    # Fill any NaN lags/rolling features with 0 so GradientBoostingRegressor
    # doesn't reject rows when the history is shorter than the window size.
    return d.fillna(0)


def _recursive_forecast(
    series: pd.Series,
    model,
    tag_name: str,
    tag_list: list,
    feature_cols: list,
    horizon: int = 30,
) -> pd.DataFrame:
    history = series.copy()
    preds = []
    last_date = history.index.max()
    for step in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=step)
        tmp = pd.concat([history, pd.Series([np.nan], index=[next_date])])
        feat = _make_features(tmp, tag_name, tag_list, feature_cols).loc[[next_date], feature_cols]
        feat = feat.fillna(0)   # guard: NaN lags from short history -> 0
        pred = max(model.predict(feat)[0], 0)
        preds.append((next_date, pred))
        history.loc[next_date] = pred
    forecast_df = pd.DataFrame(preds, columns=["date", "predicted_trend_score"])
    forecast_df["date"] = forecast_df["date"].dt.date
    forecast_df["predicted_trend_score"] = forecast_df["predicted_trend_score"].round(2)
    return forecast_df


def _indication(pct: float) -> str:
    if pct >= 15:
        return "Rising"
    elif pct <= -15:
        return "Declining"
    return "Stable"


# ── 5. Public entry point ──────────────────────────────────────────────────────

def run_prediction(keywords: list[str], use_transformer: bool = True) -> dict:
    """
    Main orchestration function called by FastAPI.

    Parameters
    ----------
    keywords       : list of search terms
    use_transformer: if True, use DistilBERT + VADER ensemble; else VADER only

    Returns
    -------
    A dict with keys:
        topic, current_trend_score,
        predicted_in_15_days, change_15d_pct, indication_15d,
        predicted_in_30_days, change_30d_pct, indication_30d,
        forecast  (list of {date, predicted_trend_score}),
        rows_collected  (total posts used)
    """
    from reddit_scrap import scrape_reddit
    from x_scrapper import scrape_x

    # ── Step 1: scrape both sources concurrently ───────────────────────────────
    print("[Predict] Starting parallel scraping …")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        reddit_future = pool.submit(scrape_reddit, keywords)
        x_future      = pool.submit(scrape_x, keywords)
        reddit_df = reddit_future.result()
        x_df      = x_future.result()

    # ── Step 2: merge ──────────────────────────────────────────────────────────
    merged = pd.concat([reddit_df, x_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"[Predict] Merged dataset: {len(merged)} rows "
          f"({len(reddit_df)} Reddit + {len(x_df)} X).")

    if merged.empty:
        return {
            "error": "No data collected for the given keywords.",
            "keywords": keywords,
        }

    # ── Step 3: sentiment analysis ─────────────────────────────────────────────
    analyzed = analyze_dataframe_sentiment(merged, text_col="text", use_transformer=use_transformer)
    # Rename 'sentiment_score' -> 'score' to match the original pipeline column name
    analyzed = analyzed.rename(columns={"sentiment_score": "score"})

    # ── Step 4: build daily series ─────────────────────────────────────────────
    if "score" not in analyzed.columns:
        analyzed["score"] = 0.0

    smoothed = build_daily_series(analyzed)

    if smoothed.empty or len(smoothed) < 2:
        return {
            "error": "Insufficient time-series data to build a forecast. "
                     "Try broader keywords or a wider time window.",
            "keywords": keywords,
            "rows_collected": len(analyzed),
        }

    # ── Step 5: load model and forecast ───────────────────────────────────────
    if not MODEL_PATH.exists():
        return {
            "error": f"Model file not found at {MODEL_PATH}. "
                     "Please place model_test_analysis.pkl next to predict.py.",
            "rows_collected": len(analyzed),
        }

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    model        = saved["model"]
    feature_cols = saved["feature_cols"]
    tag_list     = saved["tag_list"]

    topic_name = " OR ".join(keywords)
    forecast_df = _recursive_forecast(smoothed, model, topic_name, tag_list, feature_cols, horizon=30)

    # ── Step 6: summary ────────────────────────────────────────────────────────
    current = float(smoothed.iloc[-1])
    d15 = float(forecast_df.iloc[14]["predicted_trend_score"])
    d30 = float(forecast_df.iloc[29]["predicted_trend_score"])
    chg_15 = (d15 - current) / max(current, 1e-6) * 100
    chg_30 = (d30 - current) / max(current, 1e-6) * 100

    return {
        "keywords":              keywords,
        "topic":                 topic_name,
        "rows_collected":        len(analyzed),
        "current_trend_score":   round(current, 2),
        "predicted_in_15_days":  round(d15, 2),
        "change_15d_pct":        round(chg_15, 1),
        "indication_15d":        _indication(chg_15),
        "predicted_in_30_days":  round(d30, 2),
        "change_30d_pct":        round(chg_30, 1),
        "indication_30d":        _indication(chg_30),
        "forecast": forecast_df.assign(
            date=forecast_df["date"].astype(str)
        ).to_dict(orient="records"),
    }