"""
x_scrapper.py
-------------
Scrapes tweets for the given keywords using the twitterapi.io REST API,
preprocesses the results, and returns a cleaned pandas DataFrame.

No files are written to disk — data flows directly in memory.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# ── Load API key from .env ────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
_TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")

# ── Constants ─────────────────────────────────────────────────────────────────
_BASE_URL   = "https://api.twitterapi.io/twitter/tweet/advanced_search"
_QUERY_TYPE = "Latest"
_MAX_PAGES  = 2
_SLEEP_SEC  = 0.6


# ── Internal helpers ──────────────────────────────────────────────────────────

def _preprocess_text(text: str) -> str:
    """Basic text cleaning (URLs, mentions, special chars)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r'[^a-z0-9\s!?.,\'"]', " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_or_query(keywords: List[str]) -> str:
    """Join keywords with OR, quoting multi-word phrases."""
    parts = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        if " " in kw and not (kw.startswith('"') and kw.endswith('"')):
            parts.append(f'"{kw}"')
        else:
            parts.append(kw)
    return " OR ".join(parts)


def _fetch_pages(query: str) -> List[Dict[str, Any]]:
    """Paginate through twitterapi.io results and return raw tweet dicts."""
    if not _TWITTER_API_KEY:
        raise EnvironmentError(
            "TWITTER_API_KEY is not set. Please add it to your .env file."
        )

    headers = {"X-API-Key": _TWITTER_API_KEY}
    all_tweets: List[Dict[str, Any]] = []
    cursor: Optional[str] = ""

    for page in range(_MAX_PAGES):
        params: Dict[str, Any] = {"query": query, "queryType": _QUERY_TYPE}
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(_BASE_URL, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            print(f"[X] Request error on page {page + 1}: {exc}")
            break

        tweets = data.get("tweets", [])
        all_tweets.extend(tweets)

        has_next = data.get("has_next_page", False)
        cursor = data.get("next_cursor") or ""

        print(f"[X] Page {page + 1}: +{len(tweets)} tweets (total: {len(all_tweets)})")

        if not has_next or not cursor:
            break
        time.sleep(_SLEEP_SEC)

    return all_tweets


def _deduplicate(tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    unique = []
    for t in tweets:
        tid = t.get("id")
        if tid and tid not in seen:
            seen.add(tid)
            unique.append(t)
    return unique


def _flatten_tweets(tweets: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert raw tweet dicts to a clean DataFrame."""
    rows = []
    for t in tweets:
        text = _preprocess_text(t.get("text", "") or "")
        if not text:
            continue
        rows.append({
            "text":     text,
            "likes":    t.get("likeCount", 0) or 0,
            "comments": (t.get("retweetCount", 0) or 0) + (t.get("replyCount", 0) or 0),
            "date":     t.get("createdAt", "") or "",
        })

    if not rows:
        return pd.DataFrame(columns=["text", "likes", "comments", "date"])

    df = pd.DataFrame(rows)

    # Normalise the date column to a consistent string format
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%a %b %d %H:%M:%S %z %Y")
    except Exception:
        pass  # keep as-is if parsing fails

    df = df.fillna(0).reset_index(drop=True)
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def scrape_x(keywords: List[str]) -> pd.DataFrame:
    """
    Accepts a list of keyword strings, queries twitterapi.io, preprocesses
    the data, and returns a clean DataFrame with columns:
        text, likes, comments, date
    """
    if not keywords:
        return pd.DataFrame(columns=["text", "likes", "comments", "date"])

    query = _build_or_query(keywords)
    print(f"[X] Running query: {query}")

    raw_tweets = _fetch_pages(query)
    unique_tweets = _deduplicate(raw_tweets)

    print(f"[X] Collected {len(raw_tweets)} tweets → {len(unique_tweets)} unique.")
    df = _flatten_tweets(unique_tweets)
    print(f"[X] {len(df)} rows after preprocessing.")
    return df