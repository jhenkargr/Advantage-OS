"""
reddit_scrap.py
---------------
Scrapes Reddit posts for the given keywords using the ScrapeBadger API,
preprocesses the results, and returns a cleaned pandas DataFrame.

No files are written to disk — data flows directly in memory.
"""

import asyncio
import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scrapebadger import ScrapeBadger

# ── Load API key from .env ────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
_SCRAPEBADGER_API_KEY = os.getenv("SCRAPEBADGER_API_KEY", "")

# ── Constants ─────────────────────────────────────────────────────────────────
SORT_BY = "top"
TIME_FILTER = "week"
LIMIT = 50          # posts per keyword


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


def _estimate_likes(score: float, upvote_ratio: float) -> int:
    """Estimate upvotes from Reddit score + upvote_ratio."""
    denom = 2 * upvote_ratio - 1
    if denom == 0:
        return int(score)
    raw = upvote_ratio * score / denom
    return max(0, int(round(raw)))


async def _async_scrape(keywords: list[str]) -> pd.DataFrame:
    """Async inner function — fetches posts for all keywords."""
    if not _SCRAPEBADGER_API_KEY:
        raise EnvironmentError(
            "SCRAPEBADGER_API_KEY is not set. Please add it to your .env file."
        )

    all_posts: list[dict] = []

    async with ScrapeBadger(api_key=_SCRAPEBADGER_API_KEY) as client:
        for kw in keywords:
            print(f"[Reddit] Scraping keyword: '{kw}'")
            try:
                results = await client.reddit.search.posts(
                    query=kw,
                    sort=SORT_BY,
                    time_filter=TIME_FILTER,
                    limit=LIMIT,
                )
                for post in results.posts:
                    raw = post.model_dump()
                    all_posts.append({
                        "text":     raw.get("title", "") or "",
                        "body":     raw.get("selftext", "") or "",
                        "score":    raw.get("score", 0),
                        "upvote_ratio": raw.get("upvote_ratio", 0.5),
                        "num_comments": raw.get("num_comments", 0),
                        "created_utc":  raw.get("created_utc", 0),
                    })
            except Exception as exc:
                print(f"[Reddit] Error for keyword '{kw}': {exc}")

    if not all_posts:
        print("[Reddit] No posts collected — returning empty DataFrame.")
        return pd.DataFrame(columns=["text", "likes", "comments", "date"])

    df = pd.DataFrame(all_posts)

    # ── Derived columns ───────────────────────────────────────────────────────
    df["likes"] = df.apply(
        lambda r: _estimate_likes(r["score"], r["upvote_ratio"]), axis=1
    )
    df["comments"] = df["num_comments"].astype(int)
    df["date"] = pd.to_datetime(
        df["created_utc"], unit="s", utc=True
    ).dt.strftime("%a %b %d %H:%M:%S %z %Y")

    # ── Clean text ────────────────────────────────────────────────────────────
    df["text"] = df["text"].apply(_preprocess_text)

    # ── Keep only required columns ────────────────────────────────────────────
    df = df[["text", "likes", "comments", "date"]].copy()
    df = df.fillna(0)

    # Drop rows where text is empty after cleaning
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    print(f"[Reddit] Collected {len(df)} clean posts.")
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def scrape_reddit(keywords: list[str]) -> pd.DataFrame:
    """
    Synchronous entry point. Accepts a list of keyword strings, scrapes
    Reddit, preprocesses the data, and returns a clean DataFrame with columns:
        text, likes, comments, date
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an async context (e.g., FastAPI); schedule as a coroutine
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _async_scrape(keywords))
                return future.result()
        else:
            return loop.run_until_complete(_async_scrape(keywords))
    except RuntimeError:
        return asyncio.run(_async_scrape(keywords))