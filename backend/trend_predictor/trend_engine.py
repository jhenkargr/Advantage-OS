#!/usr/bin/env python3
"""
AdVantage OS - Trend Engine (Module 1)
======================================
Fetches local social-media signals for an SME category, discovers trends with
LDA topic modelling, and forecasts each trend's growth / peak / decay.

Sources
  * Reddit          (ScrapeBadger scraper as used in advantage/reddit_scrap.py)
  * YouTube         (YouTube Data API v3)
  * Google Trends   (pytrends: interest over time + rising related queries)

Pipeline
  fetch -> clean -> LDA topics -> weekly time series per topic
        -> LSTM forecast (fallback: linear trend) -> ranked trends (CSV + JSON)

Install
  pip install scrapebadger python-dotenv requests pytrends pandas numpy scikit-learn nltk
  pip install tensorflow          # optional, enables the LSTM forecaster

Credentials (environment variables or advantage/.env; a source with no key is skipped)
  SCRAPEBADGER_API_KEY
  YOUTUBE_API_KEY

Usage
  python trend_engine.py --sme cafe --city Bengaluru
  python trend_engine.py --sme boutique --local-only --topics 8
  python trend_engine.py --sme salon --sources reddit,trends
  python trend_engine.py --sme cafe --demo        # synthetic data, no API keys
  python trend_engine.py --list-smes
"""

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

def _load_env_fallback():
    env_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "advantage", ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for p in env_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("\"'")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "advantage", ".env"))
except ImportError:
    pass

_load_env_fallback()

# --------------------------------------------------------------------------
# 1. SME PROFILES  (seed keywords, subreddits, YouTube queries, Trends terms)
#    Add new SME types here.
# --------------------------------------------------------------------------
SME_PROFILES = {
    "cafe": {
        "seeds": ["coffee", "latte", "matcha", "cold brew", "brunch", "cafe",
                  "espresso", "dessert", "bubble tea", "milkshake"],
        "subreddits": ["bangalore", "Coffee", "indianfood", "IndianFoodPhotos"],
        "youtube": ["best cafes {city}", "{city} cafe vlog", "new cafe {city}",
                    "trending drinks cafe", "{city} food vlog cafe"],
        "trends": ["matcha", "cold brew", "iced latte", "cafe", "brunch"],
    },
    "restaurant": {
        "seeds": ["restaurant", "biryani", "burger", "ramen", "pizza", "dosa",
                  "street food", "buffet", "new restaurant", "food"],
        "subreddits": ["bangalore", "indianfood", "FoodPorn", "IndianFoodPhotos"],
        "youtube": ["best restaurants {city}", "{city} street food",
                    "{city} food vlog", "new restaurant {city}"],
        "trends": ["biryani", "ramen", "smash burger", "street food", "pizza"],
    },
    "bakery": {
        "seeds": ["cake", "bakery", "cheesecake", "cookies", "brownie",
                  "sourdough", "dessert", "pastry", "millet cookies"],
        "subreddits": ["bangalore", "Baking", "indianfood", "Breadit"],
        "youtube": ["best bakery {city}", "{city} dessert vlog",
                    "trending desserts", "cake trends"],
        "trends": ["cheesecake", "sourdough", "brownie", "dessert", "cake"],
    },
    "boutique": {
        "seeds": ["kurti", "co-ord set", "ethnic wear", "saree", "thrift",
                  "fashion", "outfit", "lehenga", "streetwear", "sustainable fashion"],
        "subreddits": ["bangalore", "IndianFashionAddicts", "indianfashion", "femalefashionadvice"],
        "youtube": ["{city} fashion haul", "{city} street shopping haul",
                    "latest kurti designs", "fashion trends india"],
        "trends": ["co-ord set", "kurti", "saree", "thrift", "streetwear"],
    },
    "salon": {
        "seeds": ["hair spa", "balayage", "facial", "skincare", "keratin",
                  "haircut", "makeup", "nail art", "glass skin", "salon"],
        "subreddits": ["bangalore", "IndianSkincareAddicts", "IndianHaircare", "IndianMakeupAddicts"],
        "youtube": ["salon vlog {city}", "hair trends", "skincare routine india",
                    "nail art trends", "{city} makeup"],
        "trends": ["balayage", "keratin", "glass skin", "nail art", "hair spa"],
    },
    "gym": {
        "seeds": ["gym", "pilates", "crossfit", "yoga", "functional training",
                  "protein", "zumba", "calisthenics", "pickleball", "fitness"],
        "subreddits": ["bangalore", "IndianFitness", "naturalbodybuilding", "yoga"],
        "youtube": ["{city} gym tour", "fitness trends india", "pilates workout",
                    "home workout india"],
        "trends": ["pilates", "crossfit", "yoga", "pickleball", "calisthenics"],
    },
    "gift_shop": {
        "seeds": ["gift", "hamper", "personalised gift", "diwali gift",
                  "home decor", "candles", "photo frame", "handmade", "terrarium"],
        "subreddits": ["bangalore", "giftideas", "DIY", "HomeDecorating"],
        "youtube": ["gift ideas india", "diwali decor ideas", "{city} gift shopping",
                    "handmade gifts"],
        "trends": ["gift hamper", "home decor", "candles", "personalised gifts", "diwali decor"],
    },
    "electronics": {
        "seeds": ["gadgets", "earbuds", "smart watch", "phone case", "chargers",
                  "headphones", "bluetooth speaker", "accessories", "power bank", "cable organizer"],
        "subreddits": ["bangalore", "IndianGaming", "gadgets", "android"],
        "youtube": ["tech haul {city}", "best gadgets india", "budget accessories", "smart gadgets"],
        "trends": ["earbuds", "smartwatch", "gadgets", "power bank", "phone case"],
    },
    "d2c": {
        "seeds": ["handmade", "scented candles", "artisan", "organic skincare", "thrift store",
                  "clay pottery", "custom jewellery", "tote bag", "home decor", "small business"],
        "subreddits": ["bangalore", "smallbusiness", "crafts", "IndianFashionAddicts"],
        "youtube": ["small business vlog india", "d2c packaging", "handcrafted haul {city}", "diy decor"],
        "trends": ["scented candles", "handmade jewelry", "tote bag", "artisan craft", "organic skincare"],
    },
}

SME_ALIASES = {
    "cafes": "cafe",
    "cafes_bakeries": "cafe",
    "cafes_bakeries_and_dessert_shops": "cafe",
    "bakery": "bakery",
    "bakeries": "bakery",
    "restaurants": "restaurant",
    "restaurants_cloud_kitchens": "restaurant",
    "restaurants_and_cloud_kitchens": "restaurant",
    "boutiques": "boutique",
    "boutiques_fashion": "boutique",
    "boutiques_and_fashion_stores": "boutique",
    "fashion": "boutique",
    "salons": "salon",
    "salons_beauty": "salon",
    "salons_and_beauty_studios": "salon",
    "gyms": "gym",
    "gyms_yoga": "gym",
    "gyms_and_yoga_studios": "gym",
    "gift": "gift_shop",
    "gift_decor": "gift_shop",
    "gift_and_decor_shops": "gift_shop",
    "decor": "gift_shop",
    "electronics_accessories": "electronics",
    "electronics_and_accessory_shops": "electronics",
    "small_d2c": "d2c",
    "small_d2c_and_home_based_sellers": "d2c",
    "d2c_sellers": "d2c",
}

# Neighbourhoods used for the local filter (extend for other cities)
CITY_LOCAL_TERMS = {
    "bengaluru": ["bengaluru", "bangalore", "blr", "indiranagar", "koramangala",
                  "hsr", "jayanagar", "whitefield", "malleshwaram", "basavanagudi",
                  "jp nagar", "btm", "marathahalli", "electronic city", "yelahanka",
                  "mg road", "church street", "commercial street", "rajajinagar",
                  "sadashivanagar", "ulsoor", "hebbal", "banashankari"],
}
CITY_TRENDS_GEO = {"bengaluru": "IN-KA"}  # Google Trends sub-region code

CUSTOM_STOPWORDS = {
    "like", "just", "really", "im", "ive", "dont", "didnt", "get", "got", "one",
    "also", "would", "could", "much", "even", "still", "know", "think", "going",
    "good", "great", "best", "amp", "https", "http", "www", "com", "youtube",
    "video", "subscribe", "channel", "watch", "follow", "instagram", "reddit",
    "post", "thanks", "please", "lol", "yeah", "people", "thing", "things",
    "time", "day", "new", "want", "need", "use", "make", "look", "see", "place",
    "bangalore", "bengaluru", "india", "indian", "link", "bio", "comment", "like",
}


# --------------------------------------------------------------------------
# 2. DATA FETCHERS
# --------------------------------------------------------------------------
def fetch_reddit(profile, city, days, per_query_limit, local_terms):
    """Search Reddit using ScrapeBadger (as used in advantage/reddit_scrap.py)."""
    api_key = os.getenv("SCRAPEBADGER_API_KEY")
    if not api_key:
        print("[reddit] SCRAPEBADGER_API_KEY not set -> skipped")
        return []
    try:
        from scrapebadger import ScrapeBadger
    except ImportError:
        print("[reddit] scrapebadger not installed (pip install scrapebadger) -> skipped")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    time_filter = "week" if days <= 7 else "month" if days <= 31 else "year"
    limit = min(per_query_limit, 50)
    seen, rows = set(), []

    async def _async_fetch():
        async with ScrapeBadger(api_key=api_key) as client:
            for seed in profile["seeds"]:
                query = seed
                print(f"[reddit] Scraping via ScrapeBadger: '{query}'")

                max_retries = 3
                delay_after_rate_limit = 5
                results = None

                for attempt in range(max_retries):
                    try:
                        results = await client.reddit.search.posts(
                            query=query,
                            sort="top",
                            time_filter=time_filter,
                            limit=limit,
                        )
                        break
                    except Exception as exc:
                        err_msg = str(exc)
                        if "insufficient_credits" in err_msg.lower():
                            print("[reddit] Note: ScrapeBadger credits exhausted. Update SCRAPEBADGER_API_KEY in advantage/.env.")
                            return
                        if "429" in err_msg or "rate limit" in err_msg.lower():
                            if attempt < max_retries - 1:
                                wait_sec = delay_after_rate_limit * (attempt + 1)
                                print(f"[reddit] Rate limit hit for '{query}'. Waiting {wait_sec}s before retry (attempt {attempt+1}/{max_retries})...")
                                await asyncio.sleep(wait_sec)
                                continue
                        print(f"[reddit] Error for keyword '{query}': {exc}")
                        break

                if not results or not getattr(results, "posts", None):
                    await asyncio.sleep(1.5)
                    continue

                for post in results.posts:
                    raw = post.model_dump()
                    pid = str(raw.get("id") or raw.get("name") or hash(raw.get("title", "")))
                    if pid in seen:
                        continue
                    seen.add(pid)

                    created_ts = raw.get("created_utc", 0)
                    if created_ts:
                        created = datetime.fromtimestamp(created_ts, tz=timezone.utc)
                        if created < cutoff:
                            continue
                    else:
                        created = datetime.now(timezone.utc)

                    title = raw.get("title", "") or ""
                    body = raw.get("selftext", "") or ""
                    text = f"{title}. {body}".strip()
                    if not text:
                        continue

                    score = int(raw.get("score", 0))
                    num_comments = int(raw.get("num_comments", 0))
                    sub_name = str(raw.get("subreddit", ""))

                    is_local = (
                        any(c.lower() in sub_name.lower() for c in [city, "bangalore", "bengaluru"])
                        or _is_local(text, local_terms)
                    )

                    rows.append({
                        "source": "reddit",
                        "id": pid,
                        "date": created,
                        "text": text,
                        "engagement": score + num_comments,
                        "is_local": is_local,
                        "origin": f"reddit:{seed}",
                    })

                # Polite delay between keywords to prevent hitting per-second rate limits
                await asyncio.sleep(2.0)

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, _async_fetch()).result()
        else:
            asyncio.run(_async_fetch())
    except Exception as e:
        print(f"[reddit] Failed to fetch: {e}")

    print(f"[reddit] collected {len(rows)} posts")
    return rows


def fetch_youtube(profile, city, days, max_per_query, local_terms, with_comments=False):
    """Search YouTube for each query, then pull stats (and optional comments)."""
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        print("[youtube] YOUTUBE_API_KEY not set -> skipped")
        return []
    import requests

    base = "https://www.googleapis.com/youtube/v3"
    published_after = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, seen = [], set()

    for q in profile["youtube"]:
        q = q.format(city=city)
        try:
            r = requests.get(f"{base}/search", params={
                "part": "snippet", "q": q, "type": "video", "order": "date",
                "maxResults": min(max_per_query, 50), "regionCode": "IN",
                "relevanceLanguage": "en", "publishedAfter": published_after,
                "key": key}, timeout=30)
            r.raise_for_status()
            items = r.json().get("items", [])
            ids = [i["id"]["videoId"] for i in items if i["id"]["videoId"] not in seen]
            if not ids:
                continue
            stats = requests.get(f"{base}/videos", params={
                "part": "snippet,statistics", "id": ",".join(ids), "key": key},
                timeout=30)
            stats.raise_for_status()
            for v in stats.json().get("items", []):
                vid, sn, st = v["id"], v["snippet"], v.get("statistics", {})
                seen.add(vid)
                text = f"{sn.get('title', '')}. {sn.get('description', '')[:600]}"
                eng = int(st.get("likeCount", 0)) + int(st.get("commentCount", 0)) \
                      + int(int(st.get("viewCount", 0)) / 100)
                rows.append({
                    "source": "youtube", "id": vid,
                    "date": pd.to_datetime(sn["publishedAt"], utc=True).to_pydatetime(),
                    "text": text, "engagement": eng,
                    "is_local": _is_local(text + " " + q, local_terms) or city.lower() in q.lower(),
                    "origin": f"yt:{q}",
                })
                if with_comments:
                    rows.extend(_youtube_comments(vid, key, base, local_terms, sn["publishedAt"]))
            time.sleep(0.3)
        except Exception as e:
            print(f"[youtube] query '{q}' failed: {e}")
    print(f"[youtube] collected {len(rows)} items")
    return rows


def _youtube_comments(video_id, key, base, local_terms, fallback_date, n=20):
    import requests
    out = []
    try:
        r = requests.get(f"{base}/commentThreads", params={
            "part": "snippet", "videoId": video_id, "maxResults": n,
            "order": "relevance", "textFormat": "plainText", "key": key}, timeout=30)
        if r.status_code != 200:   # comments disabled etc.
            return out
        for c in r.json().get("items", []):
            s = c["snippet"]["topLevelComment"]["snippet"]
            out.append({
                "source": "youtube_comment", "id": c["id"],
                "date": pd.to_datetime(s.get("publishedAt", fallback_date), utc=True).to_pydatetime(),
                "text": s.get("textDisplay", ""), "engagement": int(s.get("likeCount", 0)),
                "is_local": _is_local(s.get("textDisplay", ""), local_terms),
                "origin": f"yt-comment:{video_id}",
            })
    except Exception:
        pass
    return out


def fetch_google_trends(profile, geo, days):
    """Return (interest_over_time DataFrame, rising related queries DataFrame)."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("[trends] pytrends not installed (pip install pytrends) -> skipped")
        return pd.DataFrame(), pd.DataFrame()

    timeframe = "now 7-d" if days <= 7 else "today 1-m" if days <= 31 else \
                "today 3-m" if days <= 93 else "today 12-m"
    pt = TrendReq(hl="en-IN", tz=330, retries=2, backoff_factor=0.5)
    kws = profile["trends"][:5]  # Trends API allows max 5 per request
    interest, rising_rows = pd.DataFrame(), []
    try:
        pt.build_payload(kws, timeframe=timeframe, geo=geo)
        interest = pt.interest_over_time()
        if "isPartial" in interest.columns:
            interest = interest.drop(columns=["isPartial"])
        time.sleep(2)
        related = pt.related_queries()
        for kw, d in related.items():
            if d and d.get("rising") is not None:
                df = d["rising"].copy()
                df.insert(0, "seed", kw)
                rising_rows.append(df)
        print(f"[trends] interest series: {len(interest)} rows, "
              f"rising queries for {len(rising_rows)} seeds")
    except Exception as e:
        print(f"[trends] failed (rate-limit or no data): {e}")
    rising = pd.concat(rising_rows, ignore_index=True) if rising_rows else pd.DataFrame()
    if not rising.empty:
        rising = rising.rename(columns={"query": "rising_query", "value": "growth_pct"})
    return interest, rising


def _is_local(text, local_terms):
    t = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", t) for term in local_terms)


# --------------------------------------------------------------------------
# 3. SYNTHETIC DEMO DATA (no API keys needed; for testing the pipeline)
# --------------------------------------------------------------------------
def make_demo_data(days=120, seed=7):
    rng = np.random.default_rng(seed)
    topics = {
        "matcha":   (["matcha latte", "ceremonial matcha", "oat milk matcha whisk",
                      "iced matcha cafe", "matcha green tea powder"], "rise"),
        "coldbrew": (["cold brew coffee", "nitro cold brew tonic", "coffee tonic cafe",
                      "cold brew concentrate bottle", "espresso tonic"], "peak"),
        "brunch":   (["weekend brunch menu", "avocado toast brunch", "eggs benedict brunch",
                      "bottomless brunch cafe", "brunch cafe pancakes"], "flat"),
        "bubble":   (["bubble tea boba", "tapioca pearls milk tea", "boba shop queue",
                      "brown sugar boba", "taro milk tea"], "decay"),
        "dessert":  (["dubai chocolate kunafa", "pistachio kunafa bar", "viral chocolate dessert",
                      "kunafa chocolate bar", "pistachio dessert cafe"], "rise"),
    }
    local_bits = ["Indiranagar", "Koramangala", "HSR Layout", "Bangalore", "Jayanagar", ""]
    now = datetime.now(timezone.utc)
    rows, rid = [], 0
    for name, (phrases, shape) in topics.items():
        for d in range(days):
            x = d / days
            lam = {"rise": 0.3 + 5 * x ** 2, "peak": 4 * math.exp(-((x - 0.7) ** 2) / 0.02),
                   "flat": 1.5, "decay": 4 * (1 - x) ** 2 + 0.2}[shape]
            for _ in range(rng.poisson(lam)):
                rid += 1
                text = f"{rng.choice(phrases)} {rng.choice(phrases)} {rng.choice(local_bits)} loved it"
                rows.append({
                    "source": rng.choice(["reddit", "youtube"]), "id": f"demo{rid}",
                    "date": now - timedelta(days=days - d),
                    "text": text, "engagement": int(rng.integers(1, 300)),
                    "is_local": "" != text.split()[-2], "origin": "demo",
                })
    return rows


# --------------------------------------------------------------------------
# 4. TEXT PREPROCESSING
# --------------------------------------------------------------------------
_lemmatizer = None
_stopwords = None


def _init_nlp():
    global _lemmatizer, _stopwords
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    _stopwords = set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        _lemmatizer = WordNetLemmatizer()
        _lemmatizer.lemmatize("tests")  # trigger load; raises if data missing
    except Exception:
        _lemmatizer = None  # fall back to light suffix stripping


def _lemma(tok):
    if _lemmatizer is not None:
        return _lemmatizer.lemmatize(tok)
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def clean_text(text):
    if _stopwords is None:
        _init_nlp()
    text = re.sub(r"http\S+|www\.\S+", " ", text.lower())
    text = re.sub(r"[^a-z\s]", " ", text)
    toks = [_lemma(t) for t in text.split() if len(t) > 2]
    return " ".join(t for t in toks if t not in _stopwords)


# --------------------------------------------------------------------------
# 5. LDA TOPIC MODELLING
# --------------------------------------------------------------------------
def run_lda(df, n_topics, top_words=6, seed=42):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    min_df = 3 if len(df) >= 100 else 2
    vec = CountVectorizer(max_df=0.6, min_df=min_df, ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(df["clean"])
    if X.shape[1] < n_topics:
        n_topics = max(2, X.shape[1])
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method="batch",
                                    max_iter=30, random_state=seed)
    doc_topic = lda.fit_transform(X)
    vocab = np.array(vec.get_feature_names_out())
    labels = {}
    for k, comp in enumerate(lda.components_):
        words = vocab[comp.argsort()[::-1]]
        picked = []
        for w in words:  # avoid listing both "matcha" and "matcha latte"
            if not any(w in p or p in w for p in picked):
                picked.append(w)
            if len(picked) == top_words:
                break
        labels[k] = picked
    df = df.copy()
    df["topic"] = doc_topic.argmax(axis=1)
    df["topic_conf"] = doc_topic.max(axis=1)
    return df, labels


# --------------------------------------------------------------------------
# 6. TIME SERIES + FORECASTING
# --------------------------------------------------------------------------
def build_series(df, topic, freq="W"):
    d = df[df["topic"] == topic].copy()
    d["w"] = 1 + np.log1p(d["engagement"].clip(lower=0))  # engagement-weighted mention
    s = d.set_index("date")["w"].resample(freq).sum()
    full = pd.date_range(df["date"].min().floor("D"), df["date"].max().ceil("D"), freq=freq)
    return s.reindex(s.index.union(full)).fillna(0.0).sort_index()


def lstm_forecast(values, horizon=4, window=4, epochs=200):
    """Small LSTM; returns None if TensorFlow is missing or the series is too short."""
    if len(values) < window + 6:
        return None
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        return None
    v = np.asarray(values, dtype="float32")
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-6:
        return np.repeat(v[-1], horizon)
    sc = (v - lo) / (hi - lo)
    X = np.array([sc[i:i + window] for i in range(len(sc) - window)])[..., None]
    y = sc[window:]
    tf.random.set_seed(0)
    model = models.Sequential([layers.Input((window, 1)), layers.LSTM(16), layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=8, verbose=0)
    hist, preds = list(sc[-window:]), []
    for _ in range(horizon):
        p = float(model.predict(np.array(hist[-window:])[None, :, None], verbose=0)[0, 0])
        preds.append(p)
        hist.append(p)
    return np.clip(np.array(preds) * (hi - lo) + lo, 0, None)


def linear_forecast(values, horizon=4, lookback=8):
    y = np.asarray(values[-lookback:], dtype=float)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1) if len(y) > 1 else (0.0, y[-1] if len(y) else 0.0)
    fut = intercept + slope * np.arange(len(y), len(y) + horizon)
    return np.clip(fut, 0, None)


def analyse_trend(series, horizon=4):
    vals = series.values
    fc = lstm_forecast(vals, horizon)
    model_used = "LSTM"
    if fc is None:
        fc, model_used = linear_forecast(vals, horizon), "linear"
    recent = vals[-3:].mean() if len(vals) >= 3 else vals.mean()
    prev = vals[-6:-3].mean() if len(vals) >= 6 else max(vals[:-3].mean(), 1e-6) if len(vals) > 3 else 1e-6
    growth = (recent - prev) / max(prev, 1e-6)
    traj = np.concatenate([[vals[-1]], fc])
    peak_idx = int(np.argmax(traj))
    if growth > 0.25 and peak_idx > 0:
        status, weeks_to_peak = "rising", peak_idx
    elif growth > 0.25:
        status, weeks_to_peak = "rising (near peak)", 0
    elif growth < -0.25:
        status, weeks_to_peak = "declining", 0
    elif vals[-3:].mean() >= 0.8 * vals.max() and vals.max() > 0:
        status, weeks_to_peak = "at peak", 0
    else:
        status, weeks_to_peak = "stable", 0
    return {
        "recent_level": float(recent), "growth_pct": float(growth * 100),
        "forecast": [round(float(x), 2) for x in fc], "status": status,
        "weeks_to_peak": weeks_to_peak, "forecast_model": model_used,
    }


# --------------------------------------------------------------------------
# 7. TREND OBJECTS (strength / local relevance / urgency)
# --------------------------------------------------------------------------
def build_trends(df, labels, horizon, min_docs):
    out = []
    for k, words in labels.items():
        sub = df[df["topic"] == k]
        if len(sub) < min_docs:
            continue
        series = build_series(df, k)
        a = analyse_trend(series, horizon)
        out.append({
            "topic_id": int(k), "trend": ", ".join(words[:4]), "keywords": words,
            "mentions": int(len(sub)), "avg_engagement": round(float(sub["engagement"].mean()), 1),
            "local_share": round(float(sub["is_local"].mean()), 2),
            "sources": sub["source"].value_counts().to_dict(),
            "example": sub.sort_values("engagement", ascending=False)["text"].iloc[0][:140],
            **a,
        })
    if not out:
        return out
    vol = np.array([t["recent_level"] for t in out])
    gr = np.array([t["growth_pct"] for t in out])
    vol_n = vol / vol.max() if vol.max() > 0 else vol
    gr_n = np.clip(gr / 100.0, -1, 1)
    for t, v, g in zip(out, vol_n, gr_n):
        t["trend_strength"] = round(float(v), 3)
        t["urgency"] = round(float(np.clip((g + 1) / 2, 0, 1)), 3)  # 0 = dying, 1 = surging
        # blended ranking: current size, momentum, local relevance
        t["score"] = round(0.4 * t["trend_strength"] + 0.4 * t["urgency"]
                           + 0.2 * t["local_share"], 3)
    return sorted(out, key=lambda t: t["score"], reverse=True)


# --------------------------------------------------------------------------
# 8. MAIN
# --------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="AdVantage OS Trend Engine")
    ap.add_argument("--sme", default="cafe", help="SME type (see --list-smes)")
    ap.add_argument("--city", default="Bengaluru")
    ap.add_argument("--days", type=int, default=90, help="history window in days")
    ap.add_argument("--sources", default="reddit,youtube,trends",
                    help="comma list of: reddit,youtube,trends")
    ap.add_argument("--topics", type=int, default=6, help="number of LDA topics")
    ap.add_argument("--horizon", type=int, default=4, help="forecast horizon in weeks")
    ap.add_argument("--per-query", type=int, default=100, help="max posts/videos per query")
    ap.add_argument("--local-only", action="store_true", help="keep only locally-tagged items")
    ap.add_argument("--comments", action="store_true", help="also fetch YouTube comments (uses quota)")
    ap.add_argument("--min-docs", type=int, default=8, help="drop topics smaller than this")
    ap.add_argument("--out", default="trend_output", help="output folder")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (no API keys)")
    ap.add_argument("--list-smes", action="store_true")
    args = ap.parse_args()

def discover_trends(
    sme: str = "cafe",
    city: str = "Bengaluru",
    days: int = 90,
    sources = ("reddit", "youtube", "trends"),
    topics: int = 6,
    horizon: int = 4,
    per_query: int = 50,
    local_only: bool = False,
    comments: bool = False,
    min_docs: int = 8,
    demo: bool = False,
    save_files: bool = False,
    out_dir: str = "trend_output",
) -> dict:
    """Discovers trends in memory. Does NOT write files unless save_files=True."""
    if sme not in SME_PROFILES:
        raise ValueError(f"Unknown SME '{sme}'. Options: {', '.join(SME_PROFILES)}")

    profile = SME_PROFILES[sme]
    city_key = city.lower()
    local_terms = CITY_LOCAL_TERMS.get(city_key, [city_key])
    geo = CITY_TRENDS_GEO.get(city_key, "IN")
    if isinstance(sources, str):
        sources_set = {s.strip() for s in sources.split(",")}
    else:
        sources_set = set(sources)

    # ---- fetch -----------------------------------------------------------
    rows, interest, rising = [], pd.DataFrame(), pd.DataFrame()
    if demo:
        print("[demo] generating synthetic data")
        rows = make_demo_data(days=max(days, 90))
    else:
        if "reddit" in sources_set:
            rows += fetch_reddit(profile, city, days, per_query, local_terms)
        if "youtube" in sources_set:
            rows += fetch_youtube(profile, city, days, per_query, local_terms, comments)
        if "trends" in sources_set:
            interest, rising = fetch_google_trends(profile, geo, days)

    rising_list = []
    if not rising.empty:
        rising_list = rising.to_dict(orient="records")

    if not rows:
        print("No Reddit/YouTube data collected.")
        return {
            "sme": sme,
            "city": city,
            "generated": datetime.now().isoformat(),
            "trends": [],
            "rising_queries": rising_list,
            "total_documents": 0,
            "message": "No Reddit/YouTube data collected."
        }

    df = pd.DataFrame(rows).drop_duplicates(subset=["source", "id"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if local_only:
        df = df[df["is_local"]]
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.split().str.len() >= 3].reset_index(drop=True)
    print(f"\n{len(df)} documents after cleaning{' (local only)' if local_only else ''}")

    if len(df) < 5:
        return {
            "sme": sme,
            "city": city,
            "generated": datetime.now().isoformat(),
            "trends": [],
            "rising_queries": rising_list,
            "total_documents": int(len(df)),
            "message": "Not enough documents to discover topics."
        }

    actual_topics = min(topics, max(2, len(df) // 3))
    df, labels = run_lda(df, actual_topics)
    trends = build_trends(df, labels, horizon, min_docs)

    if save_files:
        os.makedirs(out_dir, exist_ok=True)
        df.drop(columns=["clean"]).to_csv(os.path.join(out_dir, "raw_documents.csv"), index=False)
        with open(os.path.join(out_dir, "trends.json"), "w", encoding="utf-8") as f:
            json.dump({"sme": sme, "city": city, "generated": datetime.now().isoformat(),
                       "trends": trends}, f, indent=2, ensure_ascii=False, default=str)
        pd.DataFrame([{k: v for k, v in t.items() if k not in ("forecast", "sources")}
                      | {"forecast_next_weeks": t["forecast"]} for t in trends]
                     ).to_csv(os.path.join(out_dir, "trends.csv"), index=False)
        if not interest.empty:
            interest.to_csv(os.path.join(out_dir, "google_trends_interest.csv"))
        if not rising.empty:
            _report_rising(rising, out_dir)
        print(f"Saved results to ./{out_dir}/")

    return {
        "sme": sme,
        "city": city,
        "generated": datetime.now().isoformat(),
        "trends": trends,
        "rising_queries": rising_list,
        "total_documents": int(len(df))
    }


def main():
    ap = argparse.ArgumentParser(description="AdVantage OS Trend Engine")
    ap.add_argument("--sme", default="cafe", help="SME type (see --list-smes)")
    ap.add_argument("--city", default="Bengaluru")
    ap.add_argument("--days", type=int, default=90, help="history window in days")
    ap.add_argument("--sources", default="reddit,youtube,trends",
                    help="comma list of: reddit,youtube,trends")
    ap.add_argument("--topics", type=int, default=6, help="number of LDA topics")
    ap.add_argument("--horizon", type=int, default=4, help="forecast horizon in weeks")
    ap.add_argument("--per-query", type=int, default=100, help="max posts/videos per query")
    ap.add_argument("--local-only", action="store_true", help="keep only locally-tagged items")
    ap.add_argument("--comments", action="store_true", help="also fetch YouTube comments (uses quota)")
    ap.add_argument("--min-docs", type=int, default=8, help="drop topics smaller than this")
    ap.add_argument("--out", default="trend_output", help="output folder")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (no API keys)")
    ap.add_argument("--list-smes", action="store_true")
    args = ap.parse_args()

    if args.list_smes:
        print("Available SME types:", ", ".join(SME_PROFILES))
        return
    if args.sme not in SME_PROFILES:
        sys.exit(f"Unknown SME '{args.sme}'. Options: {', '.join(SME_PROFILES)}")

    res = discover_trends(
        sme=args.sme,
        city=args.city,
        days=args.days,
        sources=args.sources,
        topics=args.topics,
        horizon=args.horizon,
        per_query=args.per_query,
        local_only=args.local_only,
        comments=args.comments,
        min_docs=args.min_docs,
        demo=args.demo,
        save_files=True,
        out_dir=args.out
    )

    trends = res.get("trends", [])
    print(f"\n=== TRENDING TOPICS: {args.sme.upper()} in {args.city} (last {args.days} days) ===\n")
    for i, t in enumerate(trends, 1):
        arrow = {"rising": "UP  ", "rising (near peak)": "UP  ", "at peak": "PEAK",
                 "declining": "DOWN", "stable": "FLAT"}[t["status"]]
        peak = f", peak in ~{t['weeks_to_peak']}w" if t["weeks_to_peak"] else ""
        print(f"{i:>2}. [{arrow}] {t['trend']}")
        print(f"      status={t['status']}{peak} | growth={t['growth_pct']:+.0f}% | "
              f"mentions={t['mentions']} | local={t['local_share']:.0%} | "
              f"score={t['score']} | model={t['forecast_model']}")
        safe_ex = t['example'].encode(sys.stdout.encoding or 'ascii', errors='replace').decode(sys.stdout.encoding or 'ascii')
        print(f"      e.g. \"{safe_ex}...\"\n")


def _report_rising(rising, out_dir):
    rising.to_csv(os.path.join(out_dir, "google_trends_rising_queries.csv"), index=False)
    print("\n=== GOOGLE TRENDS: RISING RELATED QUERIES ===")
    top = rising.sort_values("growth_pct", ascending=False).head(15)
    for _, r in top.iterrows():
        print(f"  {r['rising_query']:<40} (+{r['growth_pct']}%)  [seed: {r['seed']}]")
    print()


# ── FastAPI Router & SME Trend Analysis Handler ──────────────────────────────
from fastapi import APIRouter, HTTPException
from pathlib import Path

router = APIRouter(tags=["Trend Analysis (AI-Powered)"])


def process_trend_analysis(
    sme: str,
    city: str = "Bengaluru",
    days: int = 90,
    topics: int = 6,
    demo: bool = False,
    sources: list[str] = None,
    refresh: bool = False,
):
    try:
        from trend_predictor.trend_db import get_cached_trends, save_trends_to_db
        from trend_predictor.trend_analyzer_cohere import analyze_trends
    except ImportError:
        from trend_db import get_cached_trends, save_trends_to_db
        from trend_analyzer_cohere import analyze_trends

    if sources is None:
        sources = ["reddit", "youtube", "trends"]
    sme_clean = sme.strip().lower().replace(" ", "_").replace("-", "_")
    sme_clean = SME_ALIASES.get(sme_clean, sme_clean)
    if sme_clean not in SME_PROFILES:
        # fuzzy match or default
        matched = next((k for k in SME_PROFILES if k in sme_clean or sme_clean in k), None)
        if matched:
            sme_clean = matched
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown SME '{sme}'. Allowed categories: {list(SME_PROFILES.keys())}",
            )

    # 1. Check Database First ({sme}_trends table)
    if not refresh:
        cached_trends = get_cached_trends(sme=sme_clean, city=city)
        if cached_trends is not None:
            return cached_trends

        _here_dir = Path(__file__).resolve().parent
        for out_dir in [_here_dir / "trend_output", _here_dir.parent / "trend_output", Path("trend_output")]:
            out_file = out_dir / "trending_items.json"
            if out_file.is_file():
                try:
                    with open(out_file, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                    if file_data.get("sme", "").lower() == sme_clean:
                        final_payload = {
                            "status": "success",
                            "sme": sme_clean,
                            "city": city,
                            "generated": file_data.get("generated"),
                            "total_documents_analyzed": file_data.get("total_documents_analyzed", 0),
                            "total_trending_items": len(file_data.get("items", [])),
                            "trending_items": file_data.get("items", []),
                            "topic_clusters": [],
                            "rising_searches": [],
                            "from_file": True,
                        }
                        record_id = save_trends_to_db(sme=sme_clean, city=city, trend_output=final_payload)
                        final_payload["database_saved"] = bool(record_id)
                        final_payload["database_table"] = f"{sme_clean}_trends"
                        final_payload["database_record_id"] = record_id
                        print(f"[trend_db] Imported '{out_file}' into '{sme_clean}_trends' table (ID: {record_id})")
                        return final_payload
                except Exception as err:
                    print(f"[trend_db] Error checking {out_file}: {err}")

    # 2. Run Trend Engine in-memory
    try:
        trends_raw = discover_trends(
            sme=sme_clean,
            city=city,
            days=days,
            sources=tuple(sources),
            topics=topics,
            demo=demo,
            save_files=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Trend engine failed: {exc}")

    # 3. Feed in-memory results directly to Cohere LLM
    try:
        trending_items = analyze_trends(
            trends_data=trends_raw,
            rising_queries=trends_raw.get("rising_queries", []),
        )
    except Exception as exc:
        return {
            "status": "partial_success",
            "sme": sme_clean,
            "city": city,
            "error_cohere": str(exc),
            "total_documents_analyzed": trends_raw.get("total_documents", 0),
            "trending_items": [],
            "topic_clusters": trends_raw.get("trends", []),
            "rising_searches": trends_raw.get("rising_queries", []),
        }

    # 4. Build output payload
    final_output = {
        "status": "success",
        "sme": sme_clean,
        "city": city,
        "generated": trends_raw.get("generated"),
        "total_documents_analyzed": trends_raw.get("total_documents", 0),
        "total_trending_items": len(trending_items),
        "trending_items": trending_items,
        "topic_clusters": trends_raw.get("trends", []),
        "rising_searches": trends_raw.get("rising_queries", []),
    }

    # 5. Save output to database table {sme}_trends
    record_id = save_trends_to_db(sme=sme_clean, city=city, trend_output=final_output)
    final_output["database_saved"] = bool(record_id)
    final_output["database_table"] = f"{sme_clean}_trends"
    final_output["database_record_id"] = record_id

    return final_output


@router.get("/api/trends/analyze/{sme}")
@router.get("/trends/analyze/{sme}", include_in_schema=False)
def analyze_sme_trends(
    sme: str,
    city: str = "Bengaluru",
    days: int = 90,
    topics: int = 6,
    demo: bool = False,
    refresh: bool = False,
):
    """
    AI-Powered SME Trend Intelligence:
      1. Checks if output is already in database table '{sme}_trends'.
      2. Returns cached result or scrapes/analyzes live and stores to DB.
    """
    return process_trend_analysis(
        sme=sme, city=city, days=days, topics=topics, demo=demo, refresh=refresh
    )


if __name__ == "__main__":
    main()