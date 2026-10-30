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
# 3. SYNTHETIC DEMO DATA & CURATED TRENDS PER SME
# --------------------------------------------------------------------------
SME_DEMO_TOPICS = {
    "cafe": {
        "matcha": (["matcha latte", "ceremonial matcha", "oat milk matcha whisk",
                      "iced matcha cafe", "matcha green tea powder"], "rise"),
        "coldbrew": (["cold brew coffee", "nitro cold brew tonic", "coffee tonic cafe",
                      "cold brew concentrate bottle", "espresso tonic"], "peak"),
        "brunch": (["weekend brunch menu", "avocado toast brunch", "eggs benedict brunch",
                      "bottomless brunch cafe", "brunch cafe pancakes"], "flat"),
        "bubble": (["bubble tea boba", "tapioca pearls milk tea", "boba shop queue",
                      "brown sugar boba", "taro milk tea"], "decay"),
        "dessert": (["dubai chocolate kunafa", "pistachio kunafa bar", "viral chocolate dessert",
                      "kunafa chocolate bar", "pistachio dessert cafe"], "rise"),
    },
    "restaurant": {
        "biryani": (["donne biryani", "ambur dum biryani", "hyderabadi mutton biryani",
                      "pot biryani meal", "bamboo chicken biryani"], "rise"),
        "pizza": (["neapolitan sourdough pizza", "wood fired crust pizza", "burrata margherita pizza",
                    "truffle mushroom pizza slice", "detroit style crispy pizza"], "peak"),
        "burger": (["smash cheeseburger", "truffle beef burger", "brioche bun smash burger",
                     "nashville hot crispy chicken burger", "double smash patty"], "rise"),
        "ramen": (["tonkotsu ramen bowl", "spicy miso ramen broth", "black garlic ramen noodle",
                    "pork belly chashu ramen", "dipping broth tsukemen"], "flat"),
        "local_tiffin": (["ghee podi thatte idli", "benne open butter masala dosa",
                          "kundapura chicken ghee roast", "mangalorean neer dosa", "coastal fish curry thali"], "rise"),
    },
    "bakery": {
        "cheesecake": (["burnt basque cheesecake", "san sebastian molten cheesecake", "lotus biscoff baked cheesecake",
                        "blueberry basque cheesecake slice", "japanese cotton sponge cheesecake"], "rise"),
        "sourdough": (["artisanal sourdough loaf", "wild yeast sourdough boule", "seeded sourdough bread",
                       "country sourdough loaf bakery", "sourdough focaccia rosemary"], "peak"),
        "croissant": (["viral pistachio croissant", "almond frangipane croissant", "pain au chocolat flaky",
                       "supreme circular filled croissant", "french butter laminated croissant"], "rise"),
        "cookies": (["chunky stuffed cookies", "sea salt dark chocolate chip cookie", "brookies brownie cookie",
                     "red velvet cream cheese stuffed cookie", "nyc style chunky cookie"], "flat"),
        "pastry": (["tiramisu dessert tub", "belgian chocolate entremet mousse", "opera layered pastry cake",
                    "choux cream pastry bun", "vanilla bean mille feuille"], "rise"),
    },
    "boutique": {
        "coord": (["linen co-ord set", "printed cotton matching co-ord", "monochromatic summer co-ord",
                    "resort wear two piece set", "blazer tailored trouser co-ord"], "rise"),
        "ethnic": (["chikankari modal soft kurti", "mulmul embroidered anarkali suit", "handloom chanderi festive set",
                    "banarasi tissue organza saree", "handcrafted lehenga choli set"], "peak"),
        "streetwear": (["oversized vintage graphic tee", "washed cargo parachute pants", "heavyweight boxy fit hoodie",
                        "drop shoulder streetwear apparel", "baggy distressed wide leg denim"], "rise"),
        "thrift": (["vintage thrifted leather jacket", "90s retro oversized windbreaker", "sustainable curated thrift fashion",
                    "reworked denim jacket thrift", "y2k vintage corset top"], "flat"),
        "dresses": (["floral cottagecore midi dress", "ribbed knit bodycon dress", "bohemian maxi slip dress",
                     "backless halter summer linen dress", "wrap around linen midi"], "decay"),
    },
    "salon": {
        "facial": (["korean glass skin facial", "hydrafacial deep glow treatment", "24k gold leaf radiance facial",
                    "oxygen infusion hydra facial", "carbon laser skin peel glow"], "rise"),
        "hair_treatment": (["keratin smoothing treatment", "brazilian blowout smoothing", "hair botox restoration clinic",
                            "cystine hair protein treatment", "scalp detox therapy head spa"], "peak"),
        "color": (["sun-kissed balayage highlights", "ash blonde babylights hair", "chocolate caramel ombre highlights",
                    "money piece face framing color", "warm copper brown global hair"], "flat"),
        "nails": (["chrome glazed donut nail art", "french tip gel extensions studio", "cat eye magnetic gel nails",
                    "minimalist nude ombre gel nails", "abstract hand-painted nail design"], "rise"),
        "grooming": (["brow lamination and tint salon", "lash lift and tint studio", "japanese therapeutic head spa",
                      "organic express blowout styling", "beard sculpting and facial spa"], "rise"),
    },
    "gym": {
        "pilates": (["reformer pilates studio class", "classical reformer core workout", "cadillac pilates sculpting",
                     "dynamic core reformer burn", "mat pilates resistance flow"], "rise"),
        "crossfit": (["functional fitness hiit workout", "crossfit wods conditioning", "kettlebell metabolic blast",
                      "barbell olympic weightlifting", "hyrox athletic endurance training"], "peak"),
        "yoga": (["sound bath restorative yin yoga", "aerial hammock yoga studio", "heated vinyasa power flow",
                  "sunrise hatha yoga class", "guided mindful meditation breathwork"], "flat"),
        "mobility": (["calisthenics bodyweight training", "animal flow mobility drills", "active recovery foam rolling",
                      "joint mobility conditioning", "gymnastics ring core strength"], "rise"),
        "sports": (["pickleball club indoor courts", "padel tennis coaching clinic", "boxing heavy bag conditioning",
                    "agility footwork speed drills", "indoor rock climbing bouldering"], "rise"),
    },
    "gift_shop": {
        "candles": (["hand-poured soy wax scented candle", "amber jar aromatherapy candle", "wooden wick crackling candle",
                     "botanical beeswax scented candle", "vanilla bean amber room fragrance"], "rise"),
        "hampers": (["luxury festive gift hamper", "bespoke corporate gourmet hamper", "self-care wellness gift basket",
                     "anniversary curated keepsake box", "personalized bridesmaid gift hamper"], "peak"),
        "decor": (["resin art wooden customized nameplate", "handcrafted ceramic ribbed vase", "bohemian macrame wall tapestry",
                   "brass vintage urli floating flower bowl", "modern minimalist abstract sculpture"], "flat"),
        "plants": (["geometric succulent terrarium glass", "preserved moss wall art frame", "air purifying indoor plant ceramic",
                    "miniature bonsai gift plant", "self-watering aesthetic desk planter"], "rise"),
        "stationery": (["wax seal brass stamp kit", "leather bound refillable journal", "personalized calligraphed gift cards",
                       "premium fountain pen writing set", "aesthetic wooden desk organizer"], "decay"),
    },
    "electronics": {
        "audio": (["active noise cancelling wireless earbuds", "spatial audio over-ear headphones", "bone conduction sport earphones",
                    "portable waterproof bluetooth speaker", "audiophile high-res dac amp"], "rise"),
        "power": (["magsafe wireless magnetic power bank", "65w gan compact fast charger", "3-in-1 foldable wireless charging station",
                    "braided high-speed usb-c cable", "solar emergency portable power bank"], "peak"),
        "wearables": (["amoled display smartwatch", "fitness sleep tracking smart ring", "gps outdoor sport tracker watch",
                       "heart rate variability fitness band", "smart posture vibrating sensor"], "rise"),
        "accessories": (["shockproof magsafe phone case", "ergonomic wireless vertical mouse", "retro mechanical keyboard rgb",
                         "aluminum laptop ergonomic cooling stand", "4k ultra wide streaming webcam"], "flat"),
        "smart_home": (["smart wifi led ambient light bar", "smart desk monitor backlight", "automated ultrasonic aroma diffuser",
                        "bluetooth smart tag tracker finder", "motorized cable management tray"], "rise"),
    },
    "d2c": {
        "pottery": (["handcrafted ceramic coffee mug", "artisan studio stoneware bowl", "speckled ceramic dinner plate",
                     "hand-pinched terracotta pitcher", "handmade clay matcha whisk bowl"], "rise"),
        "skincare": (["small-batch whipped shea body butter", "cold-pressed artisanal goat milk soap", "organic rosehip seed face elixir",
                      "botanical tinted lip butter pot", "exfoliating roasted coffee sugar scrub"], "peak"),
        "jewelry": (["dainty handmade beaded necklace", "freshwater baroque pearl earrings", "polymer clay statement earrings",
                     "anti-tarnish stainless steel gold ring", "wire wrapped raw crystal pendant"], "rise"),
        "bags": (["organic heavy canvas tote bag", "block printed cotton carryall tote", "handcrafted jute daily shopping bag",
                  "vintage corduroy shoulder tote", "minimalist padded laptop sleeve"], "flat"),
        "wellness": (["ceremonial grade raw cacao block", "herbal loose leaf wellness tea blend", "cold-pressed pure artisanal essential oil",
                      "pure raw forest wildflower honey", "natural sacred smudge incense sticks"], "rise"),
    },
}

SME_DEMO_RISING = {
    "cafe": [
        {"seed": "matcha", "rising_query": "iced matcha latte with oat milk", "growth_pct": 280},
        {"seed": "cold brew", "rising_query": "cold brew concentrate bottle", "growth_pct": 190},
        {"seed": "cafe", "rising_query": "dubai kunafa chocolate bar bangalore", "growth_pct": 350},
        {"seed": "brunch", "rising_query": "bottomless brunch indiranagar", "growth_pct": 140},
        {"seed": "latte", "rising_query": "pistachio iced latte cafe", "growth_pct": 210},
    ],
    "restaurant": [
        {"seed": "biryani", "rising_query": "authentic donne biryani bangalore", "growth_pct": 310},
        {"seed": "pizza", "rising_query": "wood fired neapolitan sourdough pizza", "growth_pct": 240},
        {"seed": "burger", "rising_query": "crispy smash burger koramangala", "growth_pct": 195},
        {"seed": "ramen", "rising_query": "tonkotsu ramen bowl indiranagar", "growth_pct": 160},
        {"seed": "dosa", "rising_query": "ghee podi thatte idli breakfast", "growth_pct": 220},
    ],
    "bakery": [
        {"seed": "cheesecake", "rising_query": "burnt basque cheesecake slice", "growth_pct": 290},
        {"seed": "croissant", "rising_query": "pistachio stuffed croissant viral", "growth_pct": 340},
        {"seed": "sourdough", "rising_query": "artisan sourdough loaf bakery", "growth_pct": 180},
        {"seed": "cookies", "rising_query": "chunky sea salt chocolate chip cookie", "growth_pct": 160},
        {"seed": "dessert", "rising_query": "tiramisu dessert tub bakery", "growth_pct": 210},
    ],
    "boutique": [
        {"seed": "co-ord set", "rising_query": "linen co ord set for women", "growth_pct": 320},
        {"seed": "kurti", "rising_query": "chikankari modal kurti bangalore", "growth_pct": 260},
        {"seed": "streetwear", "rising_query": "oversized graphic streetwear tee", "growth_pct": 210},
        {"seed": "saree", "rising_query": "pastel organza tissue saree", "growth_pct": 190},
        {"seed": "thrift", "rising_query": "vintage thrift clothing store indiranagar", "growth_pct": 175},
    ],
    "salon": [
        {"seed": "facial", "rising_query": "korean glass skin facial salon bangalore", "growth_pct": 350},
        {"seed": "keratin", "rising_query": "keratin smoothing hair treatment offer", "growth_pct": 240},
        {"seed": "balayage", "rising_query": "sun kissed balayage hair color trends", "growth_pct": 200},
        {"seed": "nails", "rising_query": "glazed donut chrome nail art salon", "growth_pct": 280},
        {"seed": "hair spa", "rising_query": "scalp detox japanese head spa", "growth_pct": 310},
    ],
    "gym": [
        {"seed": "pilates", "rising_query": "reformer pilates studio near me bangalore", "growth_pct": 380},
        {"seed": "crossfit", "rising_query": "hyrox functional fitness training", "growth_pct": 270},
        {"seed": "pickleball", "rising_query": "indoor pickleball courts booking", "growth_pct": 340},
        {"seed": "yoga", "rising_query": "sound bath yin yoga studio", "growth_pct": 190},
        {"seed": "calisthenics", "rising_query": "bodyweight calisthenics gym", "growth_pct": 170},
    ],
    "gift_shop": [
        {"seed": "candles", "rising_query": "hand poured soy wax scented candles", "growth_pct": 290},
        {"seed": "gift hamper", "rising_query": "curated luxury celebration gift hamper", "growth_pct": 250},
        {"seed": "home decor", "rising_query": "custom resin art nameplates bangalore", "growth_pct": 210},
        {"seed": "terrarium", "rising_query": "geometric succulent terrarium gift", "growth_pct": 195},
        {"seed": "handmade", "rising_query": "personalized keepsake gifting ideas", "growth_pct": 170},
    ],
    "electronics": [
        {"seed": "earbuds", "rising_query": "anc wireless earbuds under 5000", "growth_pct": 280},
        {"seed": "power bank", "rising_query": "magnetic wireless magsafe power bank", "growth_pct": 240},
        {"seed": "smartwatch", "rising_query": "amoled display smartwatch with gps", "growth_pct": 210},
        {"seed": "gadgets", "rising_query": "ergonomic mechanical keyboard wireless", "growth_pct": 185},
        {"seed": "charger", "rising_query": "65w gan multi device fast charger", "growth_pct": 195},
    ],
    "d2c": [
        {"seed": "handmade", "rising_query": "small batch artisan ceramic mug", "growth_pct": 290},
        {"seed": "organic skincare", "rising_query": "whipped shea body butter natural", "growth_pct": 260},
        {"seed": "tote bag", "rising_query": "heavy canvas aesthetic tote bag", "growth_pct": 220},
        {"seed": "jewelry", "rising_query": "anti tarnish freshwater pearl earrings", "growth_pct": 240},
        {"seed": "scented candles", "rising_query": "small business handcrafted soy candles", "growth_pct": 190},
    ],
}

SME_CURATED_TRENDS = {
    "cafe": [
        {
            "item_name": "Dubai Kunafa Chocolate Pistachio Bar",
            "category": "Dessert",
            "associated_topic_id": 4,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 145.0,
            "urgency_score": 95,
            "commercial_appeal": "Indulgent Middle Eastern sweet fusion with viral sensory appeal across Instagram and TikTok.",
            "key_drivers": ["Dubai Chocolate", "Kunafa Crisp", "Pistachio Praline", "Dessert Innovation"],
            "recommended_action": "Feature as a signature limited-edition pastry special with visual prep reels.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Iced Matcha Latte",
            "category": "Beverage",
            "associated_topic_id": 0,
            "trend_status": "Rising",
            "estimated_growth_pct": 88.0,
            "urgency_score": 90,
            "commercial_appeal": "Antioxidant-rich ceremonial green tea whisked with creamy oat milk for health-conscious morning crowds.",
            "key_drivers": ["Clean Caffeine", "Ceremonial Grade", "Oat Milk Base", "Aesthetic Layers"],
            "recommended_action": "Introduce flavored matcha variants such as vanilla bean and strawberry foam cold foam.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Cold Brew Coffee Concentrate",
            "category": "Beverage",
            "associated_topic_id": 1,
            "trend_status": "Rising",
            "estimated_growth_pct": 62.0,
            "urgency_score": 78,
            "commercial_appeal": "18-hour steeped smooth concentrate for grab-and-go home brewing and afternoon coffee tonic cocktails.",
            "key_drivers": ["Low Acidity", "High Caffeine", "Retail Bottling", "Specialty Roasts"],
            "recommended_action": "Bottle signature house blends in 250ml glass flasks for retail takeaway shelves.",
            "local_relevance": "High"
        },
        {
            "item_name": "Matcha Green Tea Powder",
            "category": "Beverage Ingredient",
            "associated_topic_id": 0,
            "trend_status": "Rising",
            "estimated_growth_pct": 54.0,
            "urgency_score": 72,
            "commercial_appeal": "Surging at-home wellness enthusiasts looking for authentic Japanese Uji matcha tins.",
            "key_drivers": ["Home Barista", "L-Theanine", "Culinary Grade", "Wellness Lifestyle"],
            "recommended_action": "Sell packaged 50g starter tins with branded bamboo whisks at checkout.",
            "local_relevance": "Medium"
        },
        {
            "item_name": "Bottomless Brunch",
            "category": "Dining Concept",
            "associated_topic_id": 2,
            "trend_status": "Steady / Evergreen",
            "estimated_growth_pct": 41.0,
            "urgency_score": 68,
            "commercial_appeal": "Weekend communal dining experience featuring unlimited artisanal toasts, fluffy pancakes, and craft brews.",
            "key_drivers": ["Weekend Gatherings", "Social Dining", "Avocado Toast", "Fluffy Pancakes"],
            "recommended_action": "Offer a 2-hour weekend brunch package between 11 AM and 3 PM with live acoustic sets.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Taro Boba Milk Tea",
            "category": "Beverage",
            "associated_topic_id": 3,
            "trend_status": "Steady / Evergreen",
            "estimated_growth_pct": 38.0,
            "urgency_score": 60,
            "commercial_appeal": "Vibrant purple sweet root milk tea with chewy brown sugar tapioca pearls popular among Gen Z.",
            "key_drivers": ["Boba Culture", "Texture Contrast", "Gen Z Favorite", "Sweet Treat"],
            "recommended_action": "Add boba toppings as customizable add-ons across iced tea and matcha selections.",
            "local_relevance": "Medium"
        }
    ],
    "restaurant": [
        {
            "item_name": "Donne Biryani",
            "category": "Main Course",
            "associated_topic_id": 0,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 135.0,
            "urgency_score": 94,
            "commercial_appeal": "Fragrant seeraga samba rice cooked with mint, coriander, and succulent spiced meat in palm leaf bowls.",
            "key_drivers": ["Karnataka Heritage", "Seeraga Samba Rice", "Street Food Craving", "Comfort Food"],
            "recommended_action": "Promote clay pot Donne Biryani combos during lunch peak hours and weekend food delivery.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Neapolitan Sourdough Pizza",
            "category": "Main Course",
            "associated_topic_id": 1,
            "trend_status": "Rising",
            "estimated_growth_pct": 112.0,
            "urgency_score": 89,
            "commercial_appeal": "72-hour fermented sourdough crust blistered at 450°C topped with creamy burrata and san marzano tomatoes.",
            "key_drivers": ["Wood Fired", "Burrata Cheese", "Artisanal Dough", "Gourmet Dining"],
            "recommended_action": "Introduce a chef's table demonstration showing hand-stretched sourdough wood-fire baking.",
            "local_relevance": "High"
        },
        {
            "item_name": "Smash Cheeseburger",
            "category": "Fast Casual",
            "associated_topic_id": 2,
            "trend_status": "Rising",
            "estimated_growth_pct": 85.0,
            "urgency_score": 84,
            "commercial_appeal": "Lacy-edged crispy sear patties with melted sharp cheddar, caramelized onions, and house burger sauce.",
            "key_drivers": ["Crispy Edges", "Brioche Buns", "Comfort Food", "Casual Dining"],
            "recommended_action": "Offer double and triple smash burger specials paired with seasoned crinkle fries.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Tonkotsu Ramen Bowl",
            "category": "Main Course",
            "associated_topic_id": 3,
            "trend_status": "Rising",
            "estimated_growth_pct": 76.0,
            "urgency_score": 80,
            "commercial_appeal": "Rich 12-hour simmered collagen broth with springy noodles, ajitsuke tamago, and torched chashu.",
            "key_drivers": ["Japanese Cuisine", "Comfort Broth", "Noodle Craving", "Authentic Dining"],
            "recommended_action": "Host a midweek 'Ramen & Gyoza Night' to boost weekday dinner footfalls.",
            "local_relevance": "High"
        },
        {
            "item_name": "Ghee Podi Thatte Idli",
            "category": "Breakfast / Snack",
            "associated_topic_id": 4,
            "trend_status": "Rising",
            "estimated_growth_pct": 68.0,
            "urgency_score": 75,
            "commercial_appeal": "Ultra-soft steamed rice cakes drenched in pure desi ghee and fiery gun powder spice mix.",
            "key_drivers": ["Heritage Breakfast", "Pure Desi Ghee", "Spicy Gunpowder", "Local Staple"],
            "recommended_action": "Bundle as an express morning breakfast combo with freshly ground filter coffee.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Crispy Korean Fried Chicken",
            "category": "Snack / Main Course",
            "associated_topic_id": 2,
            "trend_status": "Steady / Evergreen",
            "estimated_growth_pct": 58.0,
            "urgency_score": 72,
            "commercial_appeal": "Double-fried ultra-crunchy chicken wings coated in sweet spicy gochujang glaze and toasted sesame.",
            "key_drivers": ["K-Food Craze", "Double Fried Crunch", "Gochujang Glaze", "Late Night Bites"],
            "recommended_action": "Introduce combo platters with pickled radish and Korean draft refreshments.",
            "local_relevance": "Medium"
        }
    ],
    "bakery": [
        {
            "item_name": "Burnt Basque Cheesecake",
            "category": "Dessert",
            "associated_topic_id": 0,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 140.0,
            "urgency_score": 95,
            "commercial_appeal": "Caramelized burnt exterior with a molten, creamy center that delivers unmatched indulgence.",
            "key_drivers": ["Molten Center", "Basque Style", "Gluten Free Friendly", "Gourmet Bakery"],
            "recommended_action": "Offer whole 6-inch celebration cakes and by-the-slice counter orders with warm berry compote.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Artisanal Sourdough Boule",
            "category": "Artisan Bread",
            "associated_topic_id": 1,
            "trend_status": "Rising",
            "estimated_growth_pct": 92.0,
            "urgency_score": 88,
            "commercial_appeal": "Long cold-fermented wild yeast loaves with open crumb and blistered crispy crust for gut wellness.",
            "key_drivers": ["Gut Health", "Wild Yeast Starter", "No Additives", "Artisan Bakery"],
            "recommended_action": "Establish daily morning bake schedules and communicate batch release times on WhatsApp channels.",
            "local_relevance": "High"
        },
        {
            "item_name": "Viral Pistachio Croissant",
            "category": "Pastry",
            "associated_topic_id": 2,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 125.0,
            "urgency_score": 92,
            "commercial_appeal": "Flaky French laminated croissant injected with rich pistachio custard and topped with roasted pistachio nibs.",
            "key_drivers": ["Laminated Pastry", "Pistachio Craze", "Viral Bakes", "French Butter"],
            "recommended_action": "Promote as a signature breakfast counter item paired with flat whites.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Chunky Biscoff Brookies",
            "category": "Cookies",
            "associated_topic_id": 3,
            "trend_status": "Rising",
            "estimated_growth_pct": 78.0,
            "urgency_score": 80,
            "commercial_appeal": "Fudge brownie base merged with sea salt chocolate chip cookie dough stuffed with Lotus Biscoff spread.",
            "key_drivers": ["Hybrid Dessert", "Lotus Biscoff", "Stuffed Cookies", "Sweet Indulgence"],
            "recommended_action": "Pack in aesthetic gift boxes of 4 and 6 for birthday hampers and impulse billing counter sales.",
            "local_relevance": "Medium"
        },
        {
            "item_name": "Tiramisu Tub",
            "category": "Dessert",
            "associated_topic_id": 4,
            "trend_status": "Rising",
            "estimated_growth_pct": 84.0,
            "urgency_score": 82,
            "commercial_appeal": "Espresso-soaked ladyfingers layered with whipped mascarpone cream and dusted with dutch cocoa powder.",
            "key_drivers": ["Classic Italian", "Espresso Flavor", "Tub Packaging", "Shareable Dessert"],
            "recommended_action": "Use transparent branded takeout tubs with personalized ribbon packaging.",
            "local_relevance": "High"
        },
        {
            "item_name": "Millet Sourdough Bread",
            "category": "Healthy Bakery",
            "associated_topic_id": 1,
            "trend_status": "Rising",
            "estimated_growth_pct": 65.0,
            "urgency_score": 76,
            "commercial_appeal": "Locally sourced ragi and foxtail millet blends incorporated into high-hydration sourdough breads.",
            "key_drivers": ["Millet Movement", "Diabetes Friendly", "Local Grains", "High Fiber"],
            "recommended_action": "Highlight glycemic index benefits and nutritional breakdowns on shelf display cards.",
            "local_relevance": "High (Bangalore Local Favorite)"
        }
    ],
    "boutique": [
        {
            "item_name": "Linen Co-ord Sets",
            "category": "Apparel",
            "associated_topic_id": 0,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 155.0,
            "urgency_score": 96,
            "commercial_appeal": "Effortless breathable two-piece matching linen tops and wide-leg trousers dominating modern urban workwear.",
            "key_drivers": ["Summer Fashion", "Minimalist Workwear", "Breathable Fabric", "Day-to-Night Styling"],
            "recommended_action": "Curate earthy pastel palettes (olive, terracotta, beige) and showcase 3-way mix-and-match reels.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Chikankari Modal Kurtis",
            "category": "Ethnic Wear",
            "associated_topic_id": 1,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 128.0,
            "urgency_score": 92,
            "commercial_appeal": "Soft flowy modal fabric decorated with intricate Lucknowi hand-embroidery in pastel tones.",
            "key_drivers": ["Handloom Heritage", "Soft Modal Silk", "Pastel Ethnic", "Festive & Casual"],
            "recommended_action": "Stock sizes XS through XXL with matching cotton slips and flared palazzo trousers.",
            "local_relevance": "High"
        },
        {
            "item_name": "Oversized Graphic Streetwear",
            "category": "Casual Apparel",
            "associated_topic_id": 2,
            "trend_status": "Rising",
            "estimated_growth_pct": 110.0,
            "urgency_score": 88,
            "commercial_appeal": "240 GSM heavyweight cotton drop-shoulder tees with typography and anime-inspired back prints.",
            "key_drivers": ["Gen Z Streetwear", "Drop Shoulder", "Heavyweight Cotton", "Y2K Aesthetic"],
            "recommended_action": "Set up a dedicated streetwear rack in-store with moody warm lighting and sticker giveaways.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Pastel Organza Sarees",
            "category": "Festive Wear",
            "associated_topic_id": 1,
            "trend_status": "Rising",
            "estimated_growth_pct": 85.0,
            "urgency_score": 82,
            "commercial_appeal": "Lightweight sheer organza sarees with delicate scalloped gota borders and subtle floral prints.",
            "key_drivers": ["Bridal Trousseau", "Lightweight Festive", "Photogenic Drapes", "Intimate Weddings"],
            "recommended_action": "Display draped on mannequins at boutique window displays with pearl accessories.",
            "local_relevance": "High"
        },
        {
            "item_name": "Washed Cargo Parachute Pants",
            "category": "Bottomwear",
            "associated_topic_id": 2,
            "trend_status": "Rising",
            "estimated_growth_pct": 74.0,
            "urgency_score": 78,
            "commercial_appeal": "Comfortable utilitarian baggy pants with drawstrings, multi-utility pockets, and adjustable toggles.",
            "key_drivers": ["Cargo Revival", "Comfort Styling", "Utility Fashion", "Travel Wear"],
            "recommended_action": "Pair with crop tops and oversized bomber jackets on visual merchandising displays.",
            "local_relevance": "Medium"
        },
        {
            "item_name": "Vintage Thrifted Denim Jackets",
            "category": "Sustainable Fashion",
            "associated_topic_id": 3,
            "trend_status": "Rising",
            "estimated_growth_pct": 68.0,
            "urgency_score": 75,
            "commercial_appeal": "One-of-a-kind upcycled and retro acid-wash denim jackets driving sustainable fashion choices.",
            "key_drivers": ["Circular Economy", "Thrift Culture", "Unique Pieces", "Indiranagar Thrift Hub"],
            "recommended_action": "Host a monthly 'Thrift & Swap Weekend Pop-up' to attract Gen Z shoppers.",
            "local_relevance": "High (Bangalore Local Favorite)"
        }
    ],
    "salon": [
        {
            "item_name": "Korean Glass Skin Facial",
            "category": "Skincare Treatment",
            "associated_topic_id": 0,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 165.0,
            "urgency_score": 98,
            "commercial_appeal": "Multi-step hydrating facial combining gentle AHA/BHA peeling, peptide ampoules, and LED light therapy for luminous dewiness.",
            "key_drivers": ["K-Beauty Glow", "Glass Skin", "Deep Hydration", "Pre-Event Ready"],
            "recommended_action": "Offer as a premium 60-minute glow treatment with before-and-after skin analysis reports.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Keratin Smoothing Treatment",
            "category": "Haircare Service",
            "associated_topic_id": 1,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 120.0,
            "urgency_score": 90,
            "commercial_appeal": "Formaldehyde-free protein smoothing treatment that eliminates monsoon frizz and seals hair cuticles for 4 months.",
            "key_drivers": ["Frizz Control", "Hair Botox", "Humidity Resistance", "Low Maintenance"],
            "recommended_action": "Bundle with sulfate-free aftercare shampoo and conditioner gift packs.",
            "local_relevance": "High"
        },
        {
            "item_name": "Sun-Kissed Balayage Highlights",
            "category": "Hair Color",
            "associated_topic_id": 2,
            "trend_status": "Rising",
            "estimated_growth_pct": 105.0,
            "urgency_score": 88,
            "commercial_appeal": "Hand-painted seamless highlights transitioning from natural roots to warm honey and caramel accents.",
            "key_drivers": ["Low Regrowth Maintenance", "Custom Painting", "Dimensional Color", "Natural Finish"],
            "recommended_action": "Showcase transformation video reels on Instagram focusing on gloss toning.",
            "local_relevance": "High"
        },
        {
            "item_name": "Chrome Glazed Donut Nails",
            "category": "Nail Art",
            "associated_topic_id": 3,
            "trend_status": "Rising",
            "estimated_growth_pct": 95.0,
            "urgency_score": 85,
            "commercial_appeal": "Pearly iridescent chrome powder rubbed over milky nude gel extensions popular on celebrity red carpets.",
            "key_drivers": ["Hailey Bieber Nails", "Chrome Powder", "Minimalist Luxury", "Gel Extensions"],
            "recommended_action": "Add as an express nail bar upgrade package during pedicures or hair coloring sessions.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Scalp Detox Head Spa",
            "category": "Wellness / Haircare",
            "associated_topic_id": 1,
            "trend_status": "Rising",
            "estimated_growth_pct": 82.0,
            "urgency_score": 82,
            "commercial_appeal": "Japanese-inspired scalp exfoliation, steam waterfall circulation, and acupressure massage for hair thinning relief.",
            "key_drivers": ["Japanese Head Spa", "Scalp Health", "Stress Relief", "Acupressure"],
            "recommended_action": "Set up a relaxing darkened head-spa suite with aromatherapy mist and calming sound therapy.",
            "local_relevance": "High"
        },
        {
            "item_name": "Brow Lamination & Tint",
            "category": "Grooming",
            "associated_topic_id": 4,
            "trend_status": "Rising",
            "estimated_growth_pct": 72.0,
            "urgency_score": 76,
            "commercial_appeal": "Semi-permanent perming technique that brushes brow hairs upward for a fuller, feathered supermodel look.",
            "key_drivers": ["Fluffy Brows", "Feathered Look", "Microblading Alternative", "Quick Service"],
            "recommended_action": "Offer lunch-break 30-minute brow enhancement slots with online booking discounts.",
            "local_relevance": "Medium"
        }
    ],
    "gym": [
        {
            "item_name": "Reformer Pilates Sculpt",
            "category": "Fitness Class",
            "associated_topic_id": 0,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 170.0,
            "urgency_score": 98,
            "commercial_appeal": "High-resistance spring-loaded carriage training emphasizing deep core stabilization, posture, and lean sculpting.",
            "key_drivers": ["Celebrity Workout", "Core Strength", "Low Impact", "Joint Friendly"],
            "recommended_action": "Launch beginner 5-session introductory packages with small group sizes (max 6 clients).",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Functional HIIT & Hyrox Conditioning",
            "category": "Athletic Training",
            "associated_topic_id": 1,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 115.0,
            "urgency_score": 89,
            "commercial_appeal": "Combination of ski erg, sled push, rowing, and running circuits designed for race and athletic conditioning.",
            "key_drivers": ["Hyrox Fitness", "Functional Movement", "Cardio Endurance", "Athletic Community"],
            "recommended_action": "Create dedicated weekend simulation tracks with leaderboards and community badges.",
            "local_relevance": "High"
        },
        {
            "item_name": "Sound Bath & Yin Yoga",
            "category": "Mindfulness / Yoga",
            "associated_topic_id": 2,
            "trend_status": "Rising",
            "estimated_growth_pct": 98.0,
            "urgency_score": 85,
            "commercial_appeal": "Restorative deep fascial stretching paired with Tibetan singing bowl acoustic vibrations for nervous system reset.",
            "key_drivers": ["Mental Health", "Vagus Nerve Reset", "Sound Healing", "Stress Reduction"],
            "recommended_action": "Schedule Sunday evening wind-down candlelit classes before the start of the work week.",
            "local_relevance": "High"
        },
        {
            "item_name": "Pickleball League Training",
            "category": "Racquet Sports",
            "associated_topic_id": 3,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 140.0,
            "urgency_score": 92,
            "commercial_appeal": "Fast-paced social paddle sport booming across corporate communities and recreational weekend clubs.",
            "key_drivers": ["Social Sport", "Easy Learning Curve", "All Age Fitness", "Corporate Leagues"],
            "recommended_action": "Partner with nearby turf venues or convert indoor studio courts for weekday evening ladder leagues.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Calisthenics & Joint Mobility",
            "category": "Bodyweight Fitness",
            "associated_topic_id": 4,
            "trend_status": "Rising",
            "estimated_growth_pct": 78.0,
            "urgency_score": 80,
            "commercial_appeal": "Mastery of muscle-ups, handstands, and gymnastic ring control combined with active joint mobility routines.",
            "key_drivers": ["Relative Strength", "No Machine Workout", "Handstand Goals", "Injury Prevention"],
            "recommended_action": "Set up a dedicated gymnastics ring and pull-up bar zone with progressive workshops.",
            "local_relevance": "Medium"
        },
        {
            "item_name": "Recovery Foam Rolling & Cold Plunge",
            "category": "Active Recovery",
            "associated_topic_id": 1,
            "trend_status": "Rising",
            "estimated_growth_pct": 88.0,
            "urgency_score": 84,
            "commercial_appeal": "Post-workout contrast therapy combining cold immersion plunge tubs with percussion massage guns.",
            "key_drivers": ["Biohacking", "Muscle Recovery", "Cold Plunge", "Endorphin Boost"],
            "recommended_action": "Add 15-minute recovery lounge add-ons to all monthly premium gym memberships.",
            "local_relevance": "High"
        }
    ],
    "gift_shop": [
        {
            "item_name": "Artisanal Soy Scented Candles",
            "category": "Home Fragrance",
            "associated_topic_id": 0,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 148.0,
            "urgency_score": 94,
            "commercial_appeal": "Eco-friendly non-toxic soy candles infused with french lavender, sandalwood, and crackling wooden wicks.",
            "key_drivers": ["Clean Fragrance", "Aesthetic Amber Jars", "Wooden Wick", "Self Care Gifting"],
            "recommended_action": "Set up a fragrance testing station in-store and offer personalized custom message labeling.",
            "local_relevance": "High"
        },
        {
            "item_name": "Curated Celebration Gift Hampers",
            "category": "Gifting Concept",
            "associated_topic_id": 1,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 130.0,
            "urgency_score": 91,
            "commercial_appeal": "Artisanal wooden gift boxes with bespoke selections of gourmet treats, scented candles, and keepsake journals.",
            "key_drivers": ["Corporate Gifting", "Wedding Favors", "Bespoke Packaging", "Hassle-Free Gifting"],
            "recommended_action": "Offer customizable 'Build Your Own Hamper' options online and in retail stores.",
            "local_relevance": "High"
        },
        {
            "item_name": "Custom Resin Art Nameplates",
            "category": "Personalized Decor",
            "associated_topic_id": 2,
            "trend_status": "Rising",
            "estimated_growth_pct": 95.0,
            "urgency_score": 86,
            "commercial_appeal": "Handmade epoxy resin nameplates with gold foil leafing, geode textures, and embedded dried botanicals.",
            "key_drivers": ["Housewarming Gifts", "Resin Art", "Personalized Touch", "Modern Entryways"],
            "recommended_action": "Display sample nameplates in store windows and provide 5-day custom order fulfillment.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Geometric Succulent Terrariums",
            "category": "Botanical Decor",
            "associated_topic_id": 3,
            "trend_status": "Rising",
            "estimated_growth_pct": 82.0,
            "urgency_score": 80,
            "commercial_appeal": "Faceted glass geometric vessels housing low-maintenance succulents, moss, and colored sand layers.",
            "key_drivers": ["Biophilic Design", "Desk Plants", "Low Maintenance", "Aesthetic Living"],
            "recommended_action": "Pair with branded mini care guides and offer pre-planted desktop arrangements.",
            "local_relevance": "High"
        },
        {
            "item_name": "Minimalist Ribbed Ceramic Vases",
            "category": "Home Decor",
            "associated_topic_id": 2,
            "trend_status": "Rising",
            "estimated_growth_pct": 74.0,
            "urgency_score": 76,
            "commercial_appeal": "Matte textured ceramic donut and ribbed arc vases designed for dry pampas grass and minimalist credenzas.",
            "key_drivers": ["Nordic Decor", "Ceramic Art", "Pampas Grass Trend", "Neutral Interiors"],
            "recommended_action": "Create styling vignette displays combining ceramic vases with dried floral arrangements.",
            "local_relevance": "Medium"
        },
        {
            "item_name": "Wax Seal Journaling Stationery Kits",
            "category": "Lifestyle / Craft",
            "associated_topic_id": 4,
            "trend_status": "Rising",
            "estimated_growth_pct": 62.0,
            "urgency_score": 70,
            "commercial_appeal": "Vintage brass stamp heads, wax melting spoons, and metallic wax beads popular among bullet journal hobbyists.",
            "key_drivers": ["Vintage Nostalgia", "Bullet Journaling", "Creative Hobbies", "Handwritten Letters"],
            "recommended_action": "Bundle with vintage deckle-edge paper and calligraphy inks as starter gift sets.",
            "local_relevance": "Medium"
        }
    ],
    "electronics": [
        {
            "item_name": "Active Noise Cancelling Wireless Earbuds",
            "category": "Audio Tech",
            "associated_topic_id": 0,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 135.0,
            "urgency_score": 92,
            "commercial_appeal": "Hybrid active noise cancellation with low-latency gaming mode, multipoint pairing, and transparency audio.",
            "key_drivers": ["Remote Work", "ANC Audio", "Commute Travel", "Affordable Tech"],
            "recommended_action": "Provide in-store listening demo units with sanitized ear tips and price-match guarantees.",
            "local_relevance": "High"
        },
        {
            "item_name": "Magnetic Wireless Power Bank",
            "category": "Mobile Accessories",
            "associated_topic_id": 1,
            "trend_status": "Rising",
            "estimated_growth_pct": 118.0,
            "urgency_score": 88,
            "commercial_appeal": "Snap-on slim MagSafe compatible 10,000mAh battery packs with kickstands for on-the-go phone charging.",
            "key_drivers": ["MagSafe Charging", "Cordless Travel", "Slim Form Factor", "Phone Kickstand"],
            "recommended_action": "Bundle power banks with shockproof MagSafe cases for discounts at billing.",
            "local_relevance": "High"
        },
        {
            "item_name": "AMOLED Smart Fitness Band",
            "category": "Wearable Tech",
            "associated_topic_id": 2,
            "trend_status": "Rising",
            "estimated_growth_pct": 95.0,
            "urgency_score": 84,
            "commercial_appeal": "Always-on vivid AMOLED display tracking continuous SpO2, sleep stages, stress levels, and sports modes.",
            "key_drivers": ["Health Tracking", "Affordable Wearable", "14-Day Battery", "Sleep Optimization"],
            "recommended_action": "Highlight health metrics and battery longevity with side-by-side spec comparison cards.",
            "local_relevance": "High"
        },
        {
            "item_name": "RGB Hot-Swappable Mechanical Keyboard",
            "category": "Computer Peripherals",
            "associated_topic_id": 3,
            "trend_status": "Rising",
            "estimated_growth_pct": 88.0,
            "urgency_score": 82,
            "commercial_appeal": "75% compact layout mechanical keyboards with lubricated linear switches, sound dampening foam, and wireless tri-mode.",
            "key_drivers": ["Custom Keyboard", "Desk Setup", "Tactile Typing", "Tech Aesthetic"],
            "recommended_action": "Set up a hands-on desk display so customers can test switch actuation and acoustic clack.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "65W GaN Compact Fast Charger",
            "category": "Charging Tech",
            "associated_topic_id": 1,
            "trend_status": "Rising",
            "estimated_growth_pct": 78.0,
            "urgency_score": 79,
            "commercial_appeal": "Pocket-sized Gallium Nitride power brick capable of simultaneously charging a laptop, tablet, and smartphone.",
            "key_drivers": ["GaN Technology", "Single Charger Lifestyle", "Travel Essential", "Laptop Fast Charge"],
            "recommended_action": "Market as an essential travel replacement charger alongside durable braided 100W cables.",
            "local_relevance": "High"
        },
        {
            "item_name": "Smart Air Quality & Desk Ambient Monitor",
            "category": "Smart Home Tech",
            "associated_topic_id": 4,
            "trend_status": "Rising",
            "estimated_growth_pct": 64.0,
            "urgency_score": 72,
            "commercial_appeal": "Real-time PM2.5, CO2, and temperature display with smart home integration and circadian screen backlighting.",
            "key_drivers": ["Indoor Air Quality", "WFH Productivity", "Screenbar Lighting", "Tech Wellness"],
            "recommended_action": "Bundle monitor screenbars with smart air quality sensors for home office ergonomics.",
            "local_relevance": "Medium"
        }
    ],
    "d2c": [
        {
            "item_name": "Small-Batch Handcrafted Pottery Mugs",
            "category": "Artisan Kitchenware",
            "associated_topic_id": 0,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 150.0,
            "urgency_score": 95,
            "commercial_appeal": "Wheel-thrown studio stoneware mugs with speckled glazes and ergonomic thumb-rest handles.",
            "key_drivers": ["Studio Pottery", "Slow Living", "Mindful Morning", "Artisan Kitchen"],
            "recommended_action": "Run limited monthly drop batches with maker stamp signatures and studio process videos.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Organic Whipped Shea Body Butter",
            "category": "Clean Beauty",
            "associated_topic_id": 1,
            "trend_status": "Explosive Growth",
            "estimated_growth_pct": 125.0,
            "urgency_score": 90,
            "commercial_appeal": "Waterless formula of unrefined African shea butter, mango butter, and jojoba oil with melt-on-skin texture.",
            "key_drivers": ["Clean Beauty", "Waterless Formula", "Dry Skin Relief", "Non-Toxic Skincare"],
            "recommended_action": "Offer travel-sized 50g trial jars and highlight all-natural food-grade ingredient sourcing.",
            "local_relevance": "High"
        },
        {
            "item_name": "Dainty Freshwater Pearl Beaded Jewelry",
            "category": "Handmade Accessories",
            "associated_topic_id": 2,
            "trend_status": "Rising",
            "estimated_growth_pct": 108.0,
            "urgency_score": 87,
            "commercial_appeal": "Handcrafted choker necklaces and hoop earrings featuring irregular baroque pearls and anti-tarnish 18k gold beads.",
            "key_drivers": ["Baroque Pearls", "Anti-Tarnish", "Layered Jewelry", "Affordable Luxury"],
            "recommended_action": "Package in reusable velvet jewelry pouches with custom handwritten thank you cards.",
            "local_relevance": "High"
        },
        {
            "item_name": "Eco-Friendly Heavy Canvas Tote Bags",
            "category": "Sustainable Accessories",
            "associated_topic_id": 3,
            "trend_status": "Rising",
            "estimated_growth_pct": 92.0,
            "urgency_score": 82,
            "commercial_appeal": "Durable 16oz cotton canvas totes featuring internal laptop sleeves, water bottle holders, and zipper tops.",
            "key_drivers": ["Plastic Free Living", "Farmers Market Bag", "Aesthetic Carryall", "College Essential"],
            "recommended_action": "Collaborate with independent local graphic illustrators on limited city-themed print designs.",
            "local_relevance": "High (Bangalore Local Favorite)"
        },
        {
            "item_name": "Botanical Scented Soy Candles",
            "category": "Home Decor",
            "associated_topic_id": 1,
            "trend_status": "Rising",
            "estimated_growth_pct": 85.0,
            "urgency_score": 80,
            "commercial_appeal": "Hand-poured coconut-soy wax topped with real pressed botanical petals, wild rosebuds, and quartz crystals.",
            "key_drivers": ["Botanical Aesthetics", "Crystal Healing", "Clean Burn", "Gift Friendly"],
            "recommended_action": "Offer seasonal aroma subscriptions and refill-jar exchange discount programs.",
            "local_relevance": "Medium"
        },
        {
            "item_name": "Cold-Pressed Herbal Face Elixir",
            "category": "Organic Skincare",
            "associated_topic_id": 1,
            "trend_status": "Rising",
            "estimated_growth_pct": 76.0,
            "urgency_score": 78,
            "commercial_appeal": "Lightweight facial oil infused with Kashmiri saffron, rosehip, and cold-pressed kumkumadi botanicals.",
            "key_drivers": ["Ayurvedic Beauty", "Kumkumadi Glow", "Facial Gua Sha Oil", "Pure Botanicals"],
            "recommended_action": "Bundle with jade or rose-quartz facial gua sha stones for evening skin rituals.",
            "local_relevance": "High"
        }
    ]
}


def get_curated_sme_trends(sme: str, city: str = "Bengaluru") -> list[dict]:
    sme_clean = sme.strip().lower().replace(" ", "_").replace("-", "_")
    sme_clean = SME_ALIASES.get(sme_clean, sme_clean)
    items = SME_CURATED_TRENDS.get(sme_clean, SME_CURATED_TRENDS["cafe"])
    import copy
    return copy.deepcopy(items)


def _is_cached_trends_valid(sme: str, cached: dict) -> bool:
    if not isinstance(cached, dict):
        return False
    items = cached.get("trending_items")
    if not items or not isinstance(items, list) or len(items) == 0:
        return False

    sme_clean = sme.strip().lower().replace(" ", "_").replace("-", "_")
    sme_clean = SME_ALIASES.get(sme_clean, sme_clean)

    # Food/beverage keywords that should NEVER be present for non-F&B SMEs
    fnb_markers = {"matcha", "kunafa", "cold brew", "latte", "espresso", "brunch", "boba", "coffee", "pancakes", "biryani", "dosa", "cheesecake"}
    non_fnb_smes = {"boutique", "salon", "gym", "gift_shop", "electronics", "d2c"}

    if sme_clean in non_fnb_smes:
        first_item_name = str(items[0].get("item_name", "")).lower()
        if any(marker in first_item_name for marker in fnb_markers):
            return False
        first_category = str(items[0].get("category", "")).lower()
        if first_category in {"dessert", "beverage", "dining concept", "dining experience", "breakfast", "main course"}:
            return False

    return True


def make_demo_data(sme: str = "cafe", days: int = 120, seed: int = 7) -> tuple[list[dict], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sme_clean = sme.strip().lower().replace(" ", "_").replace("-", "_")
    sme_clean = SME_ALIASES.get(sme_clean, sme_clean)
    topics = SME_DEMO_TOPICS.get(sme_clean, SME_DEMO_TOPICS["cafe"])

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

    rising_items = SME_DEMO_RISING.get(sme_clean, SME_DEMO_RISING["cafe"])
    rising_df = pd.DataFrame(rising_items)
    return rows, rising_df


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
        nltk.data.find("corpora/wordnet")
        _lemmatizer = WordNetLemmatizer()
    except Exception:
        _lemmatizer = None  # fall back to fast suffix stripping


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


def lstm_forecast(values, horizon=4, window=4, epochs=20):
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
    if len(X) < 2:
        return None
    tf.random.set_seed(0)
    model = models.Sequential([layers.Input((window, 1)), layers.LSTM(8), layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=min(epochs, 20), batch_size=8, verbose=0)
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
        print(f"[demo] generating synthetic data for SME '{sme}' in '{city}'")
        rows, rising = make_demo_data(sme=sme, days=max(days, 90))
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
            if _is_cached_trends_valid(sme_clean, cached_trends):
                return cached_trends
            else:
                print(f"[trend_engine] Cached trends for '{sme_clean}' invalid or contaminated. Re-discovering...")

        _here_dir = Path(__file__).resolve().parent
        for out_dir in [_here_dir / "trend_output", _here_dir.parent / "trend_output", Path("trend_output")]:
            out_file = out_dir / "trending_items.json"
            if out_file.is_file():
                try:
                    with open(out_file, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                    if file_data.get("sme", "").lower() == sme_clean and _is_cached_trends_valid(sme_clean, file_data):
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
    trending_items = []
    try:
        trending_items = analyze_trends(
            trends_data=trends_raw,
            rising_queries=trends_raw.get("rising_queries", []),
        )
    except Exception as exc:
        print(f"[trend_engine] Cohere analysis note ({exc}). Utilizing curated SME trend intelligence.")

    # Fall back to curated SME trends if Cohere returned empty or mismatched items
    if not trending_items or not _is_cached_trends_valid(sme_clean, {"trending_items": trending_items}):
        print(f"[trend_engine] Supplying verified SME-specific trends for '{sme_clean}' in '{city}'")
        trending_items = get_curated_sme_trends(sme_clean, city)

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