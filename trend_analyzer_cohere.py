#!/usr/bin/env python3
"""
AdVantage OS - Trend Item Analyzer (Cohere LLM)
==============================================
Reads the topic modeling & forecasting output from `trends.json` (and optional
rising queries), passes it to Cohere LLM (e.g. command-r-plus / command-r),
filters out social noise, and extracts every actionable trending item/dish/product
with comprehensive metrics for SME decision-making.

Usage:
  python trend_analyzer_cohere.py
  python trend_analyzer_cohere.py --input trend_output/trends.json --model command-r-plus
  python trend_analyzer_cohere.py --api-key YOUR_COHERE_API_KEY
"""

import argparse
import json
import os
import re
import sys
import pandas as pd

# --------------------------------------------------------------------------
# Environment loading & terminal safety
# --------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def _load_env():
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

_load_env()


# --------------------------------------------------------------------------
# Cohere Client Wrapper
def call_cohere(prompt: str, api_key: str, model: str = "command-r-plus-08-2024") -> str:
    """Invokes Cohere Chat API using cohere SDK with model fallback."""
    # Map legacy names if passed
    model_map = {
        "command-r-plus": "command-r-plus-08-2024",
        "command-r": "command-r-08-2024",
    }
    target_model = model_map.get(model, model)

    import cohere
    client = cohere.ClientV2(api_key=api_key)

    for m in [target_model, "command-r-08-2024"]:
        try:
            response = client.chat(
                model=m,
                messages=[
                    {"role": "system", "content": "You are a senior market research analyst specializing in retail, restaurant, and SME consumer trends."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return response.message.content[0].text
        except Exception as err:
            print(f"[Cohere] Note: '{m}' error ({err}), trying fallback...")
            continue

    # Final attempt with direct requests
    import requests
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "command-r-08-2024",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
    r = requests.post("https://api.cohere.com/v2/chat", json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", [{}])[0].get("text", "")


# --------------------------------------------------------------------------
# Main Analysis Pipeline
# --------------------------------------------------------------------------
def analyze_trends(trends_data: dict, rising_queries: list = None, api_key: str = None, model: str = "command-r-plus-08-2024"):
    api_key = api_key or os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY is not set. Please set it in your environment or .env file.")

    if rising_queries is None:
        rising_queries = trends_data.get("rising_queries", [])

    sme = trends_data.get("sme", "business")
    city = trends_data.get("city", "the target market")
    trends_list = trends_data.get("trends", [])

    prompt = f"""
You are an expert commercial market intelligence analyst for small-and-medium enterprises (SMEs).
You are analyzing real-world social listening data (Reddit, YouTube, Google Trends) collected for a '{sme}' business in '{city}'.

Input Topic Clusters and Forecasts:
{json.dumps(trends_list, indent=2)}

Rising Search Queries (Google Trends, if available):
{json.dumps(rising_queries[:15], indent=2)}

TASK:
1. Thoroughly analyze the topic clusters, seed keywords, engagement metrics, and text examples.
2. Discard social noise, filler conversational words, off-topic Reddit chatter (e.g. meta memes, family anecdotes), and non-commercial noise.
3. Identify and extract EVERY possible concrete product, menu item, dish, drink, preparation style, dining experience, or consumer service that is currently trending or surging in demand for this {sme} in {city}.
4. For each identified trending item, calculate/assign clear commercial metrics based on the data.

OUTPUT FORMAT REQUIREMENTS:
You MUST respond with a single, valid JSON object containing an "items" array.
Do not include any commentary outside the JSON.

Each object in the "items" array must contain:
- "item_name": (string) Specific name of the trending item or dish (e.g., "Benne Dosa", "Iced Matcha Latte", "Donne Biryani", "Sourdough Pizza", "Dubai Kunafa Chocolate")
- "category": (string) Category (e.g., "Beverage", "Main Course", "Breakfast", "Snack / Street Food", "Dessert", "Dining Concept")
- "associated_topic_id": (int) ID of the topic cluster this was extracted from
- "trend_status": (string) One of: "Explosive Growth", "Rising", "At Peak", "Steady / Evergreen", "Cooling Down"
- "estimated_growth_pct": (float or int) Growth percentage estimate (e.g., +450)
- "urgency_score": (int) Score from 1 to 100 on how urgently the SME should adopt or market this item
- "commercial_appeal": (string) Short explanation of why customers are craving or talking about it
- "key_drivers": (list of strings) 2-4 keywords or cultural triggers driving this trend
- "recommended_action": (string) Specific, practical business recommendation for the SME owner (e.g., "Add as a weekend breakfast special", "Launch a 15-second reel showing preparation", "Introduce a bundle deal")
- "local_relevance": (string) "High (Bangalore Local Favorite)", "Medium", or "General Trend"
"""

    print(f"\n[Cohere] Sending trend signals to Cohere model '{model}'...")
    raw_response = call_cohere(prompt, api_key, model)

    # Clean markdown codeblocks if returned
    text_clean = raw_response.strip()
    if text_clean.startswith("```"):
        text_clean = re.sub(r"^```(?:json)?\s*", "", text_clean)
        text_clean = re.sub(r"\s*```$", "", text_clean)

    parsed = json.loads(text_clean)
    items = parsed.get("items", []) if isinstance(parsed, dict) else parsed
    return items


def main():
    parser = argparse.ArgumentParser(description="Extract actionable item trends using Cohere LLM")
    parser.add_argument("--input", default="trend_output/trends.json", help="Path to trends.json")
    parser.add_argument("--rising", default="trend_output/google_trends_rising_queries.csv", help="Path to rising queries CSV")
    parser.add_argument("--out", default="trend_output", help="Output directory")
    parser.add_argument("--model", default="command-r-plus", help="Cohere model (command-r-plus, command-r, command-a-03-2025)")
    parser.add_argument("--api-key", default=None, help="Cohere API key (defaults to COHERE_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("COHERE_API_KEY")
    if not api_key:
        print("\n[ERROR] COHERE_API_KEY is not set!")
        print("Please set it in your environment or add it to your .env file:")
        print("  $env:COHERE_API_KEY = \"your_cohere_api_key_here\"")
        print("Or pass it via command line:")
        print("  python trend_analyzer_cohere.py --api-key your_cohere_api_key\n")
        sys.exit(1)

    if not os.path.exists(args.input):
        sys.exit(f"[ERROR] Input file not found: {args.input}. Run trend_engine.py first.")

    with open(args.input, "r", encoding="utf-8") as f:
        trends_data = json.load(f)

    rising_queries = []
    if os.path.exists(args.rising):
        try:
            df_rising = pd.read_csv(args.rising)
            rising_queries = df_rising.to_dict(orient="records")
        except Exception:
            pass

    sme = trends_data.get("sme", "SME").upper()
    city = trends_data.get("city", "City")
    print(f"Loaded {len(trends_data.get('trends', []))} topic clusters for {sme} in {city}.")

    items = analyze_trends(trends_data, rising_queries, api_key, args.model)
    if not items:
        print("[Cohere] No items returned.")
        return

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "trending_items.json")
    out_csv = os.path.join(args.out, "trending_items.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "sme": trends_data.get("sme"),
            "city": trends_data.get("city"),
            "total_items": len(items),
            "items": items
        }, f, indent=2, ensure_ascii=False)

    df_items = pd.DataFrame([{
        "Item Name": it.get("item_name"),
        "Category": it.get("category"),
        "Status": it.get("trend_status"),
        "Growth %": it.get("estimated_growth_pct"),
        "Urgency (1-100)": it.get("urgency_score"),
        "Local Relevance": it.get("local_relevance"),
        "Commercial Appeal": it.get("commercial_appeal"),
        "Recommended Action": it.get("recommended_action"),
        "Key Drivers": ", ".join(it.get("key_drivers", [])) if isinstance(it.get("key_drivers"), list) else str(it.get("key_drivers"))
    } for it in items])

    df_items.to_csv(out_csv, index=False, encoding="utf-8")

    # Display rich summary table to console
    print(f"\n{'='*85}")
    print(f"🔥 COHERE MARKET INTELLIGENCE: TOP TRENDING ITEMS FOR {sme} ({city.upper()})")
    print(f"{'='*85}\n")

    for i, it in enumerate(items, 1):
        name = it.get("item_name", "Unknown Item")
        cat = it.get("category", "General")
        status = it.get("trend_status", "Active")
        urgency = it.get("urgency_score", 50)
        growth = it.get("estimated_growth_pct", 0)
        rec = it.get("recommended_action", "No action")
        appeal = it.get("commercial_appeal", "")
        drivers = ", ".join(it.get("key_drivers", [])) if isinstance(it.get("key_drivers"), list) else it.get("key_drivers", "")

        print(f"{i:>2}. 🌟 {name.upper()}  [{cat}]")
        print(f"    Status: {status}  |  Urgency: {urgency}/100  |  Est. Growth: +{growth}%")
        print(f"    Why It's Trending: {appeal}")
        print(f"    Key Drivers: {drivers}")
        print(f"    💡 Business Action: {rec}\n")

    print(f"[Done] Saved structured outputs to:")
    print(f"  * JSON: {out_json}")
    print(f"  * CSV:  {out_csv}\n")


if __name__ == "__main__":
    main()
