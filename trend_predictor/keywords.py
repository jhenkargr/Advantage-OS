"""
keywords.py
-----------
AdVantage OS — AI-Powered Keyword Expansion via Cohere LLM.

Takes a seed keyword and SME type, calls Cohere API (Command-R / Command-R+),
and extracts a comprehensive list of commercial, high-intent related keywords,
trending product/menu variations, and consumer search terms tailored specifically
to that SME category for trend prediction and social listening.

Can be imported as a Python module, mounted as a FastAPI APIRouter, or run via CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# ── Terminal / encoding safety ────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── Load Environment Variables ────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent

def _load_environment() -> None:
    env_paths = [
        _HERE / ".env",
        _HERE / "advantage" / ".env",
        _HERE.parent / ".env",
        Path.cwd() / ".env",
    ]
    for path in env_paths:
        if path.is_file():
            load_dotenv(dotenv_path=path, override=False)

_load_environment()


# ── Schemas ───────────────────────────────────────────────────────────────────
class KeywordRequest(BaseModel):
    keyword: str = Field(
        ...,
        min_length=1,
        description="Seed keyword to expand for trend prediction.",
        json_schema_extra={"example": "coffee"},
    )
    sme_type: str = Field(
        ...,
        min_length=1,
        description="SME industry/category (e.g., cafe, restaurant, bakery, boutique, salon, fitness, etc.).",
        json_schema_extra={"example": "cafe"},
    )
    limit: int = Field(
        default=25,
        ge=5,
        le=100,
        description="Maximum number of related keywords to return.",
        json_schema_extra={"example": 25},
    )


class KeywordResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "success"})
    keyword: str = Field(..., json_schema_extra={"example": "coffee"})
    sme_type: str = Field(..., json_schema_extra={"example": "cafe"})
    count: int = Field(..., json_schema_extra={"example": 25})
    related_keywords: List[str] = Field(
        ...,
        description="List of related keywords, items, and search queries for trend prediction.",
    )


# ── Cohere Invocation Helper ──────────────────────────────────────────────────
def _call_cohere_chat(prompt: str, api_key: str, model: str = "command-r-08-2024") -> str:
    """
    Calls Cohere Chat API using cohere SDK (ClientV2) with model fallback,
    followed by direct HTTP request fallback.
    """
    model_map = {
        "command-r": "command-r-08-2024",
        "command-r-plus": "command-r-plus-08-2024",
    }
    target_model = model_map.get(model, model)
    fallback_models = [target_model, "command-r-08-2024", "command-r-plus-08-2024"]
    models_to_try = list(dict.fromkeys(fallback_models))

    # 1. Try Cohere SDK ClientV2
    try:
        import cohere
        client = cohere.ClientV2(api_key=api_key)
        for m in models_to_try:
            try:
                response = client.chat(
                    model=m,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a senior market intelligence and consumer trend analyst "
                                "specializing in local SME businesses, retail, food & beverage, "
                                "and social search demand."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                text = response.message.content[0].text
                if text:
                    return text
            except Exception as e:
                print(f"[keywords] Cohere SDK call with model '{m}' failed: {e}. Trying fallback...")
                continue
    except ImportError:
        print("[keywords] 'cohere' package not available, falling back to direct HTTP request.")

    # 2. Try direct HTTP request to Cohere v2 chat API
    import requests
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for m in models_to_try:
        try:
            payload = {
                "model": m,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
            res = requests.post("https://api.cohere.com/v2/chat", json=payload, headers=headers, timeout=60)
            if res.status_code == 200:
                data = res.json()
                content = data.get("message", {}).get("content", [])
                if content and isinstance(content, list):
                    return content[0].get("text", "")
        except Exception as e:
            print(f"[keywords] Direct HTTP call with model '{m}' failed: {e}")
            continue

    raise RuntimeError("All Cohere API calls failed. Please check your network connection and API key.")


def _clean_and_parse_json(raw_text: str) -> dict[str, Any]:
    """Safely extracts and parses JSON from LLM response text."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse valid JSON from Cohere output: {text[:200]}")


# ── Core Keyword Expansion Function ──────────────────────────────────────────
def get_related_keywords(
    keyword: str,
    sme_type: str,
    limit: int = 25,
    api_key: Optional[str] = None,
    model: str = "command-r-08-2024",
) -> List[str]:
    """
    Takes an input keyword and SME type, prompts Cohere LLM, and returns
    a list of related keywords tailored for trend prediction.

    Args:
        keyword: Primary seed term (e.g. 'coffee', 'croissant', 'saree', 'facial')
        sme_type: Industry/type of SME (e.g. 'cafe', 'bakery', 'boutique', 'salon')
        limit: Max related keywords to return (default: 25)
        api_key: Cohere API key (defaults to COHERE_API_KEY environment variable)
        model: Cohere model identifier (defaults to 'command-r-08-2024')

    Returns:
        List[str]: List of clean, normalized related keywords.
    """
    keyword_clean = keyword.strip()
    sme_clean = sme_type.strip().lower()

    if not keyword_clean:
        raise ValueError("Keyword must not be empty.")
    if not sme_clean:
        raise ValueError("SME type must not be empty.")

    key = api_key or os.getenv("COHERE_API_KEY")
    if not key:
        raise ValueError(
            "COHERE_API_KEY is not set. Please provide it or configure it in your .env file."
        )

    prompt = f"""You are a commercial market intelligence and consumer trend forecasting analyst for Small & Medium Enterprises (SMEs).

We are expanding input keywords for an automated Trend Prediction Engine that forecasts 30-day demand by scraping Reddit, X (Twitter), YouTube, and Google Trends.

Input Parameters:
- Seed Keyword: "{keyword_clean}"
- SME Business Type: "{sme_clean}"
- Desired Keyword Count: ~{limit}

Requirements:
1. Generate an exhaustive list of {limit} high-signal, actionable related keywords, trending items, flavor/style variations, menu items, product types, and consumer search queries specifically relevant to a "{sme_clean}" business around "{keyword_clean}".
2. Focus strictly on commercial, actionable terms that consumers search for, discuss, or buy in the context of a {sme_clean}.
3. Discard irrelevant words, off-topic associations, meta chatter, or generic non-commercial terms.
4. Normalize all terms as concise strings (1 to 4 words each, lowercase, trimmed).

Output Requirements:
Return strictly a valid JSON object in the following format:
{{
  "keyword": "{keyword_clean}",
  "sme_type": "{sme_clean}",
  "related_keywords": [
    "example keyword 1",
    "example keyword 2"
  ]
}}
"""

    raw_response = _call_cohere_chat(prompt=prompt, api_key=key, model=model)
    parsed = _clean_and_parse_json(raw_response)

    raw_list = (
        parsed.get("related_keywords")
        or parsed.get("keywords")
        or parsed.get("items")
        or parsed.get("results")
        or []
    )

    if not isinstance(raw_list, list):
        raise ValueError("Cohere did not return a valid list of related keywords.")

    seen = set()
    cleaned_keywords: List[str] = []

    for item in raw_list:
        if isinstance(item, dict):
            val = item.get("name") or item.get("keyword") or item.get("term") or ""
        else:
            val = str(item)

        val = val.strip().lower()
        val = re.sub(r"^[\d\.\-\*\•\s]+", "", val).strip()
        val = re.sub(r"^\"|\"$", "", val).strip()

        if val and val not in seen:
            seen.add(val)
            cleaned_keywords.append(val)

    if limit and len(cleaned_keywords) > limit:
        cleaned_keywords = cleaned_keywords[:limit]

    return cleaned_keywords


def expand_keywords(
    keyword: str,
    sme_type: str,
    limit: int = 25,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """High-level function returning a structured response dictionary."""
    keywords = get_related_keywords(
        keyword=keyword,
        sme_type=sme_type,
        limit=limit,
        api_key=api_key,
    )
    return {
        "status": "success",
        "keyword": keyword.strip(),
        "sme_type": sme_type.strip().lower(),
        "count": len(keywords),
        "related_keywords": keywords,
    }


# ── FastAPI APIRouter ─────────────────────────────────────────────────────────
router = APIRouter(tags=["Keyword Intelligence (Cohere)"])


@router.get(
    "/keywords/related",
    response_model=KeywordResponse,
    summary="Get related keywords for SME trend prediction",
    description="Takes a seed keyword and SME type as query parameters, queries Cohere LLM, and returns related trending keywords in a list.",
)
def get_related_keywords_endpoint(
    keyword: str = Query(..., min_length=1, examples=["coffee"], description="Seed keyword to expand"),
    sme_type: str = Query("cafe", min_length=1, examples=["cafe"], description="SME industry/category (e.g., cafe, restaurant, bakery, boutique, salon)"),
    limit: int = Query(25, ge=5, le=100, description="Maximum number of related keywords to return"),
) -> KeywordResponse:
    try:
        data = expand_keywords(
            keyword=keyword,
            sme_type=sme_type,
            limit=limit,
        )
        return KeywordResponse(**data)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Keyword generation failed: {exc}")


# ── CLI Interface ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expand seed keywords for SME trend prediction using Cohere LLM."
    )
    parser.add_argument("--keyword", "-k", type=str, required=True, help="Input keyword (e.g., 'coffee', 'latte')")
    parser.add_argument("--sme", "-s", type=str, required=True, help="SME type (e.g., 'cafe', 'restaurant', 'boutique')")
    parser.add_argument("--limit", "-l", type=int, default=25, help="Number of related keywords (default: 25)")
    parser.add_argument("--api-key", type=str, default=None, help="Optional Cohere API Key")

    args = parser.parse_args()

    try:
        result = expand_keywords(
            keyword=args.keyword,
            sme_type=args.sme,
            limit=args.limit,
            api_key=args.api_key,
        )
        print(json.dumps(result, indent=2))
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
