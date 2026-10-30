"""
trend_db.py
-----------
Database storage and caching layer for SME Trend Intelligence.
Stores outputs into dynamic tables named `{SME name}_trends` (e.g. cafe_trends, restaurant_trends)
in Neon PostgreSQL (with SQLite fallback).

Provides:
  - get_cached_trends(sme, city) -> checks DB first
  - save_trends_to_db(sme, city, data) -> persists processed trend output
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    desc,
    func,
    inspect,
    select,
)

_HERE = Path(__file__).resolve().parent
if (_HERE / ".env").exists():
    load_dotenv(dotenv_path=_HERE / ".env")
elif (_HERE.parent / ".env").exists():
    load_dotenv(dotenv_path=_HERE.parent / ".env")

_ENGINE = None
_METADATA = MetaData()


def get_db_engine():
    """Returns database engine (Neon PostgreSQL with SQLite fallback)."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    urls_to_try = []
    for var in ("DATABASE_URL", "DATABASE_URL_UNPOOLED"):
        val = os.getenv(var)
        if val and val not in urls_to_try:
            urls_to_try.append(val)

    for raw_url in urls_to_try:
        url = raw_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)

        try:
            eng = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 4})
            with eng.connect() as conn:
                conn.execute(select(1))
            _ENGINE = eng
            return _ENGINE
        except Exception as exc:
            print(f"[trend_db] Connection attempt to {raw_url.split('@')[-1] if '@' in raw_url else 'URL'} notice: {exc}. Trying next...")

    print(f"[trend_db] Neon PostgreSQL unavailable. Using SQLite fallback.")

    # SQLite fallback - keep in sync with inventory/data/advantage.db
    sqlite_dir = _HERE.parent / "inventory" / "data"
    if not sqlite_dir.exists():
        sqlite_dir = _HERE / "inventory" / "data"
    sqlite_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = sqlite_dir / "advantage.db"
    _ENGINE = create_engine(
        f"sqlite:///{sqlite_path}",
        connect_args={"check_same_thread": False},
    )
    return _ENGINE


def _get_table_name(sme: str) -> str:
    """Sanitizes SME name into a valid SQL table name: {sme}_trends."""
    clean_sme = re.sub(r"[^a-zA-Z0-9_]", "", sme.strip().lower())
    return f"{clean_sme}_trends"


def _get_or_create_table(sme: str, engine) -> Table:
    """Returns or creates the dynamic table {sme}_trends."""
    table_name = _get_table_name(sme)
    if table_name in _METADATA.tables:
        return _METADATA.tables[table_name]

    table = Table(
        table_name,
        _METADATA,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("city", String(100), nullable=False, index=True),
        Column("generated_at", String(50), nullable=False),
        Column("total_documents_analyzed", Integer, default=0),
        Column("total_trending_items", Integer, default=0),
        Column("trending_items_json", Text, nullable=True),
        Column("topic_clusters_json", Text, nullable=True),
        Column("rising_searches_json", Text, nullable=True),
        Column("full_output_json", Text, nullable=False),
        Column("created_at", DateTime, default=datetime.utcnow),
        extend_existing=True,
    )
    table.create(bind=engine, checkfirst=True)
    return table


def get_cached_trends(sme: str, city: str, max_age_hours: int = None) -> dict | None:
    """
    Checks the database for an existing {sme}_trends entry for the given city.
    Returns the parsed JSON response if found, else None.
    """
    try:
        engine = get_db_engine()
        inspector = inspect(engine)
        table_name = _get_table_name(sme)
        
        # Check if table exists in DB
        if not inspector.has_table(table_name):
            return None

        table = _get_or_create_table(sme, engine)
        city_clean = city.strip().lower()

        stmt = (
            select(table)
            .where(func.lower(table.c.city) == city_clean)
            .order_by(desc(table.c.id))
            .limit(1)
        )

        with engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()

        if not row:
            return None

        # Check age if max_age_hours is provided
        if max_age_hours is not None and row["created_at"]:
            age_seconds = (datetime.utcnow() - row["created_at"]).total_seconds()
            if age_seconds > max_age_hours * 3600:
                print(f"[trend_db] Cache expired for {sme} in {city} ({age_seconds/3600:.1f}h old)")
                return None

        # Parse stored JSON
        output = json.loads(row["full_output_json"])

        # Validate that cached items belong to the requested SME domain
        items = output.get("trending_items", [])
        if items and isinstance(items, list):
            fnb_markers = {"matcha", "kunafa", "cold brew", "latte", "espresso", "brunch", "boba", "coffee", "pancakes", "biryani", "dosa", "cheesecake"}
            non_fnb_smes = {"boutique", "salon", "gym", "gift_shop", "electronics", "d2c"}
            sme_lower = sme.strip().lower()
            if sme_lower in non_fnb_smes:
                first_name = str(items[0].get("item_name", "")).lower()
                first_cat = str(items[0].get("category", "")).lower()
                if any(m in first_name for m in fnb_markers) or first_cat in {"dessert", "beverage", "dining concept", "dining experience"}:
                    print(f"[trend_db] Corrupted cache detected in table '{table_name}' for SME '{sme}' (contains F&B items). Invalidating cache entry.")
                    return None

        output["from_database"] = True
        output["database_table"] = table_name
        output["database_record_id"] = row["id"]
        output["database_saved_at"] = row["created_at"].isoformat() if row["created_at"] else None
        print(f"[trend_db] Found cached result in table '{table_name}' for city '{city}' (ID: {row['id']})")
        return output

    except Exception as exc:
        print(f"[trend_db] Cache lookup warning: {exc}")
        return None


def save_trends_to_db(sme: str, city: str, trend_output: dict) -> int | None:
    """
    Saves the processed trend output to the database table `{sme}_trends`.
    """
    try:
        engine = get_db_engine()
        table = _get_or_create_table(sme, engine)
        table_name = _get_table_name(sme)

        # Prepare clean fields
        items = trend_output.get("trending_items", [])
        topics = trend_output.get("topic_clusters", [])
        rising = trend_output.get("rising_searches", [])

        # Clone and remove any ephemeral cache keys before storing
        clean_output = {k: v for k, v in trend_output.items() if not k.startswith("from_database")}

        insert_stmt = table.insert().values(
            city=city.strip(),
            generated_at=str(trend_output.get("generated", datetime.utcnow().isoformat())),
            total_documents_analyzed=int(trend_output.get("total_documents_analyzed", 0)),
            total_trending_items=int(trend_output.get("total_trending_items", len(items))),
            trending_items_json=json.dumps(items, ensure_ascii=False),
            topic_clusters_json=json.dumps(topics, ensure_ascii=False),
            rising_searches_json=json.dumps(rising, ensure_ascii=False),
            full_output_json=json.dumps(clean_output, ensure_ascii=False),
            created_at=datetime.utcnow(),
        )

        with engine.begin() as conn:
            result = conn.execute(insert_stmt)
            inserted_id = result.inserted_primary_key[0] if result.inserted_primary_key else None

        print(f"[trend_db] Successfully saved trend output to table '{table_name}' (ID: {inserted_id})")
        return inserted_id

    except Exception as exc:
        print(f"[trend_db] Error saving to database table: {exc}")
        return None
