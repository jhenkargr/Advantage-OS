"""
influencers.py
--------------
AdVantage OS — Influencer Clusters & Database API Endpoint.

Provides:
  • POST /api/influencers/upload-csv: Upload influencer_clusters.csv directly to database.
  • GET  /api/influencers: List & filter influencers by tier, cluster, engagement, bot flag.
  • GET  /api/influencers/clusters/summary: Clustered niche breakdown and audience stats.
  • GET  /api/influencers/{channel_id}: Individual influencer profile.
"""

from __future__ import annotations

import io
from typing import Any, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from inventory.db import Influencer, get_db

router = APIRouter(prefix="/influencers", tags=["Influencers"])


# ── Pydantic Response Schemas ─────────────────────────────────────────────────
class InfluencerResponseItem(BaseModel):
    channel_id: str
    channel_name: str
    country: Optional[str] = None
    published_at: Optional[str] = None
    subscriber_count: int = 0
    video_count: int = 0
    total_view_count: int = 0
    avg_views_per_video: float = 0.0
    engagement_score: float = 0.0
    influencer_tier: Optional[str] = None
    bot_flag: bool = False
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[str] = None


class InfluencerUploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    total_rows_parsed: int
    inserted_count: int
    updated_count: int
    total_influencers_in_db: int
    sample_channels: List[str]


class InfluencerClusterResponse(BaseModel):
    cluster_id: int
    cluster_label: Optional[str] = None
    count: int
    influencer_tier_breakdown: dict[str, int] = {}
    influencers: List[InfluencerResponseItem]


# ── In-Memory Fast Cache ──────────────────────────────────────────────────────
_CLUSTER_CACHE: dict[tuple[int, bool, int], InfluencerClusterResponse] = {}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/upload-csv",
    response_model=InfluencerUploadResponse,
    summary="Upload influencer clusters CSV to database",
    description="Upload a CSV file (e.g., influencer_clusters.csv) to import or update influencer records in the database.",
)
async def upload_influencers_csv(
    file: UploadFile = File(..., description="CSV file containing influencer clusters data"),
    replace_all: bool = Form(False, description="Set True to clear existing influencers before importing"),
    db: Session = Depends(get_db),
) -> InfluencerUploadResponse:
    # 1. Validate file extension
    filename = file.filename or "upload.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a .csv file.",
        )

    # 2. Read bytes into pandas
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read CSV file: {e}",
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty.",
        )

    # 3. Validate required columns
    if "channel_id" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=(
                "Missing required column 'channel_id'. "
                f"Columns detected in CSV: {list(df.columns)}"
            ),
        )

    # 4. Optional wipe if replace_all requested
    if replace_all:
        try:
            db.query(Influencer).delete()
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reset influencers table: {e}",
            )

    # 5. Process and insert/upsert rows
    inserted_count = 0
    updated_count = 0
    sample_names: List[str] = []

    try:
        # Pre-fetch existing influencers into in-memory dictionary for fast upsert
        existing_map = {inf.channel_id: inf for inf in db.query(Influencer).all()}

        new_objects = []
        for _, row in df.iterrows():
            cid = str(row.get("channel_id", "")).strip()
            if not cid or cid.lower() in ("nan", "none"):
                continue

            name = str(row.get("channel_name", cid)).strip()
            desc_val = str(row.get("description", "")) if pd.notna(row.get("description")) else None
            country_val = str(row.get("country", "")) if pd.notna(row.get("country")) else None
            pub_val = str(row.get("published_at", "")) if pd.notna(row.get("published_at")) else None
            
            subs = int(row.get("subscriber_count", 0)) if pd.notna(row.get("subscriber_count")) else 0
            vids = int(row.get("video_count", 0)) if pd.notna(row.get("video_count")) else 0
            views = int(row.get("total_view_count", 0)) if pd.notna(row.get("total_view_count")) else 0
            avg_views = float(row.get("avg_views_per_video", 0.0)) if pd.notna(row.get("avg_views_per_video")) else 0.0

            kw = str(row.get("keywords", "")) if pd.notna(row.get("keywords")) else None
            clean_d = str(row.get("clean_description", "")) if pd.notna(row.get("clean_description")) else None
            clean_kw = str(row.get("clean_keywords", "")) if pd.notna(row.get("clean_keywords")) else None

            eng = float(row.get("engagement_score", 0.0)) if pd.notna(row.get("engagement_score")) else 0.0
            tier = str(row.get("influencer_tier", "")) if pd.notna(row.get("influencer_tier")) else None
            bot = bool(row.get("bot_flag", False)) if pd.notna(row.get("bot_flag")) else False

            c_id = int(row.get("cluster_id", 0)) if pd.notna(row.get("cluster_id")) else None
            c_lbl = str(row.get("cluster_label", "")) if pd.notna(row.get("cluster_label")) else None

            if len(sample_names) < 5 and name:
                sample_names.append(name)

            if cid in existing_map:
                # Update existing record in memory
                existing = existing_map[cid]
                existing.channel_name = name
                existing.description = desc_val
                existing.country = country_val
                existing.published_at = pub_val
                existing.subscriber_count = subs
                existing.video_count = vids
                existing.total_view_count = views
                existing.avg_views_per_video = avg_views
                existing.keywords = kw
                existing.clean_description = clean_d
                existing.clean_keywords = clean_kw
                existing.engagement_score = eng
                existing.influencer_tier = tier
                existing.bot_flag = bot
                existing.cluster_id = c_id
                existing.cluster_label = c_lbl
                updated_count += 1
            else:
                # Add new record
                new_inf = Influencer(
                    channel_id=cid,
                    channel_name=name,
                    description=desc_val,
                    country=country_val,
                    published_at=pub_val,
                    subscriber_count=subs,
                    video_count=vids,
                    total_view_count=views,
                    avg_views_per_video=avg_views,
                    keywords=kw,
                    clean_description=clean_d,
                    clean_keywords=clean_kw,
                    engagement_score=eng,
                    influencer_tier=tier,
                    bot_flag=bot,
                    cluster_id=c_id,
                    cluster_label=c_lbl,
                )
                new_objects.append(new_inf)
                existing_map[cid] = new_inf
                inserted_count += 1

        if new_objects:
            db.bulk_save_objects(new_objects)
        db.commit()

        # Invalidate in-memory cluster cache on new upload
        _CLUSTER_CACHE.clear()

        total_in_db = db.query(Influencer).count()

        return InfluencerUploadResponse(
            status="success",
            message=f"Successfully processed {len(df)} rows ({inserted_count} inserted, {updated_count} updated).",
            filename=filename,
            total_rows_parsed=len(df),
            inserted_count=inserted_count,
            updated_count=updated_count,
            total_influencers_in_db=total_in_db,
            sample_channels=sample_names,
        )

    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database ingestion error: {exc}",
        )


@router.get(
    "/cluster/{cluster_id}",
    response_model=InfluencerClusterResponse,
    summary="Get influencers based on cluster ID",
    description="Ultra-fast in-memory cached retrieval of all influencers in a cluster with tier analytics.",
)
@router.get(
    "/{cluster_id}",
    response_model=InfluencerClusterResponse,
    include_in_schema=False,
)
def get_influencers_by_cluster(
    cluster_id: int,
    bot_free_only: bool = Query(True, description="Exclude accounts flagged as bot/spam"),
    limit: int = Query(100, ge=1, le=500, description="Max influencers to return"),
    db: Session = Depends(get_db),
) -> InfluencerClusterResponse:
    """Retrieves all influencers belonging to a specified cluster ID, ranked by engagement."""
    cache_key = (cluster_id, bot_free_only, limit)
    if cache_key in _CLUSTER_CACHE:
        return _CLUSTER_CACHE[cache_key]

    query = db.query(Influencer).filter(Influencer.cluster_id == cluster_id)
    if bot_free_only:
        query = query.filter(Influencer.bot_flag.is_(False))

    items = (
        query.order_by(desc(Influencer.engagement_score), desc(Influencer.subscriber_count))
        .limit(limit)
        .all()
    )

    if not items:
        # Check if any exist without bot filter
        has_any = db.query(Influencer.channel_id).filter(Influencer.cluster_id == cluster_id).first()
        if not has_any:
            raise HTTPException(
                status_code=404,
                detail=f"No influencers found for cluster_id {cluster_id}.",
            )

    cluster_label = items[0].cluster_label if items else None
    tier_counts: dict[str, int] = {}

    results = []
    for inf in items:
        t = (inf.influencer_tier or "unknown").lower()
        tier_counts[t] = tier_counts.get(t, 0) + 1
        results.append(
            InfluencerResponseItem(
                channel_id=inf.channel_id,
                channel_name=inf.channel_name,
                country=inf.country,
                published_at=inf.published_at,
                subscriber_count=inf.subscriber_count or 0,
                video_count=inf.video_count or 0,
                total_view_count=inf.total_view_count or 0,
                avg_views_per_video=inf.avg_views_per_video or 0.0,
                engagement_score=inf.engagement_score or 0.0,
                influencer_tier=inf.influencer_tier,
                bot_flag=inf.bot_flag or False,
                cluster_id=inf.cluster_id,
                cluster_label=inf.cluster_label,
                description=inf.description,
                keywords=inf.keywords,
            )
        )

    res = InfluencerClusterResponse(
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        count=len(results),
        influencer_tier_breakdown=tier_counts,
        influencers=results,
    )

    _CLUSTER_CACHE[cache_key] = res
    return res
