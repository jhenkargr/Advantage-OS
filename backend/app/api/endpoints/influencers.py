"""Influencer Matchmaker API Endpoints."""

from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import InfluencerItem, InfluencerMatchResponse
from app.services.influencer_service import InfluencerService

router = APIRouter(prefix="/influencers", tags=["Influencer Matchmaker"])


@router.get("", response_model=list[InfluencerItem], summary="List vetted micro-influencers")
def list_influencers(min_authenticity: float = Query(0.70, ge=0.0, le=1.0, description="Minimum authenticity score")):
    """List local micro-influencers clustered by niche and filtered for bot-free authentic engagement."""
    try:
        return InfluencerService.list_all_influencers(min_authenticity=min_authenticity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/match/{trend_id}", response_model=InfluencerMatchResponse, summary="Match influencers for trend")
def match_influencers(trend_id: str):
    """Compute affinity scores and match top relevant micro-influencers for a specified trend."""
    try:
        return InfluencerService.match_influencers_for_trend(trend_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
