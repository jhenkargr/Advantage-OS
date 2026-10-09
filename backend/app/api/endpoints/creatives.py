"""Creative Studio and Campaign Management API Endpoints."""

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    CreativeGenerateRequest,
    CreativeGenerateResponse,
    CTRScoreRequest,
    CTRScoreResponse,
    CampaignCreate,
    CampaignItem,
)
from app.services.creative_service import CreativeService

router = APIRouter(prefix="/creatives", tags=["Creative Studio & Campaigns"])


@router.post("/generate", response_model=CreativeGenerateResponse, summary="Generate AI ad creatives & image prompts")
def generate_creatives(request: CreativeGenerateRequest):
    """
    Generate trend-aligned ad captions, punchy headlines, target hashtags,
    and commercial image generation prompts for Stable Diffusion/DALL-E.
    """
    try:
        return CreativeService.generate_creatives(request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/score-ctr", response_model=CTRScoreResponse, summary="Predict Click-Through-Rate (CTR) for ad copy")
def score_ctr(request: CTRScoreRequest):
    """
    Evaluate marketing copy using CTR estimation models:
    Returns estimated CTR %, letter grade, hook strength, clarity score, and actionable copy improvements.
    """
    try:
        return CreativeService.score_ctr(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/campaigns", response_model=list[CampaignItem], summary="List marketing campaigns")
def list_campaigns():
    """List all created marketing campaigns across SME products."""
    try:
        return CreativeService.list_campaigns()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/campaigns", response_model=CampaignItem, status_code=201, summary="Launch new campaign")
def create_campaign(campaign: CampaignCreate):
    """Create and track a new campaign linking product, trend, influencers, and budget."""
    try:
        return CreativeService.create_campaign(campaign)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
