"""Creative Studio Service: Multimodal Ad Generation, CTR Estimation, and Campaigns."""

from datetime import datetime
import uuid
from app.models.schemas import (
    CreativeGenerateRequest,
    CreativeGenerateResponse,
    CreativeOption,
    CTRScoreRequest,
    CTRScoreResponse,
    CampaignCreate,
    CampaignItem,
)
from app.services.inventory_service import InventoryService
from app.services.trend_service import TrendService


# In-memory campaign store for demonstration
_CAMPAIGNS_STORE: list[CampaignItem] = [
    CampaignItem(
        campaign_id="CMP-001",
        campaign_name="Koramangala Boba Weekend Rush",
        product_id="SKU016",
        product_name="Brown Sugar Boba Kit",
        trend_topic="Brown Sugar Boba & Bubble Tea Hype",
        influencers_tagged=["@bengaluru_boba_diaries"],
        headline="Craving Boba? Brew Authentic Cafe-Style Tapioca in 5 Mins!",
        status="Active",
        budget_inr=7500.0,
        estimated_reach=28000,
        created_at="2026-05-01 11:30:00",
    ),
    CampaignItem(
        campaign_id="CMP-002",
        campaign_name="Indiranagar Clean Matcha Glow",
        product_id="SKU017",
        product_name="Matcha Green Tea Powder",
        trend_topic="Ceremonial Matcha & Morning Rituals",
        influencers_tagged=["@indiranagar_eats"],
        headline="Upgrade Your Morning Ritual with Kyoto-Grade Matcha.",
        status="Draft",
        budget_inr=5000.0,
        estimated_reach=18500,
        created_at="2026-05-02 16:45:00",
    ),
]


class CreativeService:
    @classmethod
    def generate_creatives(cls, req: CreativeGenerateRequest) -> CreativeGenerateResponse:
        """Generate high-converting ad copy, hashtags, and image prompts aligned with trends."""
        prod = InventoryService.get_product(req.product_id)
        trend_topic = None
        if req.trend_id:
            try:
                trend = TrendService.get_trend_by_id(req.trend_id)
                trend_topic = trend.trend.topic_name
            except Exception:
                trend_topic = "Local Bengaluru Food & Lifestyle Trend"

        pname = prod.product_name
        cat = prod.category

        options: list[CreativeOption] = [
            CreativeOption(
                headline=f"The {pname} Everyone in Bengaluru is Talking About!",
                caption=(
                    f"Stop settling for ordinary {cat.lower()}. Experience the trending sensation crafted for {req.target_audience.lower()}. "
                    f"Order yours today before local batches sell out! ⚡ Limited stocks available in store."
                ),
                call_to_action="Shop Now | In-Store & Fast Delivery",
                hashtags=[
                    f"#{pname.replace(' ', '')}",
                    "#BangaloreTrends",
                    f"#BLR{cat.replace('&', '')}",
                    "#SMEGrowth",
                    "#LocalFavourites",
                ],
                image_prompt=(
                    f"Award-winning commercial product photography of {pname}, aesthetic modern lighting, "
                    f"hyper-realistic 8k, vibrant colors, minimalist cafe table setup in Bengaluru, soft depth of field."
                ),
                estimated_ctr=4.85,
                ctr_grade="A+",
            ),
            CreativeOption(
                headline=f"Ready to Upgrade Your Daily Routine with {pname}?",
                caption=(
                    f"Elevate your lifestyle with authentic quality. Designed specially for {req.target_audience.lower()}, "
                    f"giving you the perfect balance of premium taste and everyday convenience. Tap below to grab exclusive introductory perks."
                ),
                call_to_action="Claim Your Pack Today",
                hashtags=[
                    f"#{pname.replace(' ', '')}",
                    "#HealthyLivingBLR",
                    "#TrendingNow",
                    "#CuratedForYou",
                ],
                image_prompt=(
                    f"Lifestyle portrait of young creator holding {pname}, radiant natural sunlight, modern urban Bangalore backdrop, "
                    f"cinematic color grading, Instagram aesthetic, crisp details."
                ),
                estimated_ctr=4.12,
                ctr_grade="A",
            ),
            CreativeOption(
                headline=f"Don't Miss the Trend: {pname} Back in Stock!",
                caption=(
                    f"High demand alert! Our batch of {pname} is moving fast across local neighborhood hubs. "
                    f"Loved by local creators and verified foodies. Pick up your pack or get it delivered straight to your door."
                ),
                call_to_action="Order Before It Sells Out",
                hashtags=[
                    f"#{pname.replace(' ', '')}",
                    "#StockAlert",
                    "#NeighborhoodFavorite",
                    "#MustTryBangalore",
                ],
                image_prompt=(
                    f"Close-up high-speed food photography of {pname}, dynamic splashing textures, studio lighting, "
                    f"vivid appetizing tones, sleek modern typography layout."
                ),
                estimated_ctr=3.65,
                ctr_grade="B+",
            ),
        ]

        return CreativeGenerateResponse(
            product_name=pname,
            trend_topic=trend_topic,
            options=options,
        )

    @classmethod
    def score_ctr(cls, req: CTRScoreRequest) -> CTRScoreResponse:
        """Estimate CTR and return quality breakdown based on copy heuristics."""
        headline_len = len(req.headline.split())
        caption_len = len(req.caption.split())

        # Heuristic scoring
        hook_score = 0.85 if 4 <= headline_len <= 12 else 0.65
        clarity_score = 0.90 if 15 <= caption_len <= 50 else 0.70
        urgency_score = 0.88 if any(w in (req.headline + req.caption).lower() for w in ["now", "today", "limited", "exclusive", "fast"]) else 0.60

        base_ctr = 2.2 + (hook_score * 1.5) + (clarity_score * 0.8) + (urgency_score * 0.6)
        estimated_ctr = round(base_ctr, 2)

        grade = "A+" if estimated_ctr >= 4.5 else ("A" if estimated_ctr >= 3.8 else ("B+" if estimated_ctr >= 3.0 else "B"))

        recommendations = []
        if hook_score < 0.8:
            recommendations.append("Make the headline punchier (aim for 5-9 words with clear benefits).")
        if urgency_score < 0.8:
            recommendations.append("Add subtle urgency or scarcity (e.g., 'Limited weekend batch').")
        if clarity_score < 0.8:
            recommendations.append("Ensure the caption concisely highlights who it's for.")
        if not recommendations:
            recommendations.append("Strong copy! Ready to launch on social channels.")

        return CTRScoreResponse(
            estimated_ctr=estimated_ctr,
            ctr_grade=grade,
            hook_strength=round(hook_score, 2),
            clarity_score=round(clarity_score, 2),
            urgency_score=round(urgency_score, 2),
            recommendations=recommendations,
        )

    @classmethod
    def list_campaigns(cls) -> list[CampaignItem]:
        return _CAMPAIGNS_STORE

    @classmethod
    def create_campaign(cls, data: CampaignCreate) -> CampaignItem:
        prod = InventoryService.get_product(data.product_id)
        trend_name = None
        if data.trend_id:
            try:
                trend = TrendService.get_trend_by_id(data.trend_id)
                trend_name = trend.trend.topic_name
            except Exception:
                pass

        item = CampaignItem(
            campaign_id=f"CMP-{uuid.uuid4().hex[:6].upper()}",
            campaign_name=data.campaign_name,
            product_id=data.product_id,
            product_name=prod.product_name,
            trend_topic=trend_name,
            influencers_tagged=data.influencer_ids,
            headline=data.headline,
            status="Active",
            budget_inr=data.budget_inr,
            estimated_reach=int(data.budget_inr * 3.8),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        _CAMPAIGNS_STORE.insert(0, item)
        return item
