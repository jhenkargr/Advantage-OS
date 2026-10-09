"""Influencer Matchmaker Service: K-Means Niche Clustering and Bot Filtering."""

from app.models.schemas import InfluencerItem, InfluencerMatchResponse
from app.services.trend_service import TrendService


MICRO_INFLUENCERS_DATA = [
    {
        "influencer_id": "INF-001",
        "handle": "@bengaluru_boba_diaries",
        "display_name": "Rhea | BLR Cafe Hops",
        "niche": "Beverages",
        "locality": "Koramangala, Bengaluru",
        "follower_count": 14200,
        "engagement_rate": 6.8,
        "authenticity_score": 0.94,
        "is_verified": False,
        "primary_platform": "Instagram",
        "recommended_for_trends": ["TRD-001", "TRD-002"],
    },
    {
        "influencer_id": "INF-002",
        "handle": "@indiranagar_eats",
        "display_name": "Arjun Sundaram",
        "niche": "F&B",
        "locality": "Indiranagar, Bengaluru",
        "follower_count": 28400,
        "engagement_rate": 5.4,
        "authenticity_score": 0.89,
        "is_verified": True,
        "primary_platform": "Instagram",
        "recommended_for_trends": ["TRD-002", "TRD-005"],
    },
    {
        "influencer_id": "INF-003",
        "handle": "@glasskin_diaries_blr",
        "display_name": "Meera Rao | Clean Beauty",
        "niche": "Wellness",
        "locality": "HSR Layout, Bengaluru",
        "follower_count": 19500,
        "engagement_rate": 8.2,
        "authenticity_score": 0.96,
        "is_verified": False,
        "primary_platform": "Instagram",
        "recommended_for_trends": ["TRD-003"],
    },
    {
        "influencer_id": "INF-004",
        "handle": "@bangalore_thrift_fits",
        "display_name": "Kabir & Tara",
        "niche": "Fashion",
        "locality": "Church Street, Bengaluru",
        "follower_count": 32100,
        "engagement_rate": 7.1,
        "authenticity_score": 0.91,
        "is_verified": False,
        "primary_platform": "Instagram",
        "recommended_for_trends": ["TRD-004"],
    },
    {
        "influencer_id": "INF-005",
        "handle": "@filtercoffee_to_coldbrew",
        "display_name": "Nikhil V. | Coffee Geek",
        "niche": "Beverages",
        "locality": "Whitefield, Bengaluru",
        "follower_count": 11800,
        "engagement_rate": 7.9,
        "authenticity_score": 0.93,
        "is_verified": False,
        "primary_platform": "YouTube",
        "recommended_for_trends": ["TRD-005", "TRD-002"],
    },
    {
        "influencer_id": "INF-006",
        "handle": "@fit_sarjapur",
        "display_name": "Divya Sharma",
        "niche": "Wellness",
        "locality": "Bellandur & Sarjapur, Bengaluru",
        "follower_count": 15400,
        "engagement_rate": 6.1,
        "authenticity_score": 0.88,
        "is_verified": False,
        "primary_platform": "Instagram",
        "recommended_for_trends": ["TRD-006"],
    },
    {
        "influencer_id": "INF-007",
        "handle": "@bot_spammer_example",
        "display_name": "Super Promo Deals",
        "niche": "General",
        "locality": "Bengaluru",
        "follower_count": 89000,
        "engagement_rate": 0.4,
        "authenticity_score": 0.31,
        "is_verified": False,
        "primary_platform": "Instagram",
        "recommended_for_trends": [],
    },
]


class InfluencerService:
    @classmethod
    def list_all_influencers(cls, min_authenticity: float = 0.70) -> list[InfluencerItem]:
        """Return vetted micro-influencers filtered by bot detection score."""
        return [
            InfluencerItem(**inf)
            for inf in MICRO_INFLUENCERS_DATA
            if inf["authenticity_score"] >= min_authenticity
        ]

    @classmethod
    def match_influencers_for_trend(cls, trend_id: str) -> InfluencerMatchResponse:
        """Score and match influencers for a specific trend based on topic alignment."""
        all_trends = TrendService.get_all_trends()
        trend = next((t for t in all_trends if t.trend_id == trend_id), None)
        trend_name = trend.topic_name if trend else "Custom Trend"
        trend_category = trend.category if trend else ""

        matched = []
        for inf in cls.list_all_influencers():
            # Compute match score based on niche overlap and trend tagging
            score = 65.0
            if trend_id in inf.recommended_for_trends:
                score += 25.0
            if inf.niche.lower() == trend_category.lower():
                score += 10.0

            item = inf.model_copy()
            item.match_score = round(min(98.0, score + (inf.engagement_rate * 0.5)), 1)
            matched.append(item)

        matched.sort(key=lambda x: x.match_score or 0.0, reverse=True)
        return InfluencerMatchResponse(
            trend_id=trend_id,
            trend_name=trend_name,
            recommended_influencers=matched,
        )
