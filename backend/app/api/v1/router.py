from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.v1.astro import router as astro_router

router = APIRouter(tags=["forecast"])


class FactorScore(BaseModel):
    name: str
    weight: int = Field(ge=0, description="Weight used in the composite score")
    value: float = Field(ge=0, le=1, description="Normalized factor value 0-1")
    contribution: float = Field(ge=0, le=100, description="Contribution to total score (0-100)")
    note: str | None = None


class ForecastSampleResponse(BaseModel):
    lat: float
    lng: float
    hours_ahead: int
    score: float
    rank: str
    factors: list[FactorScore]
    data_age_hours: int


def rank_from_score(score: float) -> str:
    if score >= 75:
        return "◎"
    if score >= 50:
        return "○"
    if score >= 30:
        return "△"
    return "×"


@router.get("/forecast/sample", response_model=ForecastSampleResponse)
def sample_forecast(lat: float = 35.0, lng: float = 139.0, hours_ahead: int = 0) -> ForecastSampleResponse:
    """Sample endpoint preserved for compatibility with the previous UI."""
    factor_defs = [
        {"name": "moon_phase", "weight": 20, "value": 0.9, "note": "月齢が良好"},
        {"name": "tide_window", "weight": 15, "value": 0.8, "note": "潮汐タイミング良"},
        {"name": "wind", "weight": 15, "value": 0.85, "note": "風弱め"},
        {"name": "wave_height", "weight": 20, "value": 0.75, "note": "波低め"},
        {"name": "sst", "weight": 15, "value": 0.7, "note": "水温20℃付近"},
        {"name": "cloud_cover", "weight": 10, "value": 0.6, "note": "雲量ほどほど"},
        {"name": "sns_signal", "weight": 5, "value": 0.5, "note": "SNS報告ふつう"},
    ]
    total_weight = sum(f["weight"] for f in factor_defs)
    weighted_score = sum(f["weight"] * f["value"] for f in factor_defs)
    score = (weighted_score / total_weight) * 100 if total_weight else 0

    factors = [
        FactorScore(
            name=f["name"],
            weight=f["weight"],
            value=f["value"],
            contribution=(f["weight"] * f["value"] / total_weight) * 100 if total_weight else 0,
            note=f.get("note"),
        )
        for f in factor_defs
    ]

    return ForecastSampleResponse(
        lat=lat,
        lng=lng,
        hours_ahead=hours_ahead,
        score=round(score, 1),
        rank=rank_from_score(score),
        factors=factors,
        data_age_hours=2,
    )


# Mount astro endpoints
router.include_router(astro_router)
