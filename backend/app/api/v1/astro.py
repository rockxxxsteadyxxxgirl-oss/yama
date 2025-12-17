from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.services.astro import Snapshot, make_snapshot

router = APIRouter(prefix="/astro", tags=["astro"])


class BodyPositionModel(BaseModel):
    name: str
    altitude_deg: float
    azimuth_deg: float

    model_config = {"from_attributes": True}


class SunMoonDataModel(BaseModel):
    sun: BodyPositionModel
    moon: BodyPositionModel
    moon_phase_deg: float
    moon_illumination: float
    sunrise: str | None
    sunset: str | None
    moonrise: str | None
    moonset: str | None

    model_config = {"from_attributes": True}


class StarPointModel(BaseModel):
    name: str
    magnitude: float
    altitude_deg: float
    azimuth_deg: float

    model_config = {"from_attributes": True}


class HorizonProfileModel(BaseModel):
    bearings_deg: list[float]
    elevations_deg: list[float]
    source: str

    model_config = {"from_attributes": True}


class FieldOfViewModel(BaseModel):
    horizontal_deg: float
    vertical_deg: float

    model_config = {"from_attributes": True}

class ConstellationSegmentModel(BaseModel):
    start_alt_deg: float
    start_az_deg: float
    end_alt_deg: float
    end_az_deg: float

    model_config = {"from_attributes": True}


class ConstellationModel(BaseModel):
    id: str
    name: str
    abbr: str
    label_alt_deg: float
    label_az_deg: float
    segments: list[ConstellationSegmentModel]

    model_config = {"from_attributes": True}


class SnapshotModel(BaseModel):
    timestamp_utc: str
    location: tuple[float, float]
    fov: FieldOfViewModel
    sunmoon: SunMoonDataModel
    stars: list[StarPointModel]
    horizon: HorizonProfileModel
    constellations: list[ConstellationModel]

    model_config = {"from_attributes": True}


@router.get("/snapshot", response_model=SnapshotModel)
async def get_snapshot(
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    dt: str = Query(..., description="ISO8601 datetime (with offset or UTC)"),
    focal_length_mm: float = Query(24.0, description="Focal length in mm"),
    sensor_format: str = Query("full", description="full|aps-c|mft"),
) -> SnapshotModel:
    snap: Snapshot = await make_snapshot(lat, lng, dt, focal_length_mm, sensor_format)
    return SnapshotModel.model_validate(snap)
