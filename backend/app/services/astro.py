from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import httpx
import numpy as np
from skyfield import almanac
from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos

# Load ephemeris and timescale once. The first call downloads the bsp/hipparcos
# files to the Skyfield cache (~ a few MB).
ts = load.timescale()
eph = load("de421.bsp")
STAR_LIMIT_MAG = float(os.getenv("STAR_LIMIT_MAG", "2.5"))
STAR_MAX_COUNT = int(os.getenv("STAR_MAX_COUNT", "120"))
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONSTELLATION_FILE = DATA_DIR / "index.json"
ELEVATION_CHUNK_SIZE = int(os.getenv("ELEVATION_CHUNK_SIZE", "90"))
ELEVATION_BEARING_STEP = float(os.getenv("ELEVATION_BEARING_STEP", "2.5"))
ELEVATION_DISTANCES = os.getenv(
    "ELEVATION_DISTANCES",
    "200,500,1000,2000,3000,4000,5000",
)
ELEVATION_MODE = os.getenv("ELEVATION_MODE", "opentopo").lower()
ELEVATION_ZOOM = int(os.getenv("ELEVATION_ZOOM", "15"))
GSI_NO_DATA = -9999.0
HORIZON_METHOD = os.getenv("HORIZON_METHOD", "ray").lower()  # ray | list
HORIZON_STEP_M = float(os.getenv("HORIZON_STEP_M", "200"))
HORIZON_MAX_DISTANCE_M = float(os.getenv("HORIZON_MAX_DISTANCE_M", "5000"))
HORIZON_SMOOTH_WINDOW = int(os.getenv("HORIZON_SMOOTH_WINDOW", "3"))
HORIZON_REFRACTION_K = float(os.getenv("HORIZON_REFRACTION_K", "0.0"))


@dataclass
class BodyPosition:
    name: str
    altitude_deg: float
    azimuth_deg: float


@dataclass
class SunMoonData:
    sun: BodyPosition
    moon: BodyPosition
    moon_phase_deg: float
    moon_illumination: float
    sunrise: Optional[str]
    sunset: Optional[str]
    moonrise: Optional[str]
    moonset: Optional[str]


@dataclass
class StarPoint:
    name: str
    magnitude: float
    altitude_deg: float
    azimuth_deg: float


@dataclass
class HorizonProfile:
    bearings_deg: List[float]
    elevations_deg: List[float]
    source: str


@dataclass
class FieldOfView:
    horizontal_deg: float
    vertical_deg: float


@dataclass
class Snapshot:
    timestamp_utc: str
    location: tuple[float, float]
    fov: FieldOfView
    sunmoon: SunMoonData
    stars: List[StarPoint]
    horizon: HorizonProfile
    constellations: List["ConstellationOutline"]


@dataclass
class ConstellationSegment:
    start_alt_deg: float
    start_az_deg: float
    end_alt_deg: float
    end_az_deg: float


@dataclass
class ConstellationOutline:
    id: str
    name: str
    abbr: str
    label_alt_deg: float
    label_az_deg: float
    segments: List[ConstellationSegment]


SENSOR_SPECS = {
    "full": (36.0, 24.0),
    "aps-c": (23.6, 15.6),
    "mft": (17.3, 13.0),
}

_STAR_CACHE: dict[str, list[tuple[str, float, Star]]] = {}
HIP_STAR_MAP: dict[int, Star] = {}
HIP_MAG_MAP: dict[int, float] = {}
CONSTELLATION_DEFS: list[dict] = []
GSI_TILE_CACHE: dict[tuple[str, int, int, int], list[list[float | None]]] = {}


def _ensure_hip_catalog() -> None:
    if HIP_STAR_MAP:
        return
    with load.open(hipparcos.URL) as f:
        df = hipparcos.load_dataframe(f)
    for hip_id, row in df.iterrows():
        hip = int(hip_id)
        HIP_STAR_MAP[hip] = Star.from_dataframe(row)
        HIP_MAG_MAP[hip] = float(row["magnitude"])


def _load_constellation_defs() -> list[dict]:
    if CONSTELLATION_DEFS:
        return CONSTELLATION_DEFS
    if not CONSTELLATION_FILE.exists():
        return CONSTELLATION_DEFS
    data = json.loads(CONSTELLATION_FILE.read_text(encoding="utf-8"))
    defs: list[dict] = []
    for c in data.get("constellations", []):
        cid = c.get("id", "")
        abbr = cid.split()[-1] if cid else ""
        common = c.get("common_name") or {}
        name = common.get("native") or common.get("english") or abbr or cid
        segments: list[tuple[int, int]] = []
        label_hip = None
        lines = c.get("lines", []) or []
        if lines and lines[0]:
            label_hip = int(lines[0][0])
        for poly in lines:
            if not poly or len(poly) < 2:
                continue
            for i in range(len(poly) - 1):
                segments.append((int(poly[i]), int(poly[i + 1])))
        if segments:
            defs.append({"id": cid, "abbr": abbr, "name": name, "segments": segments, "label_hip": label_hip})
    CONSTELLATION_DEFS.extend(defs)
    return CONSTELLATION_DEFS


def parse_dt(dt_text: str) -> datetime:
    """Parse ISO 8601 datetime. Naive to UTC."""
    txt = dt_text.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute_fov(focal_length_mm: float, sensor_format: str) -> FieldOfView:
    sensor = SENSOR_SPECS.get(sensor_format, SENSOR_SPECS["full"])
    w, h = sensor
    fov_h = math.degrees(2 * math.atan((w / 2) / focal_length_mm))
    fov_v = math.degrees(2 * math.atan((h / 2) / focal_length_mm))
    return FieldOfView(horizontal_deg=fov_h, vertical_deg=fov_v)


def _bright_stars(limit_mag: float = STAR_LIMIT_MAG, max_count: int = STAR_MAX_COUNT) -> list[tuple[str, float, Star]]:
    """Load bright star catalog (Hipparcos) and cache it."""
    cache_key = f"{limit_mag}-{max_count}"
    if cache_key in _STAR_CACHE:
        return _STAR_CACHE[cache_key]

    _ensure_hip_catalog()
    sorted_hips = sorted(HIP_MAG_MAP.items(), key=lambda kv: kv[1])
    selected = [(hip, mag) for hip, mag in sorted_hips if mag <= limit_mag][:max_count]

    stars: list[tuple[str, float, Star]] = []
    for hip_id, mag in selected:
        star = HIP_STAR_MAP.get(hip_id)
        if not star:
            continue
        stars.append((f"HIP {hip_id}", mag, star))
    _STAR_CACHE[cache_key] = stars
    return _STAR_CACHE[cache_key]


def _to_altaz(observer, body, t):
    apparent = observer.at(t).observe(body).apparent()
    alt, az, _ = apparent.altaz()
    return alt.degrees, az.degrees


def _constellations_altaz(earth_topo, t: datetime) -> List[ConstellationOutline]:
    """Project constellation segments to alt/az using precomputed HIP pairs."""
    _ensure_hip_catalog()
    defs = _load_constellation_defs()
    st = ts.from_datetime(t)
    outlines: list[ConstellationOutline] = []

    for c in defs:
        segments: list[ConstellationSegment] = []
        for hip1, hip2 in c["segments"]:
            star1 = HIP_STAR_MAP.get(hip1)
            star2 = HIP_STAR_MAP.get(hip2)
            if not star1 or not star2:
                continue
            alt1, az1 = _to_altaz(earth_topo, star1, st)
            alt2, az2 = _to_altaz(earth_topo, star2, st)
            # Skip segments fully far below horizon to reduce noise.
            if alt1 < -12 and alt2 < -12:
                continue
            segments.append(ConstellationSegment(alt1, az1, alt2, az2))

        if not segments:
            continue

        label_alt, label_az = segments[0].start_alt_deg, segments[0].start_az_deg
        label_hip = c.get("label_hip")
        if label_hip and label_hip in HIP_STAR_MAP:
            label_alt, label_az = _to_altaz(earth_topo, HIP_STAR_MAP[label_hip], st)

        outlines.append(
            ConstellationOutline(
                id=c.get("id", ""),
                name=c.get("name", c.get("abbr", "")),
                abbr=c.get("abbr", ""),
                label_alt_deg=label_alt,
                label_az_deg=label_az,
                segments=segments,
            )
        )

    return outlines


def _tile_xy(lat: float, lng: float, z: int) -> tuple[int, int, float, float]:
    """Convert lat/lng to tile x,y and fractional offsets within tile for XYZ scheme."""
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    xtile_f = (lng + 180.0) / 360.0 * n
    ytile_f = (1.0 - (math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi)) / 2.0 * n
    xtile = int(math.floor(xtile_f))
    ytile = int(math.floor(ytile_f))
    fx = xtile_f - xtile  # 0-1
    fy = ytile_f - ytile  # 0-1 (top to bottom)
    return xtile, ytile, fx, fy


def _gsi_tile_urls(x: int, y: int, z: int) -> list[str]:
    base = "https://cyberjapandata.gsi.go.jp/xyz"
    datasets = ["dem5a", "dem5b", "dem5c"]
    return [f"{base}/{d}/{z}/{x}/{y}.txt" for d in datasets]


async def _fetch_gsi_tile(x: int, y: int, z: int, client: httpx.AsyncClient) -> list[list[float | None]] | None:
    """Fetch GSI DEM tile (dem5a/b/c) and return 2D array of elevations or None."""
    for url in _gsi_tile_urls(x, y, z):
        cache_key = (url, x, y, z)
        if cache_key in GSI_TILE_CACHE:
            return GSI_TILE_CACHE[cache_key]
        try:
            resp = await client.get(url, timeout=20)
            resp.raise_for_status()
            lines = resp.text.strip().splitlines()
            data: list[list[float | None]] = []
            for line in lines:
                row: list[float | None] = []
                for token in line.strip().split():
                    try:
                        v = float(token)
                        row.append(None if v <= GSI_NO_DATA else v)
                    except ValueError:
                        row.append(None)
                if row:
                    data.append(row)
            if data:
                GSI_TILE_CACHE[cache_key] = data
                return data
        except Exception:
            continue
    return None


def _bilinear_sample(tile: list[list[float | None]], fx: float, fy: float) -> float | None:
    """Bilinear sample tile (256x256) at fractional offsets fx, fy (0-1)."""
    if not tile:
        return None
    h = len(tile)
    w = len(tile[0])
    px = fx * (w - 1)
    py = fy * (h - 1)
    x0 = int(math.floor(px))
    x1 = min(x0 + 1, w - 1)
    y0 = int(math.floor(py))
    y1 = min(y0 + 1, h - 1)
    dx = px - x0
    dy = py - y0

    def val(x: int, y: int) -> float | None:
        try:
            return tile[y][x]
        except Exception:
            return None

    q11 = val(x0, y0)
    q21 = val(x1, y0)
    q12 = val(x0, y1)
    q22 = val(x1, y1)
    vals = [q for q in [q11, q21, q12, q22] if q is not None]
    if not vals:
        return None

    # Fallback to nearest if any corner missing heavily
    if any(v is None for v in [q11, q21, q12, q22]):
        nearest_candidates = [(abs(px - x0) + abs(py - y0), q11), (abs(px - x1) + abs(py - y0), q21),
                              (abs(px - x0) + abs(py - y1), q12), (abs(px - x1) + abs(py - y1), q22)]
        nearest = [v for d, v in sorted(nearest_candidates, key=lambda t: t[0]) if v is not None]
        return nearest[0] if nearest else None

    # Bilinear interpolation
    return (
        q11 * (1 - dx) * (1 - dy)
        + q21 * dx * (1 - dy)
        + q12 * (1 - dx) * dy
        + q22 * dx * dy
    )


async def _gsi_elevation(lat: float, lng: float, client: httpx.AsyncClient) -> float | None:
    """Sample elevation from GSI DEM5A/B/C tiles."""
    xtile, ytile, fx, fy = _tile_xy(lat, lng, ELEVATION_ZOOM)
    tile = await _fetch_gsi_tile(xtile, ytile, ELEVATION_ZOOM, client)
    if tile is None:
        return None
    return _bilinear_sample(tile, fx, fy)


def _sun_moon(topo, earth_topo, t: datetime) -> SunMoonData:
    st = ts.from_datetime(t)
    sun_alt, sun_az = _to_altaz(earth_topo, eph["sun"], st)
    moon_alt, moon_az = _to_altaz(earth_topo, eph["moon"], st)

    # Moon phase: 0=new, 180=full.
    phase_angle = almanac.moon_phase(eph, st).degrees
    illum = (1 - math.cos(math.radians(phase_angle))) / 2

    def _find_events(func):
        try:
            start = st - 1
            end = st + 1
            t_events, events = almanac.find_discrete(start, end, func)
            rises, sets = None, None
            for t_ev, ev in zip(t_events, events):
                if ev == True or ev == 1:
                    rises = t_ev.utc_strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    sets = t_ev.utc_strftime("%Y-%m-%dT%H:%M:%SZ")
            return rises, sets
        except Exception:
            return None, None

    sun_rise, sun_set = _find_events(almanac.risings_and_settings(eph, eph["sun"], topo))
    moon_rise, moon_set = _find_events(almanac.risings_and_settings(eph, eph["moon"], topo))

    return SunMoonData(
        sun=BodyPosition("sun", sun_alt, sun_az),
        moon=BodyPosition("moon", moon_alt, moon_az),
        moon_phase_deg=phase_angle,
        moon_illumination=illum,
        sunrise=sun_rise,
        sunset=sun_set,
        moonrise=moon_rise,
        moonset=moon_set,
    )


def _star_positions(earth_topo, t: datetime) -> List[StarPoint]:
    st = ts.from_datetime(t)
    results: list[StarPoint] = []
    for name, mag, star in _bright_stars():
        alt, az = _to_altaz(earth_topo, star, st)
        results.append(StarPoint(name=name, magnitude=mag, altitude_deg=alt, azimuth_deg=az))
    # Keep only above horizon-ish (-2 deg to allow near-rising).
    return [s for s in results if s.altitude_deg > -2]


def _destination_point(lat: float, lng: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    R = 6371000.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lng)
    brng = math.radians(bearing_deg)
    ang_dist = distance_m / R
    lat2 = math.asin(math.sin(lat1) * math.cos(ang_dist) + math.cos(lat1) * math.sin(ang_dist) * math.cos(brng))
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(ang_dist) * math.cos(lat1),
        math.cos(ang_dist) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _curvature_drop(distance_m: float) -> float:
    """Curvature + refraction drop in meters (positive = drop)."""
    if distance_m <= 0:
        return 0.0
    # clamp k so denominator stays positive
    k = max(-0.2, min(0.99, HORIZON_REFRACTION_K))
    Re = 6371000.0 / (1.0 - k)
    return (distance_m * distance_m) / (2.0 * Re)


async def _batch_elevations(coords: List[tuple[float, float]], client: httpx.AsyncClient, url: str) -> List[Optional[float]]:
    """Fetch elevations from OpenTopodata-like API in chunks. Returns list with None on failure."""
    if not coords:
        return []

    # GSI DEMモードはタイルを直接サンプリング
    if ELEVATION_MODE == "gsi_dem":
        vals: list[Optional[float]] = []
        for lat, lng in coords:
            try:
                elev = await _gsi_elevation(lat, lng, client)
            except Exception:
                elev = None
            vals.append(elev)
        return vals

    # OpenTopodata系
    chunk_size = max(10, ELEVATION_CHUNK_SIZE)
    results: list[Optional[float]] = []

    for i in range(0, len(coords), chunk_size):
        chunk = coords[i : i + chunk_size]
        loc_param = "|".join(f"{lat},{lng}" for lat, lng in chunk)
        try:
            resp = await client.get(url, params={"locations": loc_param}, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            chunk_results = data.get("results", []) if isinstance(data, dict) else []
            elevs = [r.get("elevation") if isinstance(r, dict) else None for r in chunk_results]
            # 欠けがあっても位置合わせを維持するために長さを調整
            if len(elevs) < len(chunk):
                elevs.extend([None] * (len(chunk) - len(elevs)))
            results.extend(elevs[: len(chunk)])
        except Exception:
            results.extend([None] * len(chunk))

    if len(results) < len(coords):
        results.extend([None] * (len(coords) - len(results)))

    return results


async def compute_horizon(lat: float, lng: float) -> HorizonProfile:
    """Estimate horizon by sampling elevations around the point. Fallback to flat."""
    api_url = os.getenv("ELEVATION_API_URL", "https://api.opentopodata.org/v1/aster30m")
    # bearings
    bearings = [float(b) for b in np.arange(0, 360, max(0.1, ELEVATION_BEARING_STEP))]

    # list方式: 従来の距離サンプル
    distances_list = (
        [float(x) for x in ELEVATION_DISTANCES.split(",") if x.strip().replace(".", "").isdigit()]
        if ELEVATION_DISTANCES
    else [200, 500, 1000, 2000, 3000, 4000, 5000]
)
    if not distances_list:
        distances_list = [200, 500, 1000, 2000, 4000, 8000, 12000, 20000, 30000]

    # ray方式: 近距離高密度 → 中距離 → 指定ステップ
    ray_step = max(10.0, HORIZON_STEP_M)
    ray_max = max(ray_step, HORIZON_MAX_DISTANCE_M)
    stage1_end = 1500.0
    stage2_end = 3000.0

    ray_distances: list[float] = [0.0]
    d = 0.0
    while d < ray_max:
        if d < stage1_end:
            step = 50.0
        elif d < stage2_end:
            step = 100.0
        else:
            step = ray_step
        next_d = min(ray_max, d + step)
        if next_d - ray_distances[-1] > 1e-6:
            ray_distances.append(next_d)
        d = next_d

    coords: list[tuple[float, float]] = []
    method = HORIZON_METHOD
    if method == "ray":
        for b in bearings:
            for d in ray_distances:
                coords.append(_destination_point(lat, lng, b, d))
    else:
        for b in bearings:
            for d in distances_list:
                coords.append(_destination_point(lat, lng, b, d))

    elevations = []
    source = "flat"

    async with httpx.AsyncClient() as client:
        batch = await _batch_elevations(coords, client, api_url)
        if any(v is not None for v in batch):
            source = "gsi-dem" if ELEVATION_MODE == "gsi_dem" else "opentopodata"
            if method == "ray":
                stride = len(ray_distances)
                for i, b in enumerate(bearings):
                    base_idx = i * stride
                    vals = batch[base_idx : base_idx + stride]
                    base = vals[0] if vals and vals[0] is not None else 0.0
                    max_angle = 0.0
                    for d, elev in zip(ray_distances[1:], vals[1:]):
                        if elev is None:
                            continue
                        drop = _curvature_drop(d)
                        dy = elev - base - drop
                        ang = math.degrees(math.atan2(dy, d))
                        max_angle = max(max_angle, ang)
                    elevations.append(max_angle)
            else:
                stride = len(distances_list)
                base_elevs = batch[0::stride]  # first distance for each bearing
                for i, b in enumerate(bearings):
                    base = base_elevs[i] if base_elevs[i] is not None else 0.0
                    max_angle = 0.0
                    for d, elev in zip(distances_list, batch[i * stride : (i + 1) * stride]):
                        if elev is None:
                            continue
                        drop = _curvature_drop(d)
                        dy = elev - base - drop
                        ang = math.degrees(math.atan2(dy, d))
                        max_angle = max(max_angle, ang)
                    elevations.append(max_angle)

    if not elevations:
        elevations = [0.0 for _ in bearings]

    # smoothing (small window moving average)
    win = max(1, HORIZON_SMOOTH_WINDOW)
    if win > 1 and len(elevations) >= win:
        smoothed = []
        for i in range(len(elevations)):
            acc = 0.0
            cnt = 0
            for k in range(-win // 2, win // 2 + 1):
                idx = (i + k) % len(elevations)
                acc += elevations[idx]
                cnt += 1
            smoothed.append(acc / cnt if cnt else elevations[i])
        elevations = smoothed

    return HorizonProfile(bearings_deg=bearings, elevations_deg=elevations, source=source + f" ({method})")


async def make_snapshot(lat: float, lng: float, dt_text: str, focal_length_mm: float, sensor_format: str) -> Snapshot:
    dt_utc = parse_dt(dt_text)
    fov = compute_fov(focal_length_mm, sensor_format)
    topo = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lng)
    earth_topo = eph["earth"] + topo
    sunmoon = _sun_moon(topo, earth_topo, dt_utc)
    stars = _star_positions(earth_topo, dt_utc)
    horizon = await compute_horizon(lat, lng)
    constellations = _constellations_altaz(earth_topo, dt_utc)

    return Snapshot(
        timestamp_utc=dt_utc.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        location=(lat, lng),
        fov=fov,
        sunmoon=sunmoon,
        stars=stars,
        horizon=horizon,
        constellations=constellations,
    )
