from functools import lru_cache
from typing import List, Optional

import json
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    env: str = "local"
    debug: bool = True
    project_name: str = "BlueWave Forecast API"
    api_v1_prefix: str = "/api/v1"
    # Allow both list and comma-separated string to avoid env parse errors
    backend_cors_origins: List[str] | str = ["http://localhost:5173"]

    database_url: str = "postgresql+psycopg://yakouchu:password@localhost:5432/yakouchu"
    openweather_api_key: Optional[str] = None
    tide_api_key: Optional[str] = None
    google_maps_api_key: Optional[str] = None
    sentry_dsn: Optional[str] = None

    # Elevation / horizon settings
    elevation_mode: Optional[str] = None
    elevation_zoom: Optional[int] = None
    horizon_method: Optional[str] = None
    elevation_bearing_step: Optional[float] = None
    horizon_step_m: Optional[float] = None
    horizon_max_distance_m: Optional[float] = None
    horizon_smooth_window: Optional[int] = None

    @field_validator("backend_cors_origins", mode="before")
    @classmethod
    def split_cors(cls, v: str | List[str]) -> List[str]:
        try:
            if isinstance(v, list):
                return [str(origin).strip() for origin in v if str(origin).strip()]
            if isinstance(v, str):
                text = v.strip()
                # Try JSON list first
                if (text.startswith("[") and text.endswith("]")) or (text.startswith('"') and text.endswith('"')):
                    try:
                        data = json.loads(text)
                        if isinstance(data, list):
                            return [str(origin).strip() for origin in data if str(origin).strip()]
                    except Exception:
                        pass
                # Fallback: comma-separated string
                return [origin.strip() for origin in text.split(",") if origin.strip()]
        except Exception:
            pass
        # Fallback to default if parsing fails
        return ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
