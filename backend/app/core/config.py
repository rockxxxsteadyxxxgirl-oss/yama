from functools import lru_cache
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    env: str = "local"
    debug: bool = True
    project_name: str = "BlueWave Forecast API"
    api_v1_prefix: str = "/api/v1"
    backend_cors_origins: List[str] = ["http://localhost:5173"]

    database_url: str = "postgresql+psycopg://yakouchu:password@localhost:5432/yakouchu"
    openweather_api_key: Optional[str] = None
    tide_api_key: Optional[str] = None
    google_maps_api_key: Optional[str] = None
    sentry_dsn: Optional[str] = None

    @field_validator("backend_cors_origins", mode="before")
    @classmethod
    def split_cors(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
