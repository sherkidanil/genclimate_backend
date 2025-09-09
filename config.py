import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _ConfigDict(env_prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(
        env_file=".env.secret",  # т.к. .env был закоммичен
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix=env_prefix,
        extra="allow",
    )


class AI4SSettings(BaseSettings):
    model_config = _ConfigDict("ai4s_")
    secret_key: str


class S3Settings(BaseSettings):
    model_config = _ConfigDict("s3_")
    access_key_id: str
    secret_access_key: str
    bucket_name: str = ""
    region_name: str = "ru-moscow-1"
    endpoint_url: str = "https://obs.ru-moscow-1.hc.sbercloud.ru"


class Settings(BaseSettings):
    """Centralised runtime configuration."""

    # Directory paths (overridable via env‑vars)
    # forecast_dir_aifs: Path = Path(os.getenv("FORECAST_DIR_AIFS", "/data/predictions"))
    # forecast_dir_s2s: Path = Path(os.getenv("FORECAST_DIR_S2S", "/data/s2s_predictions"))
    forecast_dir_aifs: Path = Path("/data/predictions")
    forecast_dir_s2s: Path = Path("/data/s2s_predictions")

    # Cache (seconds before the in‑memory dataset is refreshed)
    cache_ttl: int = int(os.getenv("FORECAST_CACHE_TTL", 3600 * 6))

    # External geocoder (OpenStreetMap Nominatim by default)
    geocoder_endpoint: str = os.getenv("GEOCODER_ENDPOINT", "https://nominatim.openstreetmap.org/search")
    geocoder_user_agent: str = "weather-service"

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    ai4s: AI4SSettings = AI4SSettings()  # type: ignore
    s3: S3Settings = S3Settings()  # type: ignore

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
