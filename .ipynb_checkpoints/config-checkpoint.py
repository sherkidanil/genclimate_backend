import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


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
    geocoder_endpoint: str = os.getenv(
        "GEOCODER_ENDPOINT", "https://nominatim.openstreetmap.org/search"
    )
    geocoder_user_agent: str = "weather-service"

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
