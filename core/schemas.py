from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, conint


class ParamsEnum(str, Enum):
    simple = "simple"
    surface = "surface"
    full = "full"


class ModeEnum(str, Enum):
    base = "base"
    ens = "ens"  # оставляем как у тебя; можно переименовать в "ensemble"


class ModelEnum(str, Enum):
    medium = "medium"  # AIFS
    s2s = "s2s"  # S2S


class ForecastQuery(BaseModel):
    city: Optional[str] = Field(default=None, description="City name; ignored if lat/lon provided")
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lon: Optional[float] = Field(default=None, ge=-180, le=180)
    days: conint(ge=1, le=45) = 1
    params: ParamsEnum = ParamsEnum.simple
    mode: ModeEnum = ModeEnum.base
    model: ModelEnum = ModelEnum.medium


class Location(BaseModel):
    lat: float
    lon: float


class LeadtimeData(BaseModel):
    time: datetime
    params: Dict[str, Any]


class ForecastResponse(BaseModel):
    location: Location
    start_time: datetime
    data: list[LeadtimeData]


class ForecastPointResponse(BaseModel):
    lat: float
    lon: float
    start_time: datetime
    data_url: str
