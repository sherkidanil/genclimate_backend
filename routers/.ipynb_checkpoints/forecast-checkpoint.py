from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path
import re
import numpy as np
import xarray as xr
from fastapi import APIRouter, HTTPException, Query

from config import settings
from core.async_requests import aio_requests
from core.schemas import ForecastResponse, LeadtimeData, ModeEnum, ParamsEnum

router = APIRouter(tags=["Forecasts"])
logger = logging.getLogger(__name__)

request = aio_requests()

PARAM_GROUPS: Dict[ParamsEnum, List[str]] = {
    ParamsEnum.simple: ["2t", "10u", "10v", "msl"],
    ParamsEnum.surface: ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"],
    ParamsEnum.full: [],  # все переменные
}

# ------------------ filesystem loader (predictions/YYYYMMDD/HH) ------------------

def load_predictions_from_tree(base_dir: str | Path) -> xr.Dataset:
    """Открывает все .../YYYYMMDD/HH/prediction_format.zarr и склеивает по forecast_time."""
    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {base_path}")

    zarr_paths: list[Path] = []
    times: list[np.datetime64] = []
    pattern = re.compile(r"(\d{8})[/\\](\d{2})[/\\]prediction_format\.zarr$")

    for zarr_file in base_path.glob("*/*/prediction_format.zarr"):
        m = pattern.search(str(zarr_file))
        if not m:
            continue
        date_str, hour_str = m.groups()
        ts = np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T{hour_str}:00")
        zarr_paths.append(zarr_file)
        times.append(ts)

    if not zarr_paths:
        raise HTTPException(status_code=404, detail=f"No prediction_format.zarr found under {base_path}")

    # сортируем по времени запуска
    times, zarr_paths = zip(*sorted(zip(times, zarr_paths)))
    datasets: list[xr.Dataset] = []
    for ts, zp in zip(times, zarr_paths):
        ds = xr.open_zarr(zp, consolidated=True)
        ds = ds.expand_dims({"forecast_time": [ts]})
        datasets.append(ds)
        
    combined = xr.concat(datasets, dim="forecast_time", combine_attrs="override")
    combined.attrs.setdefault("model_id", "aifs")
    combined.attrs.setdefault("initial_time", str(times[0]))
    return combined

# ------------------ helpers ------------------

async def _geocode(city: str) -> Dict[str, float]:
    params = {"q": city, "format": "json", "limit": 1}
    headers = {"User-Agent": settings.geocoder_user_agent}
    payload = await request.request_json(settings.geocoder_endpoint, params=params, headers=headers)
    if not payload:
        raise HTTPException(status_code=404, detail="City not found")
    return {"lat": float(payload[0]["lat"]), "lon": float(payload[0]["lon"])}

def _normalize_lon_to_ds(lon_in: float, ds_lon_values: np.ndarray) -> float:
    lon = float(lon_in)
    lon_min, lon_max = float(np.min(ds_lon_values)), float(np.max(ds_lon_values))
    if lon_min >= 0.0 and lon_max <= 360.0:  # [0, 360)
        if lon < 0:
            lon += 360.0
    elif lon_min >= -180.0 and lon_max <= 180.0:  # [-180, 180]
        if lon > 180.0:
            lon -= 360.0
    return lon

def _list_requested_vars(ds: xr.Dataset, params: ParamsEnum) -> list[str]:
    if params == ParamsEnum.full or not PARAM_GROUPS[params]:
        return list(ds.data_vars)
    available = set(ds.data_vars)
    return [v for v in PARAM_GROUPS[params] if v in available]

def _select_point(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    lon_adj = _normalize_lon_to_ds(lon, ds["lon"].values)
    return ds.sel(lat=lat, lon=lon_adj, method="nearest")

def _get_max_days(ds: xr.Dataset) -> int:
    if "forecast_time" in ds.dims:
        return int(ds.dims["forecast_time"])
    mid = (ds.attrs.get("model_id") or "").lower()
    return 15 if "aifs" in mid else 45 if ("s2s" in mid or "fuxi" in mid) else 1

def _get_initial_time(ds: xr.Dataset) -> datetime:
    init = ds.attrs.get("initial_time")
    if isinstance(init, str):
        try:
            return datetime.fromisoformat(init.replace("Z", "+00:00"))
        except Exception:
            pass
    return datetime.utcnow()

def _reduce_variable(da: xr.DataArray, mode: ModeEnum):
    if mode == ModeEnum.base:
        if "member" in da.dims:
            da = da.mean(dim="member", keep_attrs=True, skipna=True)
        return float(np.asarray(da.values))
    else:
        if "member" in da.dims:
            return np.asarray(da.values).ravel().astype(float).tolist()
        return [float(np.asarray(da.values))]

def _extract_timeseries(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    vars_: list[str],
    days: int,
    mode: ModeEnum,
    model_id: str = Query("medium", enum=["medium", "s2s"])
) -> list[LeadtimeData]:
    max_days = _get_max_days(ds)
    days = int(max(1, min(days, max_days)))

    point = _select_point(ds, lat, lon)
    out: list[LeadtimeData] = []

    if "forecast_time" in point.dims:
        ft = np.asarray(point["forecast_time"].values)

        if model_id == 'medium':
            n = min(days, ft.shape[0])
        else:
            n = min(days*4, ft.shape[0])
        for i in range(n):
            step = point.isel(forecast_time=i)

            t_py = np.datetime_as_string(ft[i], unit="s")
            t_dt = datetime.fromisoformat(t_py.replace("Z", "+00:00")) if "Z" in t_py else datetime.fromisoformat(t_py)

            row: Dict[str, Any] = {}
            for v in vars_:
                if v in point.data_vars:
                    row[v.upper()] = _reduce_variable(step[v], mode)

            out.append({"time": t_dt, "params": row})
    else:

        t_dt = _get_initial_time(ds)
        row = {v.upper(): _reduce_variable(point[v], mode) for v in vars_ if v in point.data_vars}
        out.append({"time": t_dt, "params": row})

    return out

# ------------------ API ------------------

@router.get("/point_forecast", response_model=ForecastResponse)
async def point_forecast(
    city: str | None = Query(None, description="City name (ignored if lat/lon provided)"),
    lat: float | None = Query(None, ge=-90, le=90),
    lon: float | None = Query(None, ge=-180, le=180),
    days: int = Query(1, ge=1, le=45),
    params: ParamsEnum = Query(ParamsEnum.simple),
    mode: ModeEnum = Query(ModeEnum.base),
    model: str = Query("medium", enum=["medium", "s2s"]),
):
    if not ((lat is not None and lon is not None) or city):
        raise HTTPException(status_code=400, detail="Provide either lat/lon or city")

    if city and (lat is None or lon is None):
        coords = await _geocode(city)
        lat, lon = coords["lat"], coords["lon"]

    ds = load_predictions_from_tree(settings.forecast_dir_aifs)

    vars_requested = _list_requested_vars(ds, params)
    if not vars_requested:
        raise HTTPException(status_code=400, detail="No variables matched requested params")

    series = _extract_timeseries(ds, float(lat), float(lon), vars_requested, days, mode)
    start_dt = _get_initial_time(ds)

    return ForecastResponse(location={"lat": float(lat), "lon": float(lon)}, start_time=start_dt, data=series, model_id=model)
