from __future__ import annotations

import logging, math, os, time, threading, re, shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
import plotly.graph_objects as go
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from config import settings
from core.async_requests import aio_requests
from core.schemas import ForecastResponse, LeadtimeData, ModeEnum, ParamsEnum

router = APIRouter(tags=["Forecasts"])
logger = logging.getLogger(__name__)

request = aio_requests()

DOWNLOAD_ROOT = Path(getattr(settings, "download_dir", "/tmp/genclimate_downloads")).expanduser().resolve()
DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

PARAM_GROUPS: Dict[ParamsEnum, List[str]] = {
    ParamsEnum.simple: ["2t", "10u", "10v", "msl"],
    ParamsEnum.surface: ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"],
    ParamsEnum.full: [],  # все переменные
}

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

# --- кеш поверх loader --------------------------------------------------------
DEFAULT_RESCAN_SEC = 60

@dataclass
class _CacheEntry:
    ds: xr.Dataset
    signature: Tuple
    last_scan: float

_DS_CACHE: dict[Path, _CacheEntry] = {}
_DS_CACHE_LOCK = threading.Lock()

def _dir_signature(base_dir: Path) -> Tuple:
    items = []
    pattern = re.compile(r"(\d{8})[/\\](\d{2})[/\\]prediction_format\.zarr$")
    for z in base_dir.glob("*/*/prediction_format.zarr"):
        if not pattern.search(str(z)):
            continue
        meta = z / ".zmetadata"
        try:
            st = meta.stat() if meta.exists() else z.stat()
            items.append((str(z), st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            continue
    items.sort()
    return tuple(items)

def load_predictions_from_tree_cached(base_dir: str | Path, *, rescan_sec: int = DEFAULT_RESCAN_SEC) -> xr.Dataset:
    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {base_path}")

    now = time.time()
    with _DS_CACHE_LOCK:
        entry = _DS_CACHE.get(base_path)
        if entry and (now - entry.last_scan) < rescan_sec:
            return entry.ds

    signature = _dir_signature(base_path)
    with _DS_CACHE_LOCK:
        entry = _DS_CACHE.get(base_path)
        if entry and entry.signature == signature:
            entry.last_scan = now
            return entry.ds

    ds = load_predictions_from_tree(base_path)
    with _DS_CACHE_LOCK:
        _DS_CACHE[base_path] = _CacheEntry(ds=ds, signature=signature, last_scan=now)
    return ds

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
            n = min(days * 4, ft.shape[0])
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

def crop_forecast(
    ds: xr.Dataset,
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    days: int,
    *,
    params_group: Optional[ParamsEnum] = None,
    param_groups: Optional[Dict[ParamsEnum, List[str]]] = None,
    require_present: bool = False,
) -> xr.Dataset:
    """Кроп по времени/боксу. Долготы остаются как в ds (обычно 0..360)."""
    if days <= 0:
        raise ValueError("`days` должно быть > 0")
    if "forecast_time" not in ds.coords:
        raise ValueError("нет координаты 'forecast_time'")

    # время
    t0 = ds["forecast_time"].isel(forecast_time=0).values
    cutoff = np.datetime64(t0) + np.timedelta64(int(days), "D")
    ds_t = ds.where(ds["forecast_time"] < cutoff, drop=True)

    # широта (в любом порядке в исходнике)
    lat_min, lat_max = sorted([lat1, lat2])
    lat_asc = float(ds["lat"].isel(lat=1)) > float(ds["lat"].isel(lat=0))
    lat_slice = slice(lat_min, lat_max) if lat_asc else slice(lat_max, lat_min)

    # привести входные долготы в домен датасета (без смены самого домена)
    lon_min_ds = float(ds["lon"].min())
    lon_max_ds = float(ds["lon"].max())
    ds_is_0360 = lon_max_ds > 180.0  # типично для ERA5 0..360

    def to_ds_domain(lon):
        lon = float(lon)
        if ds_is_0360:     # 0..360
            return lon % 360.0
        # [-180,180]
        x = ((lon + 180.0) % 360.0) - 180.0
        if x == -180.0 and lon > 180.0:
            x = 180.0
        return x

    L1, L2 = to_ds_domain(lon1), to_ds_domain(lon2)

    # кроп по долготе; если пересекаем 0°, режем на два куска и конкатим
    if not (ds_is_0360 and L1 > L2):
        ds_space = ds_t.sel(lat=lat_slice, lon=slice(L1, L2))
    else:
        p1 = ds_t.sel(lat=lat_slice, lon=slice(L1, lon_max_ds))
        p2 = ds_t.sel(lat=lat_slice, lon=slice(lon_min_ds, L2))
        ds_space = xr.concat([p1, p2], dim="lon")

    # (опц.) фильтр переменных
    if params_group is not None and param_groups is not None:
        wanted = param_groups.get(params_group, [])
        if wanted:  # пустой список → все
            def match(n): return any(n == w or n.startswith(w + "_") for w in wanted)
            keep = [k for k in ds_space.data_vars if match(k)]
            if not keep and require_present:
                raise ValueError(f"Не найдено ни одной переменной из группы {params_group}")
            if keep:
                ds_space = ds_space[keep]

    return ds_space

def plot_forecast_map(
    ds: xr.Dataset,
    *,
    var_names: Optional[Sequence[str]] = None,
    dims: Tuple[str, str, str] = ("forecast_time", "lat", "lon"),
    quantiles: Tuple[float, float] = (0.02, 0.98),
    title: str = "Forecast",
    decimate: int = 1,
) -> go.Figure:
    """Интерактивная карта: dropdown по переменным + slider по времени.
    Долготы оставляем как в ds (0..360). Если кроп пересёк 0°, визуально
    «разворачиваем» ось (часть 0..L2 сдвигаем на +360) — без изменения ds.
    """
    tdim, ydim, xdim = dims
    if var_names is None:
        req = set(dims)
        var_names = [
            k for k, da in ds.data_vars.items()
            if set(da.dims) == req and all(d in da.dims for d in dims)
        ]
    if not var_names:
        raise ValueError("Нет переменных с dims == (forecast_time, lat, lon)")

    # широта по возрастанию (для оси Y)
    ds_disp = ds.sortby(ydim)
    if decimate > 1:
        ds_disp = ds_disp.isel(**{ydim: slice(0, None, decimate),
                                  xdim: slice(0, None, decimate)})

    lats = ds_disp[ydim].values
    lons = ds_disp[xdim].values
    times = ds_disp[tdim].values
    nT = len(times)

    diffs = np.diff(lons)
    cuts = np.where(diffs < 0)[0]
    if cuts.size:
        cut = int(cuts[0] + 1) 
        lons_plot = lons.copy()
        lons_plot[cut:] = lons_plot[cut:] + 360.0
        x_lo, x_hi = float(lons_plot.min()), float(lons_plot.max())
        step = 30.0
        start = math.ceil(x_lo / step) * step
        tickvals = np.arange(start, x_hi + 1e-6, step)
        ticktext = [f"{(v % 360):.0f}" for v in tickvals]
    else:
        lons_plot = lons
        x_lo, x_hi = float(lons.min()), float(lons.max())
        tickvals, ticktext = None, None

    y_lo, y_hi = float(lats.min()), float(lats.max())

    # единая шкала per variable
    var_scales = {}
    for v in var_names:
        q = ds_disp[v].quantile(quantiles).compute().values
        vmin, vmax = float(q[0]), float(q[1])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(ds_disp[v].min().compute())
            vmax = float(ds_disp[v].max().compute())
        var_scales[v] = (vmin, vmax)

    # трейсы (var,time)
    data_traces, idx = [], {}
    for vi, v in enumerate(var_names):
        vmin, vmax = var_scales[v]
        for ti in range(nT):
            z = np.asarray(ds_disp[v].isel({tdim: ti}).values, dtype=np.float32)
            tr = go.Heatmap(
                z=z, x=lons_plot, y=lats,
                zmin=vmin, zmax=vmax,
                colorbar=dict(title=v, len=0.8),
                visible=False,
                hovertemplate=f"lat=%{{y:.2f}}°, lon=%{{x:.2f}}°<br>{v}=%{{z}}<extra></extra>",
            )
            idx[(vi, ti)] = len(data_traces)
            data_traces.append(tr)
    if data_traces:
        data_traces[idx[(0, 0)]].visible = True

    def slider_steps(active_vi=0):
        steps = []
        for ti in range(nT):
            vis = [False] * len(data_traces)
            vis[idx[(active_vi, ti)]] = True
            steps.append(dict(
                method="update",
                args=[{"visible": vis},
                      {"title": f"{title} — {var_names[active_vi]} — "
                                f"{np.datetime_as_string(times[ti], unit='h')}"}],
                label=np.datetime_as_string(times[ti], unit="h")
            ))
        return steps

    sliders = [dict(active=0, currentvalue={"prefix": "Time: "}, pad={"t": 40}, steps=slider_steps(0))]
    updatemenus = [dict(
        buttons=[dict(
            label=v, method="update",
            args=[{"visible": [(i == idx[(vi, 0)]) for i in range(len(data_traces))]},
                  {"title": f"{title} — {v} — {np.datetime_as_string(times[0], unit='h')}",
                   "sliders": [dict(active=0, currentvalue={"prefix": "Time: "}, pad={"t": 40},
                                    steps=slider_steps(vi))]}]
        ) for vi, v in enumerate(var_names)],
        direction="down", showactive=True, x=0.01, y=1.12, xanchor="left", yanchor="top",
    )]

    fig = go.Figure(data=data_traces)
    fig.update_layout(
        title=f"{title} — {var_names[0]} — {np.datetime_as_string(times[0], unit='h')}",
        updatemenus=updatemenus, sliders=sliders,
        xaxis_title="Longitude (0..360)",
        yaxis_title="Latitude",
        margin=dict(l=60, r=20, t=90, b=40),
    )
    fig.update_xaxes(range=[x_lo, x_hi])

    # подгоняем размер фигуры под аспект кропа
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo
    ratio = (x_span / y_span) if y_span != 0 else 1.0

    base_h = 700
    fig.update_layout(autosize=False, height=base_h, width=int(base_h * ratio))

    if tickvals is not None:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    fig.update_yaxes(range=[y_lo, y_hi], scaleanchor="x", scaleratio=1)
    return fig


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

def _sanitize_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    import numpy as _np
    ds = ds.copy(deep=False)
    def _to_jsonable(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (_np.integer, _np.floating, _np.bool_)):
            return v.item()
        if isinstance(v, (_np.datetime64, _np.timedelta64)):
            try:
                return _np.datetime_as_string(v, unit="s")
            except Exception:
                return str(v)
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8")
            except Exception:
                return str(v)
        if isinstance(v, (list, tuple)):
            return [_to_jsonable(x) for x in v]
        if isinstance(v, _np.ndarray):
            return [_to_jsonable(x) for x in v.tolist()]
        return str(v)
    ds.attrs = {k: _to_jsonable(v) for k, v in dict(ds.attrs).items()}
    for name, var in ds.variables.items():
        if var.attrs:
            var.attrs = {k: _to_jsonable(v) for k, v in dict(var.attrs).items()}
    if "forecast_time" in ds.coords and np.issubdtype(ds["forecast_time"].dtype, np.datetime64):
        ds["forecast_time"] = ds["forecast_time"].astype("datetime64[ns]")
    return ds

def _save_subset_zarr_zip(ds_sub: xr.Dataset) -> str:
    """Сохраняет вырезанный регион как Zarr (consolidated) и упаковывает в ZIP.
    Возвращает имя ZIP-файла."""
    ds_clean = _sanitize_for_zarr(ds_sub)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    pid = os.getpid()
    store_dir = DOWNLOAD_ROOT / f"region_{ts}_{pid}.zarr"
    ds_clean.to_zarr(store_dir, mode="w", consolidated=True)
    zip_path = shutil.make_archive(str(store_dir), "zip", root_dir=store_dir)
    shutil.rmtree(store_dir, ignore_errors=True)
    return Path(zip_path).name  # например, region_20250810T123456_1234.zarr.zip

@router.get("/region_file/{file_name}")
def region_file(file_name: str):
    fpath = (DOWNLOAD_ROOT / file_name).resolve()
    if not fpath.exists() or fpath.parent != DOWNLOAD_ROOT:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(fpath), media_type="application/zip", filename=fpath.name)

@router.get("/region_forecast")
async def region_forecast(
    lat1: float = Query(..., ge=-90, le=90, description="Северо-западный угол: широта"),
    lon1: float = Query(..., ge=-180, le=360, description="Северо-западный угол: долгота (0..360 или -180..180)"),
    lat2: float = Query(..., ge=-90, le=90, description="Юго-восточный угол: широта"),
    lon2: float = Query(..., ge=-180, le=360, description="Юго-восточный угол: долгота (0..360 или -180..180)"),
    days: int = Query(3, ge=1, le=45, description="Сколько суток от первого forecast_time"),
    params: ParamsEnum = Query(ParamsEnum.simple),
    model: str = Query("medium", enum=["medium", "s2s"]),
):
    if model != "medium":
        raise HTTPException(status_code=400, detail="Only model=medium is supported for region_forecast now.")

    ds = load_predictions_from_tree_cached(settings.forecast_dir_aifs, rescan_sec=60)

    try:
        subset = crop_forecast(
            ds,
            lat1=lat1, lon1=lon1,
            lat2=lat2, lon2=lon2,
            days=days,
            params_group=params,
            param_groups=PARAM_GROUPS,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Crop failed: {e}")

    if subset.dims.get("lat", 0) == 0 or subset.dims.get("lon", 0) == 0:
        raise HTTPException(status_code=400, detail="Empty selection for given box.")

    # 3) краткая сводка
    ft = subset.coords.get("forecast_time")
    t_start = str(np.datetime_as_string(ft.values[0], unit="h")) if ft is not None and ft.size else None
    t_end = str(np.datetime_as_string(ft.values[-1], unit="h")) if ft is not None and ft.size else None
    lat_range = [float(subset["lat"].min()), float(subset["lat"].max())]
    lon_range = [float(subset["lon"].min()), float(subset["lon"].max())]
    summary = {
        "grid": {
            "lat_size": int(subset.dims["lat"]),
            "lon_size": int(subset.dims["lon"]),
            "lat_range": lat_range,
            "lon_range": lon_range,
        },
        "time": {
            "n_steps": int(subset.dims.get("forecast_time", 1)),
            "start": t_start,
            "end": t_end,
            "initial_time": str(_get_initial_time(ds)),
        },
        "variables": list(subset.data_vars),
        "model_id": ds.attrs.get("model_id", "aifs"),
    }

    try:
        file_name = _save_subset_zarr_zip(subset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write Zarr ZIP: {e}")

    download_url = f"/region_file/{file_name}"

    try:
        fig = plot_forecast_map(subset, title="Region forecast")
        preview_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception as e:
        logger.exception("Plotly preview failed")
        preview_html = f"<div>Preview error: {e}</div>"

    return {
        "summary": summary,
        "download_url": download_url,          # .zarr.zip
        "preview_html": preview_html,
        "bbox_request": {
            "lat1": lat1, "lon1": lon1,
            "lat2": lat2, "lon2": lon2,
            "days": days,
            "params": params.value if hasattr(params, "value") else str(params),
        },
    }
