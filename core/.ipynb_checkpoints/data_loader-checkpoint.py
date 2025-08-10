from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple
import re
import numpy as np
import xarray as xr
from cachetools import TTLCache, cached

from config import settings

logger = logging.getLogger(__name__)

_dataset_cache: TTLCache = TTLCache(maxsize=3, ttl=settings.cache_ttl)


def _collect_prediction_format_zarrs(root: Path) -> Tuple[List[Path], List[np.datetime64]]:
    """
    Ищет все prediction_format.zarr в дереве /YYYYMMDD/HH/,
    возвращает списки путей и соответствующих времён запуска.
    """
    pattern = re.compile(r"(\d{8})/(\d{2})")
    paths: List[Path] = []
    times: List[np.datetime64] = []

    for zarr_path in root.glob("*/*/prediction_format.zarr"):
        m = pattern.search(str(zarr_path))
        if not m:
            continue
        date_str, hour_str = m.groups()
        ts = np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T{hour_str}:00")
        paths.append(zarr_path)
        times.append(ts)

    if not paths:
        raise FileNotFoundError(f"No prediction_format.zarr found under {root}")

    times, paths = zip(*sorted(zip(times, paths)))
    return list(paths), list(times)


@cached(_dataset_cache)
def load_latest_dataset(model_name: str) -> xr.Dataset:
    """
    Загружает все prediction_format.zarr для указанной модели,
    склеивает их по измерению forecast_time и кэширует в памяти.
    """
    if model_name != "medium":
        raise ValueError(f"Only 'medium' model supported with prediction_format.zarr, got '{model_name}'.")

    root = settings.forecast_dir_aifs
    paths, times = _collect_prediction_format_zarrs(root)

    datasets = []
    for ts, p in zip(times, paths):
        logger.info("Opening %s", p)
        ds = xr.open_zarr(p, consolidated=True)
        ds = ds.expand_dims({"forecast_time": [ts]})
        datasets.append(ds)

    logger.info("Concatenating %d datasets along forecast_time", len(datasets))
    combined = xr.concat(datasets, dim="forecast_time")

    combined.attrs["model_id"] = "aifs"
    combined.attrs["initial_time"] = str(times[0])

    return combined
