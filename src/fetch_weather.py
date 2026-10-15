"""气象数据采集：Open-Meteo 预报 + 历史再分析（ERA5 / ERA5-Land 驱动）。

设计原则
--------
1. 只依赖免密钥的公开 API，任何评审者都能复现；
2. 原始 JSON 一律落盘缓存，重复运行不重复请求；
3. 区域气象场在本站尺度（约 3 km）上分辨率不足——这正是需要微气候降尺度的原因，
   采集模块只负责把"区域强迫场"拿回来。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config as C


def _cache(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _get(url: str, params: dict, cache_path: Path, retries: int = 3) -> dict:
    cached = _cache(cache_path)
    if cached is not None:
        return cached
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            if "error" in data and data.get("error"):
                raise RuntimeError(f"API error: {data.get('reason')}")
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 + 3 * i)
    raise RuntimeError(f"请求失败 {url} params={params}: {last_err}")


def _to_frame(payload: dict) -> pd.DataFrame:
    h = payload.get("hourly", {})
    if not h:
        raise RuntimeError("响应中没有 hourly 数据")
    df = pd.DataFrame({k: v for k, v in h.items()})
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df


# --------------------------------------------------------------------------
# 未来预报（驱动 Demo 的"实时模式"）
# --------------------------------------------------------------------------
def fetch_forecast(days: int = 6, refresh: bool = False, past_days: int = 3
                   ) -> tuple[pd.DataFrame, dict]:
    cache = C.RAW / f"forecast_{days}d_{past_days}p.json"
    if refresh and cache.exists():
        cache.unlink()
    payload = _get(
        C.OPENMETEO_FORECAST,
        {
            "latitude": C.SITE["lat"], "longitude": C.SITE["lon"],
            "hourly": ",".join(C.FORECAST_HOURLY),
            "forecast_days": days, "past_days": past_days,
            "timezone": C.TIMEZONE, "wind_speed_unit": "ms",
        },
        cache,
    )
    df = _to_frame(payload)
    meta = {
        "source": "Open-Meteo Forecast API",
        "model_grid_lat": payload.get("latitude"),
        "model_grid_lon": payload.get("longitude"),
        "model_elevation_m": payload.get("elevation"),
        "timezone": payload.get("timezone"),
        "retrieved_fields": C.FORECAST_HOURLY,
    }
    return df, meta


# --------------------------------------------------------------------------
# 历史再分析（驱动"回溯验证"与"情景回放"）
# --------------------------------------------------------------------------
def fetch_archive(start: str, end: str, tag: str, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    cache = C.RAW / f"archive_{tag}.json"
    if refresh and cache.exists():
        cache.unlink()
    payload = _get(
        C.OPENMETEO_ARCHIVE,
        {
            "latitude": C.SITE["lat"], "longitude": C.SITE["lon"],
            "start_date": start, "end_date": end,
            "hourly": ",".join(C.ARCHIVE_HOURLY),
            "timezone": C.TIMEZONE, "wind_speed_unit": "ms",
        },
        cache,
    )
    df = _to_frame(payload)
    meta = {
        "source": "Open-Meteo Archive API (ERA5 / ERA5-Land)",
        "period": f"{start} ~ {end}",
        "model_grid_lat": payload.get("latitude"),
        "model_grid_lon": payload.get("longitude"),
        "model_elevation_m": payload.get("elevation"),
        "retrieved_fields": C.ARCHIVE_HOURLY,
    }
    return df, meta


def fetch_backtest_years(refresh: bool = False) -> dict:
    """逐年份拉取夏季逐小时序列，用于历史灾情复盘。"""
    out = {}
    for y in C.BACKTEST_YEARS:
        start = f"{y}-{C.BACKTEST_START}"
        end = f"{y}-{C.BACKTEST_END}"
        try:
            df, meta = fetch_archive(start, end, f"summer_{y}", refresh=refresh)
            out[y] = {"df": df, "meta": meta}
            print(f"[weather] 历史 {y} 夏季: {len(df)} 小时记录")
        except Exception as e:  # noqa: BLE001
            print(f"[weather] 历史 {y} 拉取失败: {e}")
    return out


def regional_summary(df: pd.DataFrame) -> dict:
    """区域强迫场摘要（用于报告与"为什么必须降尺度"的论证）。"""
    d = df.copy()
    daily = d.resample("D").agg(
        tmax=("temperature_2m", "max"),
        tmin=("temperature_2m", "min"),
        tmean=("temperature_2m", "mean"),
        precip=("precipitation", "sum"),
        et0=("et0_fao_evapotranspiration", "sum"),
        rh_mean=("relative_humidity_2m", "mean"),
        sw_sum=("shortwave_radiation", "sum"),
    ).dropna(how="all")
    return {
        "hours": int(len(d)),
        "start": str(d.index.min()),
        "end": str(d.index.max()),
        "tmax_max": float(np.nanmax(d["temperature_2m"])) if len(d) else None,
        "tmin_min": float(np.nanmin(d["temperature_2m"])) if len(d) else None,
        "daily": daily,
    }


if __name__ == "__main__":
    df, meta = fetch_forecast()
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(df.head())
