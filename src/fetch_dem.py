"""DEM 采集：AWS Open Data / Mapzen Terrarium 地形瓦片（免密钥，公开数据集）。

- 数据源：https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
  覆盖 SRTM / GMTED / 国家高程数据融合产物，纬度 30° 处 z=15 约 3.5–4 m/像素。
- 高程解码：elev = (R * 256 + G + B / 256) - 32768（米）
- 输出：研究区规则网格 10 m DEM（numpy .npz，含经纬度坐标轴）

说明：Terrarium 为纯高程数据（不含任何行政边界/地图注记），
用作微地形分析不涉及地图服务合规问题。
"""
from __future__ import annotations

import io
import math

import numpy as np
import requests
from PIL import Image

import config as C


def deg2tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def tile2lon(x: float, z: int) -> float:
    return x / (2 ** z) * 360.0 - 180.0


def tile2lat(y: float, z: int) -> float:
    n = math.pi - 2.0 * math.pi * y / (2 ** z)
    return math.degrees(math.atan(math.sinh(n)))


def _fetch_tile(z: int, x: int, y: int, cache_dir=C.RAW / "terrarium") -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"{z}_{x}_{y}.png"
    if not fp.exists():
        url = C.TERRARIUM_URL.format(z=z, x=x, y=y)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        fp.write_bytes(r.content)
    img = Image.open(io.BytesIO(fp.read_bytes())).convert("RGB")
    arr = np.asarray(img).astype(np.float64)
    z_val = arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0 - 32768.0
    # 瓦片边缘的无数据像元解码后会得到极端负值（如 -32768 或 -1362），置为 NaN
    z_val = np.where(z_val < -100.0, np.nan, z_val)
    return z_val


def _fill_nodata(a: np.ndarray) -> np.ndarray:
    """用最近邻有效像元填补无数据空洞。"""
    if not np.isnan(a).any():
        return a
    try:
        from scipy import ndimage
        nan_mask = np.isnan(a)
        idx = ndimage.distance_transform_edt(nan_mask, return_distances=False,
                                             return_indices=True)
        filled = a[tuple(idx)]
        return np.where(nan_mask, filled, a)
    except Exception:                                    # noqa: BLE001
        med = np.nanmedian(a)
        return np.where(np.isnan(a), med if np.isfinite(med) else 0.0, a)


def load_dem(refresh: bool = False) -> dict:
    """拼接瓦片 → 重采样到 10 m 规则网格。"""
    out_fp = C.PROC / "dem_10m.npz"
    if out_fp.exists() and not refresh:
        z = np.load(out_fp)
        return {k: z[k] for k in z.files}

    b = C.BBOX
    z = C.DEM_Z
    x0, y0 = deg2tile(b["lat_max"], b["lon_min"], z)     # 左上（北西）
    x1, y1 = deg2tile(b["lat_min"], b["lon_max"], z)     # 右下（南东）

    # 按瓦片行拼接：从最北的瓦片行开始（瓦片内行序本身即为北→南）
    rows = []
    for yy in range(y0, y1 + 1):
        cols = [_fetch_tile(z, xx, yy) for xx in range(x0, x1 + 1)]
        rows.append(np.hstack(cols))
    mosaic = np.vstack(rows)                    # 行 0 = 最北
    print(f"[dem] 瓦片拼接完成: {mosaic.shape}  (z={z}, {x1-x0+1}×{y1-y0+1} tiles)")

    n_pix_y, n_pix_x = mosaic.shape
    # 每个像元中心对应的绝对瓦片坐标（瓦片号 + 瓦片内偏移），再转经纬度
    lon_c = np.array([tile2lon(x0 + (i // 256) + ((i % 256) + 0.5) / 256.0, z)
                      for i in range(n_pix_x)])
    lat_c = np.array([tile2lat(y0 + (j // 256) + ((j % 256) + 0.5) / 256.0, z)
                      for j in range(n_pix_y)])
    # 统一约定：DEM 行 0 = 最南（纬度随行号递增），与坡度/坡向的差分方向保持一致
    mosaic = mosaic[::-1]
    lat_c = lat_c[::-1]
    assert np.all(np.diff(lat_c) > 0) and np.all(np.diff(lon_c) > 0), "经纬度轴必须单调递增"

    # 目标 10 m 规则网格（纬度/经度均递增，行 0 = 最南）
    res = C.GRID_RES_M
    lat_step = res / 111_320.0
    lon_step = res / (111_320.0 * math.cos(math.radians((b["lat_min"] + b["lat_max"]) / 2)))
    lat_new = np.arange(b["lat_min"], b["lat_max"], lat_step)
    lon_new = np.arange(b["lon_min"], b["lon_max"], lon_step)

    idx_lon = np.clip(np.searchsorted(lon_c, lon_new), 1, len(lon_c) - 1)
    wx = (lon_new - lon_c[idx_lon - 1]) / (lon_c[idx_lon] - lon_c[idx_lon - 1])
    with np.errstate(invalid="ignore"):
        tmp = (mosaic[:, idx_lon - 1] * (1 - wx) + mosaic[:, idx_lon] * wx)
    idx_lat = np.clip(np.searchsorted(lat_c, lat_new), 1, len(lat_c) - 1)
    wy = (lat_new - lat_c[idx_lat - 1]) / (lat_c[idx_lat] - lat_c[idx_lat - 1])
    with np.errstate(invalid="ignore"):
        dem = (tmp[idx_lat - 1, :] * (1 - wy)[:, None] + tmp[idx_lat, :] * wy[:, None])
    dem = _fill_nodata(dem)

    out = {
        "dem": dem.astype(np.float32),
        "lat": lat_new.astype(np.float64),
        "lon": lon_new.astype(np.float64),
        "res_m": np.array([res]),
        "dem_source": np.array([f"Mapzen/AWS Terrarium z{z} (~3.5–4 m/px) → 重采样 10 m"]),
    }
    np.savez_compressed(out_fp, **out)
    print(f"[dem] 10 m DEM: {dem.shape}  高程范围 {np.nanmin(dem):.0f}–{np.nanmax(dem):.0f} m")
    return out


if __name__ == "__main__":
    d = load_dem(refresh=True)
    print(d["dem"].shape, d["dem"].mean())
