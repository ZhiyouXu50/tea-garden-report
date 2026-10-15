"""地形因子：坡度、坡向、地形位置指数、汇流累积、地形湿度指数、
地平线遮蔽与天空可视因子（10 m 网格）。"""
from __future__ import annotations

import numpy as np

from . import solar


def slope_aspect(dem: np.ndarray, res_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Horn(1981) 三阶差分。返回 (坡度°, 坡向°)。坡向正北 0°，顺时针。"""
    z = dem.astype(np.float64)
    zp = np.pad(z, 1, mode="edge")
    a = zp[:-2, :-2]; b = zp[:-2, 1:-1]; c = zp[:-2, 2:]
    d = zp[1:-1, :-2]; f = zp[1:-1, 2:]
    g = zp[2:, :-2];  h = zp[2:, 1:-1]; i = zp[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * res_m)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * res_m)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # 图像行向下 = 向南（y 增加为南），dzdy>0 表示向南倾斜
    aspect = np.degrees(np.arctan2(-dzdx, -dzdy)) % 360.0
    return slope.astype(np.float32), aspect.astype(np.float32)


def tpi(dem: np.ndarray, res_m: float, radius_m: float = 150.0) -> np.ndarray:
    """地形位置指数：单元高程 − 邻域均值。负值 = 谷地/洼地，正值 = 山脊/凸地。"""
    r = max(1, int(round(radius_m / res_m)))
    z = dem.astype(np.float64)
    zp = np.pad(z, r, mode="edge")
    cs = np.cumsum(np.cumsum(zp, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    ny, nx = z.shape
    ys = np.arange(ny); xs = np.arange(nx)
    # 边长 (2r+1) 的方形窗口：zp 中行范围为 [i, i+2r]，对应积分图索引 [i, i+2r+1)
    y0 = ys; y1 = ys + 2 * r + 1
    x0 = xs; x1 = xs + 2 * r + 1
    s = (cs[np.ix_(y1, x1)] - cs[np.ix_(y0, x1)]
         - cs[np.ix_(y1, x0)] + cs[np.ix_(y0, x0)])
    n = float((2 * r + 1) ** 2)
    return (z - s / n).astype(np.float32)


def flow_accumulation(dem: np.ndarray, res_m: float) -> np.ndarray:
    """D8 单流向汇流累积面积（m²）。按高程降序逐单元累加，O(n log n)。"""
    z = dem.astype(np.float64)
    ny, nx = z.shape
    zp = np.pad(z, 1, mode="edge")
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    best_drop = np.full((ny, nx), -np.inf)
    best = np.full((ny, nx), -1, dtype=np.int64)
    for k, (di, dj) in enumerate(dirs):
        nb = zp[1 + di: 1 + di + ny, 1 + dj: 1 + dj + nx]
        dist = res_m * (np.sqrt(2.0) if (di and dj) else 1.0)
        drop = (z - nb) / dist
        upd = drop > best_drop
        best_drop = np.where(upd, drop, best_drop)
        best = np.where(upd, k, best)
    best = np.where(best_drop <= 0, -1, best)

    flat = z.ravel()
    order = np.argsort(-flat)
    acc = np.full(ny * nx, res_m * res_m)
    bflat = best.ravel()
    for idx in order:
        t = bflat[idx]
        if t < 0:
            continue
        di, dj = dirs[t]
        yi, xi = divmod(idx, nx)
        ty, tx = yi + di, xi + dj
        if 0 <= ty < ny and 0 <= tx < nx:
            acc[ty * nx + tx] += acc[idx]
    return acc.reshape(ny, nx).astype(np.float32)


def twi(acc: np.ndarray, slope_deg: np.ndarray) -> np.ndarray:
    """地形湿度指数 TWI = ln(a / tanβ)。"""
    a = acc / 10.0                                    # 单位等高线宽度取 10 m
    tanb = np.maximum(np.tan(np.radians(slope_deg)), 0.02)
    return np.log(a / tanb).astype(np.float32)


def hillshade(dem: np.ndarray, res_m: float, az: float = 315.0, alt: float = 45.0) -> np.ndarray:
    """晕渲图，用于仪表盘底图（0–255）。"""
    slope, aspect = slope_aspect(dem, res_m)
    sl = np.radians(slope); asp = np.radians(aspect)
    zen = np.radians(90.0 - alt); azr = np.radians(360.0 - az + 90.0)
    hs = (np.cos(zen) * np.cos(sl)
          + np.sin(zen) * np.sin(sl) * np.cos(azr - asp))
    return np.clip(255.0 * hs, 0, 255).astype(np.uint8)


def compute_terrain(dem: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                    res_m: float) -> dict:
    """一次性算齐全部地形因子。"""
    slope, aspect = slope_aspect(dem, res_m)
    t = tpi(dem, res_m)
    acc = flow_accumulation(dem, res_m)
    w = twi(acc, slope)
    horizon, az_bins = solar.horizon_map(dem, res_m)
    # 天空可视因子按方位取平均 → 每个像元的可见天空比例
    svf = solar.sky_view_factor(horizon).mean(axis=0)
    svf = np.clip(svf, 0.05, 1.0).astype(np.float32)
    # 坡向正弦/余弦：避免 0/360 断裂，同时便于物理解释（sin>0 偏东）
    asp_rad = np.radians(aspect)
    extent = dict(
        lon_min=float(lon[0]), lon_max=float(lon[-1]),
        lat_min=float(lat[0]), lat_max=float(lat[-1]),
        res_m=float(res_m), ny=int(dem.shape[0]), nx=int(dem.shape[1]),
    )
    return dict(
        dem=dem, slope=slope, aspect=aspect,
        aspect_sin=np.sin(asp_rad).astype(np.float32),
        aspect_cos=np.cos(asp_rad).astype(np.float32),
        tpi=t, twi=w, flow_acc=acc, horizon=horizon, az_bins=az_bins,
        svf=svf, extent=extent,
        stats=dict(
            elev_min=float(dem.min()), elev_max=float(dem.max()),
            elev_mean=float(dem.mean()),
            slope_mean=float(slope.mean()), slope_max=float(slope.max()),
        ),
    )


def aspect_label(aspect_deg: float) -> str:
    """坡向中文分类。135–225° 记为阳坡，315–45° 记为阴坡。"""
    a = float(aspect_deg) % 360.0
    if a >= 135 and a < 225:
        return "阳坡（南向）"
    if a >= 225 and a < 315:
        return "半阳坡（西向）"
    if a >= 315 or a < 45:
        return "阴坡（北向）"
    return "半阴坡（东向）"


def position_label(tpi_val: float) -> str:
    if tpi_val <= -3:
        return "谷地/洼地"
    if tpi_val < -1:
        return "下坡位"
    if tpi_val <= 1:
        return "中坡位"
    if tpi_val <= 3:
        return "上坡位"
    return "山脊/凸地"
