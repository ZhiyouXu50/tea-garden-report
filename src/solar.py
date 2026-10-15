"""太阳几何与坡面辐射：把水平面区域辐射转换到 10 m 坡面网格上。

参考实现
--------
- 太阳位置：NOAA Solar Calculator 算法（精度 <0.1°），无需网络。
- 坡面短波：各向同性天空 + 直射投影 + 地表反射三项分解，
  直射项用地形地平线遮蔽（horizon shading）直接置零。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def solar_position(times: pd.DatetimeIndex, lat: float, lon: float,
                   tz_hours: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """返回太阳高度角与方位角（度）。方位角自正北起、顺时针增加。

    关键点：传入的 times 是**当地时间**（Asia/Shanghai），
    因此真太阳时要减去时区偏移 60·tz_hours，否则正午会算成午夜。
    """
    t = pd.DatetimeIndex(times)
    jc = (t.to_julian_date().values - 2451545.0) / 36525.0

    geom_mean_long = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    geom_mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eccent = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    m = np.radians(geom_mean_anom)
    eq_ctr = (np.sin(m) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
              + np.sin(2 * m) * (0.019993 - 0.000101 * jc)
              + np.sin(3 * m) * 0.000289)
    app_long = (geom_mean_long + eq_ctr) - 0.00569 - 0.00478 * np.sin(
        np.radians(125.04 - 1934.136 * jc))

    seconds = 21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))
    obliq = (23.0 + (26.0 + seconds / 60.0) / 60.0) + 0.00256 * np.cos(
        np.radians(125.04 - 1934.136 * jc))
    eps = np.radians(obliq)

    decl = np.degrees(np.arcsin(np.sin(eps) * np.sin(np.radians(app_long))))
    y = np.tan(eps / 2.0) ** 2
    eq_time = 4.0 * np.degrees(
        y * np.sin(2 * np.radians(geom_mean_long))
        - 2 * eccent * np.sin(m)
        + 4 * eccent * y * np.sin(m) * np.cos(2 * np.radians(geom_mean_long))
        - 0.5 * y * y * np.sin(4 * np.radians(geom_mean_long))
        - 1.25 * eccent * eccent * np.sin(2 * m))

    minutes = (t.hour * 60 + t.minute + t.second / 60.0).values
    tst = (minutes + eq_time + 4.0 * lon - 60.0 * tz_hours) % 1440.0
    H = np.radians(tst / 4.0 - 180.0)                 # 时角：上午为负、下午为正

    phi = np.radians(lat)
    dec = np.radians(decl)
    sin_alt = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(H)
    elev = np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))

    # 方位角（自正北顺时针）：atan2(东分量, 北分量)
    az = np.degrees(np.arctan2(-np.cos(dec) * np.sin(H),
                               np.sin(dec) * np.cos(phi)
                               - np.cos(dec) * np.sin(phi) * np.cos(H))) % 360.0
    return elev, az


def _shift(a: np.ndarray, di: int, dj: int, fill: float = np.nan) -> np.ndarray:
    """把数组整体平移 (di 行, dj 列)，空出的部分填 fill。"""
    out = np.full_like(a, fill, dtype=np.float64)
    ny, nx = a.shape
    src = (slice(max(0, di), min(ny, ny + di)), slice(max(0, dj), min(nx, nx + dj)))
    dst = (slice(max(0, -di), min(ny, ny - di)), slice(max(0, -dj), min(nx, nx - dj)))
    out[dst] = a[src]
    return out


def horizon_map(dem: np.ndarray, res_m: float, n_az: int = 32,
                max_dist_m: float = 600.0, step_m: float = 20.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """逐方位地形地平线仰角（度）。

    返回 (horizon[n_az, ny, nx], az_bins[deg])
    """
    ny, nx = dem.shape
    az_bins = np.arange(n_az) * (360.0 / n_az)
    n_step = int(max_dist_m // step_m)
    horizon = np.full((n_az, ny, nx), -90.0, dtype=np.float32)

    for k, azd in enumerate(az_bins):
        # DEM 行号随纬度递增（行 0 = 最南），因此正北 = 行号增大方向
        rad = np.radians(azd)
        dy = np.cos(rad)       # 北为 +1（行增大）
        dx = np.sin(rad)       # 东为 +1（列增大）
        best = np.full((ny, nx), -90.0, dtype=np.float64)
        for s in range(1, n_step + 1):
            d = s * step_m
            di = int(round(dy * s * step_m / res_m))
            dj = int(round(dx * s * step_m / res_m))
            if di == 0 and dj == 0:
                continue
            nb = _shift(dem.astype(np.float64), di, dj)
            ang = np.degrees(np.arctan2(dem - nb, np.hypot(di * res_m, dj * res_m)))
            best = np.fmax(best, ang)
        horizon[k] = best.astype(np.float32)
    return horizon, az_bins


def sky_view_factor(horizon: np.ndarray) -> np.ndarray:
    """由逐方位地平线仰角估算天空可视因子（Dozier & Frew 近似）。

    SVF = (1/N) Σ cos²(H_i)，H_i 为方位 i 上的地平线仰角。
    平地（H=0）→1.0；深谷陡壁 → 趋近 0。
    """
    h = np.clip(horizon, 0.0, 89.9)
    return np.cos(np.radians(h)) ** 2


def shade_mask(horizon: np.ndarray, az_bins: np.ndarray,
               sun_elev: np.ndarray, sun_az: np.ndarray) -> np.ndarray:
    """遮挡掩膜：True = 直射光被地形挡住。形状 (T, ny, nx)。"""
    n_az = len(az_bins)
    idx = np.clip(np.round(np.mod(sun_az, 360.0) / (360.0 / n_az)).astype(int), 0, n_az - 1)
    h_at_sun = horizon[idx]                       # (T, ny, nx)
    return (sun_elev[:, None, None] <= h_at_sun) | (sun_elev[:, None, None] <= 0.5)


def radiation_on_slope(dem: np.ndarray, slope_deg: np.ndarray, aspect_deg: np.ndarray,
                       svf: np.ndarray, horizon: np.ndarray, az_bins: np.ndarray,
                       sun_elev: np.ndarray, sun_az: np.ndarray,
                       dni: np.ndarray, dif: np.ndarray, ghi: np.ndarray,
                       albedo: float = 0.18) -> np.ndarray:
    """坡面总短波辐射 (W/m²)，形状 (T, ny, nx)。"""
    lat_r = np.radians(sun_elev)          # 用高度角做投影
    zen = np.radians(90.0 - sun_elev)
    az_r = np.radians(sun_az)
    sl = np.radians(slope_deg)
    asp = np.radians(aspect_deg)

    # 坡面法线与太阳入射角
    cos_inc = (np.cos(zen)[:, None, None] * np.cos(sl)[None] + 
               np.sin(zen)[:, None, None] * np.sin(sl)[None] *
               np.cos(az_r[:, None, None] - asp[None]))
    cos_inc = np.clip(cos_inc, 0.0, 1.0)

    mask = shade_mask(horizon, az_bins, sun_elev, sun_az)
    direct = dni[:, None, None] * cos_inc * (~mask)
    sky = dif[:, None, None] * svf[None] * (1.0 + np.cos(sl))[None] / 2.0
    refl = albedo * ghi[:, None, None] * (1.0 - np.cos(sl))[None] / 2.0
    return np.maximum(direct + sky + refl, 0.0)


def horizontal_radiation(sun_elev: np.ndarray, dni: np.ndarray, dif: np.ndarray) -> np.ndarray:
    """水平面短波（参考面），与 Open-Meteo 的 shortwave_radiation 同口径。"""
    return np.maximum(dni * np.clip(np.sin(np.radians(sun_elev)), 0, None) + dif, 0.0)
