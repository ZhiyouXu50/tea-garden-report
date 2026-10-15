"""微气候降尺度：把公里级区域气象强迫场还原到 10 m 网格。

模型结构（物理内核 + 可学习残差）
--------------------------------
1) 高程项：环境温度直减率 Γ（默认 −0.60 ℃/100 m）
2) 辐射项：坡面短波与参考水平面的差值 × k_rad，捕捉阳坡/阴坡昼间温差
3) 冷池项：夜间谷地负地形位置指数 → 降温，捕捉"谷底夜冷、山腰暖带"
4) 湿度：露点在本尺度视为均匀场，由修正后的气温反算相对湿度（Magnus 公式）
5) 土壤水：FAO-56 单层水量平衡（降水/灌溉 − 蒸散 − 坡面产流），
   产流系数随坡度增大，因此陡坡茶园先旱、谷地茶园后旱

k_rad / k_drain 等系数为"文献先验 + 待校准"参数：先用先验值给出物理可解释的场，
再由 src/models/downscaler_lstm.py 用茶园小气候站观测做残差学习。

spin-up：历史情景回放时，先跑一段"预热期"把土壤水桶积分到平衡，
只对评估窗口内的日期出结果，避免初值主导结论。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from . import solar


def dewpoint_from_rh(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """由气温与相对湿度反算露点（Magnus，b=17.625, c=243.04）。"""
    rh = np.clip(rh, 1.0, 100.0)
    g = np.log(rh / 100.0) + (17.625 * t_c) / (243.04 + t_c)
    return 243.04 * g / (17.625 - g)


def rh_from_dewpoint(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    g = (17.625 * td_c) / (243.04 + td_c)
    return np.clip(100.0 * np.exp(g - (17.625 * t_c) / (243.04 + t_c)), 5.0, 100.0)


def _sat_vpd(t_c: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def vpd_kpa(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    es = _sat_vpd(t_c)
    return np.maximum(es * (1.0 - np.clip(rh, 0, 100) / 100.0), 0.0)


def run_downscaling(terrain: dict, forcing: pd.DataFrame,
                    params: dict | None = None,
                    series_cells: list[tuple[int, int]] | None = None,
                    horizon_days: int | None = None,
                    assess_start: str | None = None) -> dict:
    """执行降尺度，返回逐日面场 + 代表单元逐小时序列。"""
    p = dict(C.DOWNSCALE)
    if params:
        p.update(params)
    t = dict(C.TEA)
    dem, slope, aspect, svf = terrain["dem"], terrain["slope"], terrain["aspect"], terrain["svf"]
    horizon, az_bins = terrain["horizon"], terrain["az_bins"]
    ny, nx = dem.shape

    f = forcing.copy()
    f = f.loc[:, ~f.columns.duplicated()]
    f["dew_point_2m"] = dewpoint_from_rh(f["temperature_2m"].values,
                                         f["relative_humidity_2m"].values)
    for col, default in (("et0_fao_evapotranspiration", np.nan),
                         ("soil_moisture_3_to_9cm", 0.25)):
        if col not in f:
            f[col] = default
    f = f.ffill().bfill()

    elev_ref = float(f.attrs.get("ref_elev", np.median(dem)))
    sun_elev, sun_az = solar.solar_position(f.index, C.SITE["lat"], C.SITE["lon"])

    dni = np.nan_to_num(f["direct_normal_irradiance"].values, nan=0.0)
    dif = np.nan_to_num(f["diffuse_radiation"].values, nan=0.0)
    ghi = np.nan_to_num(f["shortwave_radiation"].values, nan=0.0)
    T_ref = f["temperature_2m"].values
    Td = f["dew_point_2m"].values
    P = np.nan_to_num(f["precipitation"].values, nan=0.0)
    ET0r = np.nan_to_num(f["et0_fao_evapotranspiration"].values, nan=0.0)
    SMvol = np.nan_to_num(f["soil_moisture_3_to_9cm"].values, nan=0.25)
    SMvol = np.where(SMvol <= 0.02, np.nanmedian(SMvol[SMvol > 0.02]) if (SMvol > 0.02).any() else 0.25, SMvol)

    # ---- 固定空间项 ----
    lapse_field = p["lapse_rate"] / 100.0 * (dem.astype(np.float64) - elev_ref)
    drain_field = -np.clip(p["k_drain"] * np.maximum(-terrain["tpi"], 0.0),
                           0.0, p["drain_cap"])
    sw_ref_h = solar.horizontal_radiation(sun_elev, dni, dif)

    sm_rel0 = float(np.clip((np.median(SMvol) - 0.12) / (0.35 - 0.12), 0.10, 1.0))
    sm_cell = np.full((ny, nx), sm_rel0 * t["TAW_mm"], dtype=np.float64)
    runoff_frac = np.clip(0.45 + 0.45 * np.sin(np.radians(slope)), 0.30, 0.90)

    # ---- 评估窗口（其余小时仅用于土壤水预热）----
    all_dates = sorted(set(pd.DatetimeIndex(f.index).normalize()))
    if assess_start is not None:
        dates = [d for d in all_dates if d >= pd.Timestamp(assess_start)]
    elif horizon_days:
        dates = all_dates[-horizon_days:]
    else:
        dates = all_dates
    nd = len(dates)
    day_idx = {d: i for i, d in enumerate(dates)}
    warmup_days = len(all_dates) - nd

    acc = {k: np.zeros((nd, ny, nx), dtype=np.float32) for k in
           ("tmax", "tmin", "hsu", "hsu35", "hsu38", "sm_rel_min", "sw_sum",
            "vpd_max", "rh_min", "et_sum", "stress_hours", "tday_sum", "tday_cnt")}
    # 注意：只有"最大值/最小值"类累积量才能用 ±inf 初始化；
    # 求和类（sw_sum / et_sum）必须从 0 起，否则 -inf + x 恒为 -inf。
    for k in ("tmax", "vpd_max"):
        acc[k][:] = -np.inf
    for k in ("tmin", "sm_rel_min", "rh_min"):
        acc[k][:] = np.inf
    precip_day = np.zeros(nd)
    et0_day = np.zeros(nd)
    treg_max = np.full(nd, -np.inf)
    tmin_reg = np.full(nd, np.inf)
    sm_rel = np.full((ny, nx), sm_rel0, dtype=np.float32)

    series = {c: {k: [] for k in ("t", "temp", "rh", "sm_rel", "sw", "vpd")}
              for c in (series_cells or [])}
    LAI_Kc = t["Kc"]

    for i, (ts, t_ref) in enumerate(zip(f.index, T_ref)):
        se, sa = float(sun_elev[i]), float(sun_az[i])
        sw_plot = solar.radiation_on_slope(
            dem, slope, aspect, svf, horizon, az_bins,
            np.array([se]), np.array([sa]), np.array([dni[i]]),
            np.array([dif[i]]), np.array([ghi[i]]))[0]
        sw_ratio = sw_plot / max(sw_ref_h[i], 1.0)

        # ---- 逐单元水量平衡（FAO-56 单层桶）----
        infil = P[i] * (1.0 - runoff_frac)
        et0_plot = ET0r[i] * np.clip(sw_ratio, 0.25, 1.6)
        ks = np.clip((sm_rel - t["sm_rel_wilt"])
                     / max(t["sm_rel_optimal"] - t["sm_rel_wilt"], 1e-6), 0.0, 1.0)
        eta = et0_plot * LAI_Kc * ks
        sm_cell = np.clip(sm_cell + infil - eta, 0.0, t["TAW_mm"])
        sm_rel = (sm_cell / t["TAW_mm"]).astype(np.float32)

        d = day_idx.get(ts.normalize())
        if d is None:                       # 预热期：只积分土壤水
            continue

        rad_term = np.clip(p["k_rad"] * (sw_plot - sw_ref_h[i]),
                           -p["rad_cap"], p["rad_cap"]) \
            * (1.0 - p["svf_rad_weight"] * (1.0 - svf))
        temp = t_ref + lapse_field + rad_term + (drain_field if se < 5.0 else 0.0)
        rh_plot = rh_from_dewpoint(temp, Td[i])
        vpd = vpd_kpa(temp, rh_plot)

        acc["tmax"][d] = np.maximum(acc["tmax"][d], temp)
        acc["tmin"][d] = np.minimum(acc["tmin"][d], temp)
        acc["sw_sum"][d] += sw_plot.astype(np.float32)
        acc["et_sum"][d] += eta.astype(np.float32)
        acc["vpd_max"][d] = np.maximum(acc["vpd_max"][d], vpd.astype(np.float32))
        acc["rh_min"][d] = np.minimum(acc["rh_min"][d], rh_plot.astype(np.float32))
        acc["sm_rel_min"][d] = np.minimum(acc["sm_rel_min"][d], sm_rel)
        acc["hsu"][d] += np.maximum(temp - t["heat_hours_base"], 0.0).astype(np.float32)
        acc["hsu35"][d] += np.maximum(temp - t["heat_mild"], 0.0).astype(np.float32)
        acc["hsu38"][d] += np.maximum(temp - t["heat_moderate"], 0.0).astype(np.float32)
        acc["stress_hours"][d] += (vpd > 2.0).astype(np.float32)
        if 10 <= ts.hour < 16:                 # 昼间（10–16 时）均温：坡向差异比日最高气温更显著
            acc["tday_sum"][d] += temp.astype(np.float32)
            acc["tday_cnt"][d] += 1.0
        precip_day[d] += P[i]
        et0_day[d] += ET0r[i]
        treg_max[d] = max(treg_max[d], t_ref)
        tmin_reg[d] = min(tmin_reg[d], t_ref)

        for c in series:
            series[c]["t"].append(ts.strftime("%m-%d %H:%M"))
            series[c]["temp"].append(round(float(temp[c]), 2))
            series[c]["rh"].append(round(float(rh_plot[c]), 1))
            series[c]["sm_rel"].append(round(float(sm_rel[c]), 3))
            series[c]["sw"].append(round(float(sw_plot[c]), 1))
            series[c]["vpd"].append(round(float(vpd[c]), 2))

    # 昼间均温 = 累积温度 / 有效小时数
    acc["tday_mean"] = (acc.pop("tday_sum")
                        / np.maximum(acc.pop("tday_cnt"), 1.0)).astype(np.float32)

    # 把 1 维时间序列并入 daily，便于下游（stress / risk_model）只用一个字典
    acc["dates"] = [d.strftime("%Y-%m-%d") for d in dates]
    acc["precip_mm"] = precip_day
    acc["et0_mm"] = et0_day
    acc["tmax_region"] = treg_max
    acc["tmin_region"] = tmin_reg

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "daily": acc,
        "daily_scalar": {
            "precip_mm": np.round(precip_day, 1).tolist(),
            "et0_mm": np.round(et0_day, 2).tolist(),
            "tmax_region": np.round(treg_max, 1).tolist(),
            "tmin_region": np.round(tmin_reg, 1).tolist(),
        },
        "series": series,
        "meta": {
            "params": p,
            "elev_ref_m": round(elev_ref, 1),
            "n_hours": int(len(f)),
            "warmup_days": int(warmup_days),
            "sm0_rel": round(sm_rel0, 3),
        },
    }
