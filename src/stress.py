"""胁迫指数与风险分级：把微气候场转成"热害 / 旱害 / 霜冻"三类可解释指标。

指标定义
--------
HSI 热害指数：>32 ℃ 的度时累积（℃·h），同时记录 >35 ℃、>38 ℃ 的度时，
              以及日最高气温的绝对水平（用于区分"闷热累积"与"瞬时灼伤"）。
DSI 旱害指数：相对有效水亏缺 (S_opt − S) / (S_opt − S_wilt) 与连续干旱日数放大项的组合。
FSI 霜冻指数：以日最低气温的受害程度映射（春季采摘期关键，供 3–4 月使用）。
复合风险分：max(三类) + 0.25 × 次高值，反映"高温+干旱"复合胁迫的放大效应。
"""
from __future__ import annotations

import numpy as np

import config as C


def _ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """线性映射到 [0,1]，超界截断。"""
    return np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def heat_index(daily: dict, horizon: int) -> dict:
    """热害：取展望窗口内累积度时与日最高气温。"""
    hsu = daily["hsu"][:horizon]
    tmax = daily["tmax"][:horizon]
    hsu_sum = hsu.sum(axis=0)
    tmax_peak = tmax.max(axis=0)
    hours_over_35 = (daily["hsu35"][:horizon] > 0).sum(axis=0)
    hours_over_38 = (daily["hsu38"][:horizon] > 0).sum(axis=0)

    score_hsu = _ramp(hsu_sum, C.HSI_SCALE["low"], C.HSI_SCALE["high"])
    score_peak = _ramp(tmax_peak, C.TEA["heat_mild"], C.TEA["heat_severe"])
    score_days = _ramp(hours_over_38.astype(float), 0.0, float(horizon * 8))
    score = np.maximum.reduce([score_hsu, score_peak * 0.95, score_days])
    return dict(score=score.astype(np.float32), hsu_sum=hsu_sum.astype(np.float32),
                tmax_peak=tmax_peak.astype(np.float32),
                days_over_35=hours_over_35.astype(np.int16),
                days_over_38=hours_over_38.astype(np.int16))


def drought_index(daily: dict, horizon: int) -> dict:
    """旱害：水分亏缺 + 连续无有效降水。"""
    sm = daily["sm_rel_min"][:horizon]
    sm_min = sm.min(axis=0)
    deficit = np.clip((C.TEA["sm_rel_optimal"] - sm_min)
                      / max(C.TEA["sm_rel_optimal"] - C.TEA["sm_rel_wilt"], 1e-6), 0.0, 1.5)
    precip = np.asarray(daily["precip_mm"][:horizon])
    dry_streak = float((precip < C.TEA["dry_spell_mm"]).sum())
    dry_gain = _ramp(np.array(dry_streak), C.TEA["dry_spell_days"] - 3, C.TEA["dry_spell_days"] + 3)
    net = float(np.sum(daily["et0_mm"][:horizon]) - precip.sum())     # 净水分亏缺 mm
    net_gain = _ramp(np.array(net), 8.0, 30.0)
    score = np.clip(0.62 * _ramp(deficit, C.DSI_SCALE["low"], C.DSI_SCALE["high"])
                    + 0.22 * dry_gain + 0.16 * net_gain, 0.0, 1.0)
    return dict(score=score.astype(np.float32), sm_rel_min=sm_min.astype(np.float32),
                deficit=deficit.astype(np.float32), dry_streak=dry_streak,
                net_deficit_mm=round(net, 1),
                et0_sum_mm=round(float(np.sum(daily["et0_mm"][:horizon])), 1),
                precip_sum_mm=round(float(precip.sum()), 1))


def frost_index(daily: dict, horizon: int) -> dict:
    """霜冻：日最低气温映射。4 ℃ 起始受害，0 ℃ 严重。"""
    tmin = daily["tmin"][:horizon]
    tmin_min = tmin.min(axis=0)
    score = _ramp(C.TEA["frost_mild"] - tmin_min,
                  C.TEA["frost_mild"] - C.TEA["frost_severe"] - 2.0,
                  C.TEA["frost_mild"] - C.TEA["frost_severe"] + 2.0)
    cold_nights = (tmin_min <= C.TEA["frost_mild"]).astype(np.int16)
    return dict(score=score.astype(np.float32), tmin_min=tmin_min.astype(np.float32),
                cold_nights=cold_nights)


def composite(h: dict, d: dict, f: dict) -> dict:
    """复合风险分级。"""
    stack = np.stack([h["score"], d["score"], f["score"]])
    order = np.argsort(-stack, axis=0)
    names = np.array(["heat", "drought", "frost"])
    primary = stack.max(axis=0)
    second = np.take_along_axis(stack, order[1:2], axis=0)[0]
    score = np.clip(primary + C.RISK["compound_gain"] * second, 0.0, 1.0)
    cls = np.where(score < C.RISK["low_upper"], 0,
                   np.where(score < C.RISK["high_lower"], 1, 2)).astype(np.int8)
    dominant = names[order[0]]
    dominant[score < C.RISK["low_upper"]] = "none"
    # 复合型：主控与次控都达到"中"以上
    compound = (order[0] != order[1]) & (second > C.RISK["low_upper"]) & \
               (primary > C.RISK["low_upper"]) & (score >= C.RISK["low_upper"])
    return dict(score=score.astype(np.float32), cls=cls, dominant=dominant,
                compound=compound, primary=primary.astype(np.float32),
                second=second.astype(np.float32))


DOMINANT_CN = {"heat": "热害", "drought": "旱害", "frost": "霜冻",
               "heat+drought": "高温干旱复合", "none": "无显著胁迫"}
CLASS_CN = {0: "低", 1: "中", 2: "高"}


def label_of(dom: str, compound: bool) -> str:
    if dom == "none":
        return DOMINANT_CN["none"]
    if compound and dom in ("heat", "drought"):
        return DOMINANT_CN["heat+drought"]
    return DOMINANT_CN.get(dom, dom)


def assess(daily: dict, horizon: int | None = None) -> dict:
    """一次算齐三类指标与复合风险。"""
    hz = horizon or C.RISK["horizon_days"]
    hz = min(hz, len(daily["dates"]))
    h = heat_index(daily, hz)
    d = drought_index(daily, hz)
    f = frost_index(daily, hz)
    c = composite(h, d, f)
    labels = np.empty(c["dominant"].shape, dtype=object)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            labels[i, j] = label_of(str(c["dominant"][i, j]), bool(c["compound"][i, j]))
    c["dominant_cn"] = labels
    return dict(heat=h, drought=d, frost=f, comp=c, horizon_days=hz)
