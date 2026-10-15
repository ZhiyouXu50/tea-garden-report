"""历史灾情回溯验证：用真实再分析数据回放历年夏季，评估预警提前量与漏报／误报率。

与"自说自话"的精度报告不同，本模块的做法是：
  1. 从 Open-Meteo 历史再分析（ERA5 / ERA5-Land 驱动）重建各年夏季逐小时气象；
  2. 用与实时模式**完全相同**的指标口径计算逐日风险等级；
  3. 用"累积热害度时"与"累积土壤缺水度"定义**灾情形成日**
     （植物受害是累积过程，不是瞬时事件），与"预警首日"对比得到**提前量**；
  4. 逐日评估：把"未来 7 天内是否出现新的累积胁迫增长"作为真值标签，
     统计预警日的**准确率（1 − 误报率）**与对胁迫日的**召回率（1 − 漏报率）**。

注意口径
--------
- 热害：日累积 >38 ℃ 度时（℃·h）。灾情形成日 = 累积值首次跨过 30 ℃·h。
- 干旱：逐日土壤相对有效水亏缺 max(0, 0.72 − S_rel) 的累积（无量纲·日）。
  灾情形成日 = 累积值首次跨过 3.0（约相当于连续 10 天缺水量 0.30）。
- 区域气象是灾情的驱动因子，所以"提前量"实质来自累积过程的滞后性——
  这正是预警系统的价值：把"几天后才致害"的累积过程提前揭示出来。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from . import stress

HEAT_DAMAGE_CUM = 30.0      # ℃·h 累积热害度时阈值
DROUGHT_DAMAGE_CUM = 3.0    # 累积缺水度阈值
DAMAGE_LOOKAHEAD = 7        # 判定"是否继续累积胁迫"的展望天数
DAMAGE_GROWTH_HEAT = 5.0    # ℃·h
DAMAGE_GROWTH_DROUGHT = 0.4


def regional_daily(df: pd.DataFrame) -> pd.DataFrame:
    """逐小时 → 逐日区域指标。"""
    d = df.copy()
    t = d["temperature_2m"]
    d["hsu35"] = np.maximum(t - C.TEA["heat_mild"], 0.0)
    d["hsu38"] = np.maximum(t - C.TEA["heat_moderate"], 0.0)
    d["hsu"] = np.maximum(t - C.TEA["heat_hours_base"], 0.0)
    d["vpd"] = _vpd(t.values, d["relative_humidity_2m"].values)
    g = d.resample("D")
    out = pd.DataFrame({
        "tmax": g["temperature_2m"].max(),
        "tmin": g["temperature_2m"].min(),
        "tmean": g["temperature_2m"].mean(),
        "precip": g["precipitation"].sum(),
        "et0": g["et0_fao_evapotranspiration"].sum(),
        "rh_min": g["relative_humidity_2m"].min(),
        "vpd_max": g["vpd"].max(),
        "hsu": g["hsu"].sum(),
        "hsu35": g["hsu35"].sum(),
        "hsu38": g["hsu38"].sum(),
        "sm_vol": g["soil_moisture_3_to_9cm"].min(),
    })
    out["sm_rel"] = np.clip((out["sm_vol"] - 0.12) / (0.35 - 0.12), 0.0, 1.0)
    out["drought_sev"] = np.maximum(0.0, C.TEA["sm_rel_optimal"] - out["sm_rel"])
    return out.dropna(how="all")


def _vpd(t: np.ndarray, rh: np.ndarray) -> np.ndarray:
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return np.maximum(es * (1 - np.clip(rh, 0, 100) / 100.0), 0.0)


def daily_risk(rd: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """逐日滚动窗口风险分（与实时模式同一套公式与阈值）。"""
    n = len(rd)
    score = np.zeros(n)
    cls = np.zeros(n, dtype=int)
    for i in range(n):
        lo = max(0, i - window + 1)
        w = rd.iloc[lo:i + 1]
        hsu38 = float(w["hsu38"].sum())
        sm_rel = float(w["sm_rel"].min())
        dry = int((w["precip"] < C.TEA["dry_spell_mm"]).sum())
        h = 0.6 * np.clip(hsu38 / 40.0, 0, 1) + 0.4 * np.clip((w["tmax"].max() - 35) / 5, 0, 1)
        dr = (0.62 * np.clip((C.TEA["sm_rel_optimal"] - sm_rel)
                             / (C.TEA["sm_rel_optimal"] - C.TEA["sm_rel_wilt"]), 0, 1.5)
              + 0.22 * np.clip((dry - 4) / 6, 0, 1)
              + 0.16 * np.clip((float(w["drought_sev"].mean()) - 0.05) / 0.25, 0, 1))
        sc = min(1.0, max(h, dr) + C.RISK["compound_gain"] * min(h, dr))
        score[i] = sc
        cls[i] = 0 if sc < C.RISK["low_upper"] else (1 if sc < C.RISK["high_lower"] else 2)
    return pd.DataFrame({"score": score, "cls": cls}, index=rd.index)


def season_metrics(rd: pd.DataFrame) -> dict:
    """单年夏季气候与胁迫指标。"""
    return {
        "days": int(len(rd)),
        "tmax_peak": round(float(rd["tmax"].max()), 1),
        "tmax_peak_date": str(rd["tmax"].idxmax().date()),
        "days_ge35": int((rd["tmax"] >= 35).sum()),
        "days_ge38": int((rd["tmax"] >= 38).sum()),
        "days_ge40": int((rd["tmax"] >= 40).sum()),
        "precip_total_mm": round(float(rd["precip"].sum()), 1),
        "et0_total_mm": round(float(rd["et0"].sum()), 1),
        "max_dry_streak": int(_max_streak(rd["precip"].values < C.TEA["dry_spell_mm"])),
        "sm_rel_min": round(float(rd["sm_rel"].min()), 3),
        "days_sm_stressed": int((rd["sm_rel"] < C.TEA["sm_rel_stress"]).sum()),
        "cum_drought_severity": round(float(rd["drought_sev"].sum()), 2),
        "hsu38_season": round(float(rd["hsu38"].sum()), 1),
    }


def _max_streak(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def backtest_year(year: int, rd: pd.DataFrame, window: int = 3) -> dict:
    """单年回溯：预警事件段 + 逐日漏报/误报统计。"""
    rk = daily_risk(rd, window)
    df = pd.concat([rd, rk], axis=1)
    df["cum_heat"] = df["hsu38"].cumsum()
    df["cum_drought"] = df["drought_sev"].cumsum()

    heat_damage = df.index[df["cum_heat"] >= HEAT_DAMAGE_CUM]
    drought_damage = df.index[df["cum_drought"] >= DROUGHT_DAMAGE_CUM]
    damage_heat = heat_damage[0] if len(heat_damage) else None
    damage_drought = drought_damage[0] if len(drought_damage) else None

    # ---- 预警事件段 ----
    events, idx = [], list(df.index)
    i = 0
    while i < len(idx):
        if df["cls"].iloc[i] >= 1:
            j = i
            while j + 1 < len(idx) and df["cls"].iloc[j + 1] >= 1:
                j += 1
            seg = df.iloc[i:j + 1]
            events.append({
                "start": str(idx[i].date()), "end": str(idx[j].date()),
                "days": int(j - i + 1), "peak_cls": int(seg["cls"].max()),
                "peak_score": round(float(seg["score"].max()), 3),
                "peak_tmax": round(float(seg["tmax"].max()), 1),
            })
            i = j + 1
        else:
            i += 1

    lead_heat = lead_drought = None
    if events:
        first_warn = pd.Timestamp(events[0]["start"])
        if damage_heat is not None:
            lead_heat = int((damage_heat - first_warn).days)
        if damage_drought is not None:
            lead_drought = int((damage_drought - first_warn).days)

    # ---- 逐日评估 ----
    dh = df["cum_heat"].values
    dd = df["cum_drought"].values
    n = len(df)
    growth = np.zeros(n, dtype=bool)
    for k in range(n):
        e = min(n - 1, k + DAMAGE_LOOKAHEAD)
        growth[k] = ((dh[e] - dh[k]) >= DAMAGE_GROWTH_HEAT
                     or (dd[e] - dd[k]) >= DAMAGE_GROWTH_DROUGHT)
    warn = (df["cls"].values >= 1)
    # 只在季中之后评估，避免季初"未来窗口不足"造成的假落空
    valid = np.arange(n) < n - DAMAGE_LOOKAHEAD
    w_days = warn & valid
    tp_days = int((w_days & growth).sum())
    fp_days = int((w_days & ~growth).sum())
    precision = tp_days / max(tp_days + fp_days, 1)

    d_days = growth & valid
    led = np.zeros(n, dtype=bool)
    for k in np.where(warn)[0]:
        led[max(0, k):min(n, k + 6)] = True
    recall = float((d_days & led).sum()) / max(int(d_days.sum()), 1)

    return {
        "year": year,
        "metrics": season_metrics(rd),
        "n_warning_segments": len(events),
        "warning_days": int((rk["cls"] >= 1).sum()),
        "high_risk_days": int((rk["cls"] == 2).sum()),
        "events": events[:12],
        "damage_day_heat": str(damage_heat.date()) if damage_heat is not None else None,
        "damage_day_drought": str(damage_drought.date()) if damage_drought is not None else None,
        "lead_days_heat": lead_heat,
        "lead_days_drought": lead_drought,
        "day_eval": {
            "warning_days_evaluated": int(w_days.sum()),
            "true_positive_days": tp_days,
            "false_alarm_days": fp_days,
            "precision": round(precision, 3),
            "false_alarm_rate": round(1.0 - precision, 3),
            "stress_days_evaluated": int(d_days.sum()),
            "recall": round(recall, 3),
            "miss_rate": round(1.0 - recall, 3),
            "criteria": (f"预警日判定为命中：未来 {DAMAGE_LOOKAHEAD} 天内累积热害度时增加 ≥"
                         f"{DAMAGE_GROWTH_HEAT} ℃·h 或累积缺水度增加 ≥{DAMAGE_GROWTH_DROUGHT}"),
        },
        "series": {
            "date": [str(d.date()) for d in df.index],
            "score": [round(float(v), 3) for v in df["score"]],
            "cls": [int(v) for v in df["cls"]],
            "tmax": [round(float(v), 1) for v in df["tmax"]],
            "tmin": [round(float(v), 1) for v in df["tmin"]],
            "precip": [round(float(v), 1) for v in df["precip"]],
            "sm_rel": [round(float(v), 3) for v in df["sm_rel"]],
        },
    }


def run_backtest(years: dict) -> dict:
    out = {}
    for y, pack in years.items():
        try:
            rd = regional_daily(pack["df"])
            out[int(y)] = backtest_year(int(y), rd)
            r = out[int(y)]
            de = r["day_eval"]
            print(f"[backtest] {y}: 预警段 {r['n_warning_segments']} 个 / 高风险日 "
                  f"{r['high_risk_days']} 天 / 热害提前量 {r['lead_days_heat']} 天 / "
                  f"准确率 {de['precision']*100:.0f}% / 召回率 {de['recall']*100:.0f}%")
        except Exception as e:  # noqa: BLE001
            print(f"[backtest] {y} 失败: {e}")
    # 汇总
    if out:
        lead = [v["lead_days_heat"] for v in out.values() if v["lead_days_heat"] is not None]
        prec = [v["day_eval"]["precision"] for v in out.values()]
        rec = [v["day_eval"]["recall"] for v in out.values()]
        out["summary"] = {
            "years": sorted(k for k in out if isinstance(k, int)),
            "lead_days_mean": round(float(np.mean(lead)), 1) if lead else None,
            "lead_days_max": int(np.max(lead)) if lead else None,
            "precision_mean": round(float(np.mean(prec)), 3),
            "recall_mean": round(float(np.mean(rec)), 3),
            "note": ("准确率/召回率按『预警日 vs 未来 7 天是否继续累积胁迫』逐日统计；"
                     "提前量为『首次预警日』到『累积胁迫跨过致害阈值日』的天数。"),
        }
    return out
