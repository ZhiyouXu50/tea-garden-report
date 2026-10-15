"""梅家坞微气候风险预警与行动决策助手 —— 一键运行管线。

用法
----
    python run_pipeline.py                  # 完整运行（含历史回溯与代理模型训练）
    python run_pipeline.py --fast           # 减少蒙特卡洛抽样，快速出结果
    python run_pipeline.py --refresh        # 强制重新抓取气象与 DEM 数据
    python run_pipeline.py --skip-backtest  # 跳过历年回溯

产物
----
    outputs/run_config.json        本次运行的全部参数（可复现）
    outputs/dashboard_data.json    仪表盘数据包
    outputs/risk_report.md         分析报告
    outputs/plot_units.geojson     管理单元矢量
    app/dashboard.html             单文件可运行 Demo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                    # noqa: E402
from src import (advice, backtest, build_dashboard, fetch_dem, fetch_weather,  # noqa: E402
                 microclimate, plots, render, risk_model, stress)
from src import terrain as terr                       # noqa: E402

T0 = time.time()
LOG_PATH = Path(__file__).resolve().parent / "outputs" / "run.log"


def log(msg: str) -> None:
    line = f"[{time.time() - T0:6.1f}s] {msg}"
    print(line, flush=True)
    try:                                     # 同时写文件，避免终端编码问题丢日志
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:                        # noqa: BLE001
        pass


# --------------------------------------------------------------------------
def build_mode(terrain: dict, forcing: pd.DataFrame, meta: dict, units: list[dict],
               cells: list[tuple[int, int]], assess_start: str | None,
               horizon: int, mc_draws: int) -> tuple[dict, dict, dict]:
    """单个情景/模式：降尺度 → 风险评估 → 参数不确定性 → 地块状态。"""
    res = microclimate.run_downscaling(terrain, forcing, series_cells=cells,
                                       horizon_days=horizon, assess_start=assess_start)
    a = stress.assess(res["daily"], horizon)
    # 种子必须可复现：不能用内置 hash()（CPython 对 str 的 hash 逐进程随机化），
    # 否则同一份输入两次运行会得到不同的蒙特卡洛抽样与风险分级。
    mc = risk_model.mc_risk_ensemble(terrain, forcing, horizon,
                                     draws=mc_draws,
                                     seed=zlib.crc32(f"meijiawu|{meta.get('tag')}".encode()) % 100_000)

    daily, comp = res["daily"], a["comp"]
    tea = plots.tea_suitability_mask(terrain)
    hz = a["horizon_days"]

    tmax_peak = daily["tmax"][:hz].max(axis=0)
    tday_peak = daily["tday_mean"][:hz].max(axis=0)
    # 坡向对比在"适茶地形带"（高程 25–230 m、坡度 4–25°）上统计，
    # 这样四个坡向都有足够样本，避免被 TWI/汇流约束裁掉某类坡向
    zone = ((terrain["dem"] >= 25) & (terrain["dem"] <= 230)
            & (terrain["slope"] >= 4.0) & (terrain["slope"] <= 25.0))
    asp = terrain["aspect"]
    quads = {
        "阳坡(南)": zone & (asp >= 135) & (asp < 225),
        "阴坡(北)": zone & ((asp >= 315) | (asp < 45)),
        "东坡": zone & (asp >= 45) & (asp < 135),
        "西坡": zone & (asp >= 225) & (asp < 315),
    }
    q_tmax = {k: round(float(tmax_peak[m].mean()), 2) for k, m in quads.items() if m.any()}
    q_tday = {k: round(float(tday_peak[m].mean()), 2) for k, m in quads.items() if m.any()}
    pairs = [(a, b) for i, a in enumerate(q_tmax) for b in list(q_tmax)[i + 1:]]
    best_pair = max(pairs, key=lambda p: abs(q_tmax[p[0]] - q_tmax[p[1]])) if pairs else None
    zone_vals = tmax_peak[zone]
    zone_tday = tday_peak[zone]
    region_peak = float(np.max(daily["tmax_region"][:hz]))

    # ---- 统一阈值盲区度量 ------------------------------------------------
    # 两个口径都要看：
    #   35 ℃（轻度热害起点，与区域预警口径一致）——"区域说没事、地块已受害"
    #   32 ℃（度时累积起点，最敏感）           ——"区域还没热、阳坡已在累积胁迫"
    # 用"最热日受影响像元占比"而非均值，避免真实但细微的效应被四舍五入抹平。
    mild = float(C.TEA["heat_mild"])
    onset = float(C.TEA["heat_hours_base"])
    masked_days, silent_hot, onset_hot = [], [], []
    silent_days = onset_days = 0
    for k in range(hz):
        reg = float(daily["tmax_region"][k])
        cell = daily["tmax"][k][tea]
        hot = float((cell >= mild).mean())
        stressed = float((cell >= onset).mean())
        masked_days.append({"date": res["dates"][k], "region_tmax": round(reg, 1),
                            "hot_cell_frac": round(hot, 3),
                            "stress_cell_frac": round(stressed, 3)})
        if reg < mild:
            silent_days += 1
            if hot > 0:
                silent_hot.append(hot)
        if reg < onset:
            onset_days += 1
            onset_hot.append(stressed)

    # 地块级最高温统一在"适茶地形带"上取，和报告里的同名指标、以及坡向统计口径保持一致
    cell_peak = float(zone_vals.max())
    blindspot = {
        "region_peak_c": round(region_peak, 1),
        "cell_peak_c": round(cell_peak, 1),
        "exceedance_c": round(cell_peak - region_peak, 2),
        "mild_threshold_c": mild,
        "silent_days": silent_days,                 # 区域未越 35 ℃ 的天数
        "silent_hot_days": len(silent_hot),         # 其中仍有 ≥35 ℃ 地块的天数
        "hot_frac_max_pct": round(100.0 * max(silent_hot), 2) if silent_hot else 0.0,
        "onset_threshold_c": onset,
        "onset_silent_days": onset_days,            # 区域未越 32 ℃ 的天数
        "onset_hot_frac_max_pct": round(100.0 * max(onset_hot), 2) if onset_hot else 0.0,
        "onset_hot_frac_mean_pct": round(100.0 * float(np.mean(onset_hot)), 2) if onset_hot else 0.0,
    }

    layer_stats = {
        "microclimate_spread_c": round(float(zone_vals.max() - zone_vals.min()), 2),
        "microclimate_spread_daytime_c": round(float(zone_tday.max() - zone_tday.min()), 2),
        "microclimate_p05_p95_c": round(float(np.percentile(zone_vals, 95)
                                              - np.percentile(zone_vals, 5)), 2),
        "tmax_peak_mean": round(float(zone_vals.mean()), 2),
        "tmax_peak_max": round(float(zone_vals.max()), 2),
        "aspect_quadrant_tmax": q_tmax,
        "aspect_quadrant_tday": q_tday,
        "sunny_minus_shady": round(q_tmax.get("阳坡(南)", 0) - q_tmax.get("阴坡(北)", 0), 2)
        if "阳坡(南)" in q_tmax and "阴坡(北)" in q_tmax else None,
        "sunny_minus_shady_daytime": round(q_tday.get("阳坡(南)", 0) - q_tday.get("阴坡(北)", 0), 2)
        if "阳坡(南)" in q_tday and "阴坡(北)" in q_tday else None,
        "max_aspect_pair": list(best_pair) if best_pair else None,
        "max_aspect_diff": round(abs(q_tmax[best_pair[0]] - q_tmax[best_pair[1]]), 2)
        if best_pair else None,
        "comparison_basis": "适茶地形带（高程 25–230 m、坡度 4–25°）",
        "zone_cells": int(zone.sum()),
        "region_tmax_peak": round(region_peak, 1),
        "blindspot": blindspot,
        # 兼容字段：区域未越 32 ℃ 的日子里，微气候已达 32 ℃ 的茶园像元占比（最热日）
        "region_underestimates_cells_frac": round(blindspot["onset_hot_frac_max_pct"] / 100.0, 4),
        "tea_cells": int(tea.sum()),
    }

    unit_states = {}
    for u in units:
        i, j = u["cell_index"]
        unit_states[u["unit_id"]] = {
            "cls": int(mc["cls"][i, j]), "score": float(mc["score"][i, j]),
            "agree": float(mc["agree"][i, j]), "draws": mc["draws"],
            "dominant_cn": stress.label_of(str(comp["dominant"][i, j]),
                                           bool(comp["compound"][i, j])),
            "hsu_sum": float(a["heat"]["hsu_sum"][i, j]),
            "tmax_peak": float(a["heat"]["tmax_peak"][i, j]),
            "days_over_35": int(a["heat"]["days_over_35"][i, j]),
            "days_over_38": int(a["heat"]["days_over_38"][i, j]),
            "sm_rel_min": float(a["drought"]["sm_rel_min"][i, j]),
            "tmin_min": float(a["frost"]["tmin_min"][i, j]),
            "raw_cls": int(comp["cls"][i, j]), "raw_score": float(comp["score"][i, j]),
        }

    advices = advice.build_all(units, unit_states, res["dates"])
    summary = advice.overall_summary(advices, meta.get("label", ""), res["dates"])

    base = render.base_map(terrain)
    risk_img = render.draw_units(
        render.overlay(mc["score"], 0.0, 1.0, "risk", 0.82, tea, base), terrain, units)
    dev = np.where(zone, tmax_peak - float(np.mean(zone_vals)), np.nan)
    dev_img = render.draw_units(
        render.overlay(dev, -1.5, 3.5, "temp", 0.75, tea, base), terrain, units)
    sm = np.where(tea, daily["sm_rel_min"][:hz].min(axis=0), np.nan)
    sm_img = render.draw_units(
        render.overlay(sm, 0.15, 0.95, "moist", 0.78, tea, base), terrain, units)

    series = {u["unit_id"]: res["series"].get(tuple(u["cell_index"]), {}) for u in units}

    mode = {
        "tag": meta.get("tag", ""), "label": meta.get("label", ""),
        "note": meta.get("note", ""), "window": summary["window"],
        "dates": res["dates"], "daily_scalar": res["daily_scalar"],
        "summary": summary, "advices": advices,
        "patrol": advice.patrol_plan(advices),
        "units": unit_states, "layer_stats": layer_stats, "masked_days": masked_days,
        "images": {
            "risk": render.to_png_b64(risk_img), "deviation": render.to_png_b64(dev_img),
            "moisture": render.to_png_b64(sm_img),
        },
        "series": series, "model_meta": res["meta"],
        "regional": {
            "tmax_region_peak": round(region_peak, 1),
            "precip_mm": res["daily_scalar"]["precip_mm"],
            "et0_mm": res["daily_scalar"]["et0_mm"],
            "tmin_region_min": round(float(np.min(res["daily_scalar"]["tmin_region"])), 1),
        },
        "source": meta.get("source", ""),
        "grid": {"ny": int(terrain["dem"].shape[0]), "nx": int(terrain["dem"].shape[1]),
                 "extent": terrain["extent"]},
    }

    # 训练用：特征立方体 + 弱标签
    tea_mask = plots.tea_suitability_mask(terrain)
    fp = risk_model.feature_stack(terrain, daily, hz, forcing)
    fp["mask"] = tea_mask
    lp = {"cls": mc["cls"], "score": mc["score"], "agree": mc["agree"]}
    return mode, fp, lp


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--mc-draws", type=int, default=None)
    args = ap.parse_args()
    mc_draws = args.mc_draws or (4 if args.fast else C.ML["mc_draws"])
    horizon = C.RISK["horizon_days"]

    C.dump_config(C.OUT / "run_config.json")
    log(f"站点：{C.SITE['name']}｜网格 {C.GRID_RES_M:.0f} m｜展望 {horizon} 天｜MC 抽样 {mc_draws} 次")

    # ---------- 1. 地形 ----------
    dem_pack = fetch_dem.load_dem(refresh=args.refresh)
    terrain = terr.compute_terrain(dem_pack["dem"], dem_pack["lat"], dem_pack["lon"],
                                   float(dem_pack["res_m"][0]))
    terrain["lat"], terrain["lon"] = dem_pack["lat"], dem_pack["lon"]
    log(f"地形完成 {terrain['dem'].shape}｜高程 {terrain['stats']['elev_min']:.0f}–"
        f"{terrain['stats']['elev_max']:.0f} m｜平均坡度 {terrain['stats']['slope_mean']:.1f}°")

    # ---------- 2. 管理单元 ----------
    units = plots.extract_units(terrain)
    cells = [tuple(u["cell_index"]) for u in units]
    log(f"管理单元 {len(units)} 个，合计 {round(sum(u['area_ha'] for u in units), 2)} ha，"
        f"坡向分布 { {k: sum(1 for u in units if u['aspect_label'] == k) for k in set(u['aspect_label'] for u in units)} }")

    # ---------- 3. 实时预报 ----------
    modes: dict[str, dict] = {}
    fpacks, lpacks = [], []
    pack_index: dict[str, int] = {}
    fdf, fmeta = fetch_weather.fetch_forecast(days=6, refresh=args.refresh)
    ref_elev = float(fmeta.get("model_elevation_m") or np.median(dem_pack["dem"]))
    fdf.attrs["ref_elev"] = ref_elev
    today = str(pd.Timestamp.now(tz=C.TIMEZONE).normalize().date())
    fmeta.update(tag="forecast", label=f"实时预报（{today} 起 {horizon*24} 小时）",
                 note="Open-Meteo 区域预报场 + 10 m 地形降尺度")
    m, fp, lp = build_mode(terrain, fdf, fmeta, units, cells, today, horizon, mc_draws)
    modes["forecast"] = m
    fpacks.append(fp)
    lpacks.append(lp)
    pack_index["forecast"] = len(fpacks) - 1
    log(f"实时预报完成：{m['summary']['n_high']} 高 / {m['summary']['n_mid']} 中 / "
        f"{m['summary']['n_low']} 低")

    # ---------- 4. 历史情景回放 ----------
    for key, sc in C.SCENARIOS.items():
        d = pd.Timestamp(sc["date"])
        start = (d - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        end = (d + pd.Timedelta(days=horizon - 1)).strftime("%Y-%m-%d")
        try:
            sdf, smeta = fetch_weather.fetch_archive(start, end, f"scenario_{key}",
                                                     refresh=args.refresh)
            sdf.attrs["ref_elev"] = float(smeta.get("model_elevation_m") or ref_elev)
            smeta.update(tag=key, label=sc["label"], note=sc["note"])
            mm, ff, ll = build_mode(terrain, sdf, smeta, units, cells,
                                    sc["date"], horizon, mc_draws)
            mm["window"] = f"{sc['date']} → {sdf.index[-1].date()}"
            modes[key] = mm
            fpacks.append(ff)
            lpacks.append(ll)
            pack_index[key] = len(fpacks) - 1
            log(f"情景回放 {key}（{sc['label']}）完成："
                f"{mm['summary']['n_high']} 高 / {mm['summary']['n_mid']} 中")
        except Exception as e:  # noqa: BLE001
            log(f"情景 {key} 失败：{e}")

    if not modes:
        raise SystemExit("没有任何模式成功构建，请检查网络。")

    # ---------- 5. 代理模型（弱监督蒸馏） ----------
    trained = None
    try:
        trained = risk_model.train_surrogate(fpacks, lpacks)
        log(f"代理模型训练完成：样本 {trained['metrics']['n_samples']}，"
            f"对照弱标签准确率 {trained['metrics']['test_accuracy']}，"
            f"宏平均 F1 {trained['metrics']['test_macro_f1']}")
        top = trained["feature_importance"][:5]
        log("  主要驱动特征：" + "、".join(f"{n}({v:.2f})" for n, v in top))
        for key, idx in pack_index.items():
            pred = risk_model.predict_surrogate(trained["model"], fpacks[idx])
            modes[key]["surrogate"] = {
                "agreement_with_physics": round(
                    float((pred["cls"] == lpacks[idx]["cls"]).mean()), 3),
                "high_prob_mean": round(float(pred["proba"][:, :, 2][
                    pred["cls"] >= 0].mean()), 3),
            }
    except Exception as e:  # noqa: BLE001
        log(f"代理模型训练失败（不影响主流程）：{e}")

    # ---------- 6. 历史回溯 ----------
    bt = {}
    if not args.skip_backtest:
        years = fetch_weather.fetch_backtest_years(refresh=args.refresh)
        bt = backtest.run_backtest(years)
        json.dump(risk_model.json_safe(bt), (C.OUT / "backtest.json").open("w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    # ---------- 7. 选定默认展示模式 ----------
    # 默认模式要"能说明问题"：优先选择风险等级分布有梯度的情景
    # （高/中/低三档都出现 = 最能体现地块级差异），其次才看高风险数量。
    def info_key(k: str):
        s = modes[k]["summary"]
        mixed = min(s["n_high"], 1) + min(s["n_mid"], 1) + min(s["n_low"], 1)
        return (mixed, s["n_high"], s["n_mid"])

    default_mode = max(modes, key=info_key)
    log(f"默认展示模式：{default_mode}（{modes[default_mode]['label']}）"
        f"｜模式信息量排序 { {k: info_key(k) for k in modes} }")

    # ---------- 8. 矢量与管理单元统计落盘 ----------
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [{
            "type": "Feature",
            "properties": {k: v for k, v in u.items() if k != "polygon"},
            "geometry": {"type": "Polygon", "coordinates": [u["polygon"] + [u["polygon"][0]]]}
            if u["polygon"] else None,
        } for u in units],
    }
    (C.OUT / "plot_units.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False, default=str), encoding="utf-8")

    # 供 GEE 采样的地块质心表（scripts/gee_sentinel2_export.js 使用）
    cent = ["unit_id,lat,lon,area_ha,aspect_mean_deg,slope_mean_deg,elev_mean_m"]
    for u in units:
        cent.append(f"{u['unit_id']},{u['center_lat']},{u['center_lon']},{u['area_ha']},"
                    f"{u['aspect_mean_deg']},{u['slope_mean_deg']},{u['elev_mean_m']}")
    (C.OUT / "plot_units_centroids.csv").write_text("\n".join(cent), encoding="utf-8")

    # ---------- 9. 组装数据包 ----------
    pack = {
        "generated_at": pd.Timestamp.now(tz=C.TIMEZONE).strftime("%Y-%m-%d %H:%M"),
        "site": C.SITE,
        "grid": {"res_m": C.GRID_RES_M, "extent": terrain["extent"],
                 "shape": list(terrain["dem"].shape),
                 "dem_stats": terrain["stats"]},
        "units": units,
        "modes": modes,
        "default_mode": default_mode,
        "backtest": bt,
        "ml": {"metrics": trained["metrics"], "feature_importance": trained["feature_importance"],
               "rs_features_available": risk_model.load_rs_indices() is not None}
        if trained else {"metrics": None},
        "data_sources": [
            {"name": "区域气象预报/再分析", "provider": "Open-Meteo（ICON/ECMWF、ERA5-Land 驱动）",
             "use": "区域强迫场、积温、蒸散、土壤水分初值", "access": "免密钥 REST API"},
            {"name": "高分辨率 DEM", "provider": "Mapzen / AWS Open Data Terrarium 瓦片",
             "use": "坡度、坡向、地形位置、汇流、地平线遮蔽、天空可视因子",
             "access": "免密钥公开数据集（纯高程，无行政边界）"},
            {"name": "Sentinel-2 多光谱", "provider": "Google Earth Engine（S2_SR_HARMONIZED）",
             "use": "NDVI/NDRE/NDWI 长势与胁迫信号（接口已就绪，需 GEE 账号）",
             "access": "scripts/gee_sentinel2_export.js"},
            {"name": "茶园地块边界", "provider": "2022 年全国 10 m 分辨率茶园分布数据集",
             "use": "把『候选管理单元』替换为真实地块边界（当前用地形适宜性聚类代替）",
             "access": "国家地球系统科学数据中心，待下载"},
        ],
        "disclaimer": C.DISCLAIMER,
        "reproducibility": {
            "mc_draws": mc_draws,
            "mc_seed_rule": "seed = crc32('meijiawu|<tag>') % 100000（与进程无关，可复现）",
            "random_state": C.ML["random_state"],
            "note": "风险分级来自蒙特卡洛参数集合投票，抽样种子固定；"
                    "数据源为实时接口，重新抓取会引入天气本身的变化。",
        },
        "limits": [
            "地块边界为地形适宜性聚类得到的候选管理单元，尚未与 10 m 茶园分布数据集叠加校正；",
            "微气候系数（k_rad、k_drain、直减率）为文献先验值，尚未用茶园小气候站观测标定；",
            "代理模型的精度是相对物理内核弱标签的一致性，不是相对田间真实受害记录；",
            "遥感胁迫信号接口已就绪但当前运行未接入（缺 GEE 账号与影像导出流程）；",
            "灌溉建议的时长基于水量平衡估算，需结合实地灌溉设备流量二次校准。",
        ],
    }
    (C.OUT / "dashboard_data.json").write_text(
        json.dumps(risk_model.json_safe(pack), ensure_ascii=False), encoding="utf-8")
    log(f"数据包已写入 outputs/dashboard_data.json "
        f"（{round((C.OUT / 'dashboard_data.json').stat().st_size / 1e6, 2)} MB）")

    # ---------- 10. 仪表盘与报告 ----------
    html = build_dashboard.render_html(pack)
    (C.APP / "dashboard.html").write_text(html, encoding="utf-8")
    log(f"仪表盘已写入 app/dashboard.html（{round(len(html) / 1e6, 2)} MB）")
    build_dashboard.write_report(pack, C.OUT / "risk_report.md")
    log("分析报告已写入 outputs/risk_report.md")


if __name__ == "__main__":
    main()
