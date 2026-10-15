"""Streamlit 版交互式 Demo（等价于 app/dashboard.html，但支持现场调参）。

    pip install streamlit
    streamlit run app/streamlit_app.py

注意：为满足地图合规要求，本页**不使用** OSM / Mapbox 等底图，
风险网格使用自绘栅格图；需要底图时请使用 app/dashboard.html 中的腾讯位置服务面板。
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "outputs" / "dashboard_data.json"

st.set_page_config(page_title="一垄一候 · 茶园微气候风险预警", layout="wide")

if not DATA.exists():
    st.error("未找到 outputs/dashboard_data.json，请先运行 `python run_pipeline.py`。")
    st.stop()

pack = json.loads(DATA.read_text(encoding="utf-8"))
st.title("一垄一候 · 梅家坞茶园地块级微气候风险预警与行动决策助手")
st.caption(f"{pack['site']['village']}｜网格 {pack['grid']['res_m']} m"
           f"｜生成于 {pack['generated_at']}")

# ---------------- 侧栏：模式与地块 ----------------
mode_keys = list(pack["modes"].keys())
labels = [pack["modes"][k]["label"] for k in mode_keys]
default = mode_keys.index(pack["default_mode"]) if pack["default_mode"] in mode_keys else 0
st.sidebar.header("情景")
mk = st.sidebar.radio("选择分析模式", mode_keys, index=default,
                      format_func=lambda k: pack["modes"][k]["label"])
m = pack["modes"][mk]
st.sidebar.caption(m.get("note", ""))
st.sidebar.caption(f"窗口：{m['window']}")

layer = st.sidebar.selectbox("地图图层", ["risk", "deviation", "moisture"],
                             format_func=lambda x: {"risk": "风险分级",
                                                    "deviation": "微气候温差",
                                                    "moisture": "土壤有效水"}[x])

# ---------------- 结论区 ----------------
s, L = m["summary"], m["layer_stats"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("高风险地块", s["n_high"], f"共 {s['n_units']} 个单元")
c2.metric("中等风险", s["n_mid"])
c3.metric("阳坡 − 阴坡", f"{L['sunny_minus_shady']} ℃" if L["sunny_minus_shady"] else "—")
c4.metric("区域最高温", f"{L['region_tmax_peak']} ℃")
c5.metric("地块级最高温", f"{L['tmax_peak_max']} ℃")

st.info(s["text"])
_bs = L.get("blindspot") or {}
if _bs:
    _sil = (f"区域最高温未达 {_bs['mild_threshold_c']:.0f} ℃ 的 {_bs['silent_days']} 天里，"
            f"仍有 {_bs['silent_hot_days']} 天出现越阈地块（最热日受影响像元占 {_bs['hot_frac_max_pct']:.1f}%）"
            if _bs["silent_days"] else
            f"窗口内区域气象场本身已越 {_bs['mild_threshold_c']:.0f} ℃，该口径下无盲区")
    st.caption(f"统一阈值盲区：地块级最高温 {_bs['cell_peak_c']} ℃ 比区域气象场 "
               f"{_bs['region_peak_c']} ℃ 高 {_bs['exceedance_c']} ℃；{_sil}。"
               f"收紧到度时累积起点 {_bs['onset_threshold_c']:.0f} ℃ 时，区域未越阈的 "
               f"{_bs['onset_silent_days']} 天里最热日已有 {_bs['onset_hot_frac_max_pct']:.1f}% 的像元越阈"
               f"（坡向对比基准：{L['comparison_basis']}）")
else:
    st.caption(f"统一阈值盲区：区域最高温未达 35 ℃ 时，仍有 "
               f"{L['region_underestimates_cells_frac']*100:.1f}% 的茶园像元微气候已达热害起点"
               f"（坡向对比基准：{L['comparison_basis']}）")

# ---------------- 图与建议 ----------------
left, right = st.columns([1.3, 1])
with left:
    st.subheader("10 m 微气候风险网格")
    st.image(m["images"][layer], use_container_width=True)
    st.caption({"risk": "红=高风险，黄=中等，绿=低；灰化区为地形不适宜",
                "deviation": "相对全域均值的日最高气温偏差（℃）",
                "moisture": "窗口内土壤相对有效水最低值"}[layer])

    st.subheader("代表地块 72 小时微气候过程")
    uids = [a["unit_id"] for a in m["advices"]]
    sel = st.selectbox("地块", uids, index=0)
    se = m["series"].get(sel, {})
    if se.get("t"):
        import pandas as pd
        df = pd.DataFrame({
            "气温(℃)": se["temp"], "土壤有效水": se["sm_rel"],
            "饱和水汽压差(kPa)": se["vpd"],
        }, index=se["t"])
        st.line_chart(df, height=280)
        st.caption("红色阴影区为气温 ≥35 ℃ 时段（热害起点）")

with right:
    st.subheader("地块预警与行动建议")
    for a in m["advices"]:
        with st.expander(f"#{a['priority']} {a['unit_id']}｜{a['risk_level']}风险｜{a['dominant']}",
                         expanded=a["priority"] <= 2):
            st.write(a["headline"])
            st.caption(f"{a['area_ha']} ha｜海拔 {a['elev_mean_m']} m｜坡度 {a['slope_mean_deg']}°"
                       f"｜置信度 {a['confidence']*100:.0f}%")
            for e in a["evidence"]:
                st.caption("· " + e)
            for i, x in enumerate(a["actions"], 1):
                st.markdown(f"**{i}. {x['action']}**　`{x['when']}`")
                st.caption(f"做法：{x['how']}")
                st.caption(f"理由：{x['why']}")
    st.warning(m["patrol"]["action"] + "：" +
               "；".join(f"{o['rank']}.{o['unit_id']}({o['aspect']})" for o in m["patrol"]["order"]))

# ---------------- 回溯与模型 ----------------
if pack.get("backtest"):
    st.subheader("历史灾情回溯验证（真实再分析数据）")
    rows = []
    for y, r in sorted(pack["backtest"].items()):
        k = r["metrics"]
        rows.append({"年份": y, "最高气温": k["tmax_peak"], "≥35℃日数": k["days_ge35"],
                     "≥38℃日数": k["days_ge38"], "夏季降水/mm": k["precip_total_mm"],
                     "净亏缺/mm": k["net_deficit_mm"], "高风险日": r["high_risk_days"],
                     "预警提前量/天": r["lead_days_heat"], "漏报率": r["false_alarm_rate"]})
    st.dataframe(rows, use_container_width=True)

if pack["ml"].get("metrics"):
    st.subheader("AI 层")
    ml = pack["ml"]["metrics"]
    st.write(f"弱监督样本 {ml['n_samples']}，对照弱标签准确率 {ml['test_accuracy']*100:.1f}%，"
             f"宏平均 F1 {ml['test_macro_f1']}，参数集合一致度 "
             f"{ml['mean_ensemble_agreement']*100:.0f}%")
    st.caption(ml["labels_note"])
    st.dataframe([{"特征": n, "重要性": v} for n, v in pack["ml"]["feature_importance"][:12]],
                 use_container_width=True)

st.subheader("已知局限")
for l in pack["limits"]:
    st.caption("· " + l)
st.caption(pack["disclaimer"])
