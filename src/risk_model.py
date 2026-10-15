"""机器学习层：地形降尺度残差模型 + 风险分类代理模型。

本模块承担三件事
----------------
1. **参数不确定性传播**：对物理内核的关键系数做蒙特卡洛抽样，得到每个网格单元
   的风险分数分布 → 提供"预警置信度"，避免把参数先验当成真值。
2. **弱监督风险分类**（风险分级代理模型）：以物理内核的风险等级为弱标签，
   用随机森林学习「可观测特征 → 风险等级」的映射。其价值在于：
   - 给出可校准的概率输出与特征重要性（哪些信号真正驱动风险）；
   - 提供一个**可以换成真实田间标签直接再训练**的接口（换 y 即可，特征不变）。
3. **遥感特征接口**：`load_rs_indices()` 会读取 Sentinel-2 导出的地块级植被指数
   （NDVI/NDRE/NDWI），文件不存在时自动退化为不含遥感特征的基线特征集。

诚实声明：本模块的精度评估是**相对弱标签**的（模型一致性），
不是相对田间真实受害记录的。真实精度需在拿到小气候站/农情调查数据后重算。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config as C
from . import microclimate, stress

BASE_FEATURES = [
    "elev_m", "slope_deg", "aspect_sin", "aspect_cos", "tpi", "twi", "svf",
    "log_flow_acc", "sw_exposure",
    "tmax_region", "tmin_region", "precip_sum", "et0_sum", "vpd_max_region",
    "rh_min_region", "dry_streak", "sm_surface",
]
RS_FEATURES = ["ndvi_mean", "ndre_mean", "ndwi_mean"]


# --------------------------------------------------------------------------
# 参数不确定性传播
# --------------------------------------------------------------------------
def mc_risk_ensemble(terrain: dict, forcing, horizon: int,
                     draws: int | None = None, seed: int = 7) -> dict:
    """对物理内核参数做蒙特卡洛抽样，返回风险分数分布。"""
    n = draws or C.ML["mc_draws"]
    rng = np.random.default_rng(seed)
    amp = C.ML["mc_perturb"]
    scores, classes = [], []
    used = []
    for k in range(n):
        jitter = {
            "k_rad": C.DOWNSCALE["k_rad"] * (1 + rng.normal(0, amp)),
            "k_drain": C.DOWNSCALE["k_drain"] * (1 + rng.normal(0, amp)),
            "lapse_rate": C.DOWNSCALE["lapse_rate"] * (1 + rng.normal(0, amp * 0.5)),
        }
        tea = dict(C.TEA)
        tea["heat_mild"] = C.TEA["heat_mild"] + rng.normal(0, 0.6)
        tea["sm_rel_stress"] = C.TEA["sm_rel_stress"] + rng.normal(0, 0.03)
        res = microclimate.run_downscaling(terrain, forcing, params=jitter,
                                           horizon_days=horizon)
        a = stress.assess(res["daily"], horizon)
        scores.append(a["comp"]["score"])
        classes.append(a["comp"]["cls"].astype(np.int8))
        used.append(jitter)
    S = np.stack(scores)
    Cc = np.stack(classes)
    mean = S.mean(axis=0)
    modal = np.array([np.bincount(Cc[:, i, j], minlength=3).argmax()
                      for i, j in np.ndindex(mean.shape)]).reshape(mean.shape).astype(np.int8) \
        if mean.size <= 200_000 else np.median(Cc, axis=0).astype(np.int8)
    agree = np.array([np.bincount(Cc[:, i, j], minlength=3).max() / len(Cc)
                      for i, j in np.ndindex(mean.shape)]).reshape(mean.shape)
    if mean.size > 200_000:                       # 大网格用向量化投票
        modal = (Cc == np.array([np.bincount(Cc[:, i, j], minlength=3).argmax()
                                 for i, j in np.ndindex(mean.shape)]).reshape(Cc[0].shape))
    return dict(score=mean.astype(np.float32), std=S.std(axis=0).astype(np.float32),
                cls=modal, agree=agree.astype(np.float32), draws=n,
                params=used)


# --------------------------------------------------------------------------
# 特征工程
# --------------------------------------------------------------------------
def load_rs_indices(path: Path | None = None) -> dict | None:
    """读取 Sentinel-2 地块级植被指数（需先运行 scripts/gee_sentinel2_export.js）。"""
    p = path or (C.PROC / "rs_indices.csv")
    if not p.exists():
        return None
    import pandas as pd
    df = pd.read_csv(p)
    return {c: df[[ "lat", "lon", c]].values for c in RS_FEATURES if c in df.columns}


def feature_stack(terrain: dict, daily: dict, horizon: int,
                  forcing=None) -> dict:
    """构建特征立方体：所有特征都来自"运行时可获得"的信号。"""
    d = daily
    sw_exposure = d["sw_sum"][:horizon].mean(axis=0) / 1000.0     # kW·h·m⁻²·d⁻¹
    feats = {
        "elev_m": terrain["dem"],
        "slope_deg": terrain["slope"],
        "aspect_sin": terrain["aspect_sin"],
        "aspect_cos": terrain["aspect_cos"],
        "tpi": terrain["tpi"],
        "twi": terrain["twi"],
        "svf": terrain["svf"],
        "log_flow_acc": np.log10(np.maximum(terrain["flow_acc"], 1.0)),
        "sw_exposure": sw_exposure,
    }
    ny, nx = d["tmax"].shape[1:]
    tmax_r = float(np.max(d["tmax_region"][:horizon]))
    tmin_r = float(np.min(daily_min_region(daily, horizon)))
    precip = float(np.sum(d["precip_mm"][:horizon]))
    et0 = float(np.sum(d["et0_mm"][:horizon]))
    rh_min = float(np.min(d["rh_min"][:horizon]))
    vpd_max = float(np.max(d["vpd_max"][:horizon]))
    dry = float(np.sum(np.asarray(d["precip_mm"][:horizon]) < C.TEA["dry_spell_mm"]))
    if forcing is not None and len(forcing):
        sm_surface = float(np.nanmedian(forcing["soil_moisture_3_to_9cm"].values))
    else:
        sm_surface = 0.25
    scalars = dict(tmax_region=tmax_r, tmin_region=tmin_r, precip_sum=precip,
                   et0_sum=et0, rh_min_region=rh_min, vpd_max_region=vpd_max,
                   dry_streak=dry, sm_surface=sm_surface)
    for k, v in scalars.items():
        feats[k] = np.full((ny, nx), v, dtype=np.float32)
    names = list(BASE_FEATURES)
    rs = load_rs_indices()
    if rs:
        for k in RS_FEATURES:
            f = np.full((ny, nx), np.nan, dtype=np.float32)
            if k in rs:
                arr = rs[k]
                iy = np.clip(((arr[:, 0] - terrain["extent"]["lat_min"])
                              / (terrain["extent"]["lat_max"] - terrain["extent"]["lat_min"])
                              * (ny - 1)).astype(int), 0, ny - 1)
                jx = np.clip(((arr[:, 1] - terrain["extent"]["lon_min"])
                              / (terrain["extent"]["lon_max"] - terrain["extent"]["lon_min"])
                              * (nx - 1)).astype(int), 0, nx - 1)
                f[iy, jx] = arr[:, 2]
            feats[k] = f
            names.append(k)
    return {"cube": feats, "names": names, "scalars": scalars}


def daily_min_region(daily: dict, horizon: int) -> np.ndarray:
    """区域参考最低气温（由每日面场中心值近似）。"""
    if "tmin_region" in daily:
        return np.asarray(daily["tmin_region"][:horizon])
    ny, nx = daily["tmin"].shape[1:]
    return np.array([daily["tmin"][k][ny // 2, nx // 2] for k in range(horizon)])


# --------------------------------------------------------------------------
# 代理模型（弱监督蒸馏）
# --------------------------------------------------------------------------
def _matrix(cube: dict, names: list[str], mask: np.ndarray | None = None,
            max_n: int = 24000, seed: int = 1):
    X = np.stack([np.asarray(cube[n], dtype=np.float32).ravel() for n in names], axis=1)
    ok = np.isfinite(X).all(axis=1)
    if mask is not None:
        ok &= mask.ravel()
    idx = np.where(ok)[0]
    if len(idx) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, max_n, replace=False)
    return X[idx], idx


def train_surrogate(feature_packs: list[dict], label_packs: list[dict]) -> dict:
    """在多个情景上池化训练（保证标签类别均衡、跨条件泛化）。"""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    names = feature_packs[0]["names"]
    Xs, ys, sc, wt = [], [], [], []
    for fp, lp in zip(feature_packs, label_packs):
        X, idx = _matrix(fp["cube"], names, fp.get("mask"))
        y = lp["cls"].ravel()[idx]
        s = lp["score"].ravel()[idx]
        Xs.append(X); ys.append(y); sc.append(s); wt.append(lp["agree"].ravel()[idx])
    X = np.vstack(Xs); y = np.concatenate(ys); s = np.concatenate(sc); w = np.concatenate(wt)

    Xtr, Xte, ytr, yte, str_, ste = train_test_split(
        X, y, s, test_size=C.ML["test_size"],
        random_state=C.ML["random_state"], stratify=y)
    clf = RandomForestClassifier(n_estimators=C.ML["n_estimators"], max_depth=C.ML["max_depth"],
                                 min_samples_leaf=C.ML["min_samples_leaf"], n_jobs=-1,
                                 random_state=C.ML["random_state"], class_weight="balanced")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)
    reg = RandomForestRegressor(n_estimators=C.ML["n_estimators"] // 2, max_depth=C.ML["max_depth"],
                                min_samples_leaf=C.ML["min_samples_leaf"], n_jobs=-1,
                                random_state=C.ML["random_state"])
    reg.fit(Xtr, str_)
    from sklearn.metrics import mean_absolute_error, r2_score
    imp = sorted(zip(names, clf.feature_importances_), key=lambda kv: -kv[1])
    return {
        "model": {"clf": clf, "reg": reg, "names": names},
        "metrics": {
            "n_samples": int(len(X)),
            "class_balance": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "test_accuracy": round(float(accuracy_score(yte, pred)), 3),
            "test_macro_f1": round(float(f1_score(yte, pred, average="macro")), 3),
            "confusion_matrix": confusion_matrix(yte, pred).tolist(),
            "score_mae": round(float(mean_absolute_error(ste, reg.predict(Xte))), 4),
            "score_r2": round(float(r2_score(ste, reg.predict(Xte))), 3),
            "prob_calibration": _calibration(yte, prob),
            "labels_note": "对照标签来自物理内核（弱监督），非田间真实受害记录",
            "mean_ensemble_agreement": round(float(np.nanmean(w)), 3),
        },
        "feature_importance": [[n, round(float(v), 4)] for n, v in imp],
    }


def _calibration(yte: np.ndarray, prob: np.ndarray, bins: int = 5) -> dict:
    """高风险的可靠性曲线（预测概率 vs 实际频率），用于说明概率是否可信。"""
    p = prob[:, 2] if prob.shape[1] > 2 else np.max(prob, axis=1)
    obs = (yte == 2).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] + (1e-9 if i == bins - 1 else 0))
        if m.sum() >= 5:
            out.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                        "n": int(m.sum()),
                        "pred": round(float(p[m].mean()), 3),
                        "obs": round(float(obs[m].mean()), 3)})
    return {"bins": out, "n": int(len(yte))}


def predict_surrogate(model_pack: dict, fp: dict) -> dict:
    names = model_pack["names"]
    X, idx = _matrix(fp["cube"], names, None, max_n=10 ** 9)
    proba = model_pack["clf"].predict_proba(X)
    cls = model_pack["clf"].predict(X)
    ny, nx = fp["cube"]["elev_m"].shape
    out_cls = np.full(ny * nx, -1, dtype=np.int8)
    out_prob = np.zeros((ny * nx, 3), dtype=np.float32)
    out_cls[idx] = cls
    out_prob[idx] = proba
    return {"cls": out_cls.reshape(ny, nx), "proba": out_prob.reshape(ny, nx, 3),
            "mask": idx}


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
