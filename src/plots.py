"""管理单元（地块）提取：基于地形适宜性阈值 + 连通域聚类 + 面积细分。

重要说明
--------
离线环境下无法取得「2022 年全国 10 m 分辨率茶园分布数据集」，
因此本模块用**地形适宜性掩膜 + 连通域聚类**生成"候选管理单元"。
其作用是：让地块级别的预警有明确的空间载体；拿到茶园分布数据后，
把 `tea_suitability_mask()` 换成真实地块栅格即可，下游全部代码不变
（见 scripts/gee_tea_mask.js 与 docs/数据来源与待补清单.md）。

适宜性判据（可配置）
  海拔 25–230 m（避开河谷平地与水田、山顶林地）
  坡度 4°–25°（成龄梯田茶园区间）
  TWI ≤ 8.5（排除汇水/易涝处）
  汇流面积 ≤ 25000 m²（排除溪沟与道路汇水线）
面积过大的连通域用 KMeans 按空间位置细分为约 6 ha 的管理单元，
最终保留面积最大且空间分散的若干单元作为预警载体。
"""
from __future__ import annotations

import numpy as np

import config as C
from .terrain import aspect_label


def tea_suitability_mask(terrain: dict) -> np.ndarray:
    dem, slope, w, acc = terrain["dem"], terrain["slope"], terrain["twi"], terrain["flow_acc"]
    return (dem >= 25) & (dem <= 230) & (slope >= 4.0) & (slope <= 25.0) \
        & (w <= 8.5) & (acc <= 25000.0)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        return lab, int(n)
    except Exception:                                    # noqa: BLE001  scipy 缺失时降级
        lab = np.zeros(mask.shape, dtype=np.int32)
        ny, nx = mask.shape
        cur = 0
        stack: list[tuple[int, int]] = []
        for i in range(ny):
            for j in range(nx):
                if mask[i, j] and lab[i, j] == 0:
                    cur += 1
                    stack.append((i, j)); lab[i, j] = cur
                    while stack:
                        y, x = stack.pop()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                yy, xx = y + dy, x + dx
                                if 0 <= yy < ny and 0 <= xx < nx and mask[yy, xx] and lab[yy, xx] == 0:
                                    lab[yy, xx] = cur; stack.append((yy, xx))
        return lab, cur


def extract_units(terrain: dict, min_area_m2: float = 3000.0, max_units: int = 14,
                  target_unit_m2: float = 60000.0, min_sep_m: float = 140.0) -> list[dict]:
    """返回管理单元列表（含几何、地形统计、代表单元索引）。"""
    res = C.GRID_RES_M
    cell_area = res * res
    mask = tea_suitability_mask(terrain)
    lab, n = _label_components(mask)

    # ---- 连通域 → 单元（过大的连通域按空间位置细分）----
    cands: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        if len(ys) * cell_area < min_area_m2:
            continue
        area = len(ys) * cell_area
        k_split = max(1, int(np.ceil(area / target_unit_m2)))
        if k_split == 1:
            cands.append((ys, xs))
            continue
        try:
            from sklearn.cluster import KMeans
            X = np.column_stack([ys * res, xs * res]).astype(np.float32)
            labels = KMeans(n_clusters=k_split, n_init=4,
                            random_state=0).fit_predict(X)
            for c in range(k_split):
                m = labels == c
                if m.sum() * cell_area >= min_area_m2:
                    cands.append((ys[m], xs[m]))
        except Exception:                                # noqa: BLE001
            cands.append((ys, xs))

    # ---- 空间分散地挑选最大单元 ----
    cands.sort(key=lambda t: -len(t[0]))
    picked: list[tuple[np.ndarray, np.ndarray]] = []
    centers: list[tuple[float, float]] = []
    for ys, xs in cands:
        cy, cx = float(ys.mean()) * res, float(xs.mean()) * res
        if all(np.hypot(cy - a, cx - b) > min_sep_m for a, b in centers):
            picked.append((ys, xs)); centers.append((cy, cx))
        if len(picked) >= max_units:
            break

    ulab = np.zeros(mask.shape, dtype=np.int32)
    for i, (ys, xs) in enumerate(picked, start=1):
        ulab[ys, xs] = i

    out = []
    for i, (ys, xs) in enumerate(picked, start=1):
        sl = terrain["slope"][ys, xs]
        elev = terrain["dem"][ys, xs]
        asp = terrain["aspect"][ys, xs]
        tpi_ = terrain["tpi"][ys, xs]
        sr = np.radians(asp)                             # 单位向量平均，避免 0/360 环绕
        asp_mean = float(np.degrees(np.arctan2(float(np.sin(sr).mean()),
                                               float(np.cos(sr).mean()))) % 360.0)
        # 代表单元：地形因子最接近单元均值的一点
        d = np.abs(tpi_ - tpi_.mean()) + np.abs(elev - elev.mean()) / 10.0
        r = int(np.argmin(d))
        out.append({
            "unit_id": f"MDW-{i:02d}",
            "area_ha": round(len(ys) * cell_area / 10000.0, 2),
            "n_cells": int(len(ys)),
            "cell_index": [int(ys[r]), int(xs[r])],
            "center_lat": round(float(terrain["lat"][int(ys.mean())]), 6),
            "center_lon": round(float(terrain["lon"][int(xs.mean())]), 6),
            "elev_mean_m": round(float(elev.mean()), 1),
            "elev_range_m": [round(float(elev.min()), 1), round(float(elev.max()), 1)],
            "slope_mean_deg": round(float(sl.mean()), 1),
            "slope_max_deg": round(float(sl.max()), 1),
            "aspect_mean_deg": round(asp_mean, 1),
            "aspect_label": aspect_label(asp_mean),
            "tpi_mean": round(float(tpi_.mean()), 2),
            "svf_mean": round(float(terrain["svf"][ys, xs].mean()), 3),
            "frac_sunny": round(float(((asp >= 135) & (asp < 225)).mean()), 2),
            "polygon": _outline(terrain, ulab, i),
        })
    return out


def _outline(terrain: dict, ulab: np.ndarray, k: int, max_pts: int = 200) -> list[list[float]]:
    """把栅格连通域转成简化多边形（经纬度）。

    做法：取每行左右端点与每列上下端点组成边界点集，按质心角度排序成环后抽稀。
    精度足以在地图上示意地块范围，不用于面积计算（面积以像元计数为准）。
    """
    import math
    ys, xs = np.where(ulab == k)
    if len(ys) == 0:
        return []
    pts: set[tuple[int, int]] = set()
    for i in range(ys.min(), ys.max() + 1):
        row = np.where(ulab[i] == k)[0]
        if len(row):
            pts.add((i, int(row.min()))); pts.add((i, int(row.max())))
    for j in range(xs.min(), xs.max() + 1):
        col = np.where(ulab[:, j] == k)[0]
        if len(col):
            pts.add((int(col.min()), j)); pts.add((int(col.max()), j))
    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    plist = sorted(pts, key=lambda p: math.atan2(p[0] - cy, p[1] - cx))
    if len(plist) > max_pts:
        step = len(plist) / max_pts
        plist = [plist[int(i * step)] for i in range(max_pts)]
    return [[round(float(terrain["lon"][j]), 6), round(float(terrain["lat"][i]), 6)]
            for i, j in plist]
