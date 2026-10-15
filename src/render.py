"""渲染层：把 10 m 网格场渲染成仪表盘可直接使用的底图 PNG（base64 内嵌）。

只用 Pillow，不依赖任何地图服务——风险网格是**自绘数据可视化**，
底图的行政/地理上下文由合规地图（腾讯位置服务）在另一个面板提供。
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

CMAPS = {
    "risk": [(0.00, (46, 139, 87)), (0.35, (140, 198, 63)), (0.55, (232, 185, 59)),
             (0.75, (233, 126, 45)), (1.00, (205, 42, 40))],
    "temp": [(0.00, (49, 84, 149)), (0.35, (120, 178, 214)), (0.55, (245, 245, 235)),
             (0.75, (240, 158, 96)), (1.00, (178, 34, 34))],
    "moist": [(0.00, (150, 90, 40)), (0.30, (214, 170, 90)), (0.55, (176, 214, 160)),
              (0.80, (72, 152, 190)), (1.00, (24, 74, 140))],
    "grey": [(0.0, (30, 34, 40)), (1.0, (235, 240, 245))],
}


def _apply_cmap(x: np.ndarray, name: str) -> np.ndarray:
    """x ∈ [0,1] → RGB。"""
    stops = CMAPS[name]
    xs = np.clip(x, 0, 1)
    out = np.zeros(x.shape + (3,), dtype=np.float64)
    for i in range(len(stops) - 1):
        x0, c0 = stops[i]
        x1, c1 = stops[i + 1]
        m = (xs >= x0) & (xs <= x1)
        if not m.any():
            continue
        w = ((xs[m] - x0) / max(x1 - x0, 1e-9))[:, None]
        out[m] = np.array(c0)[None, :] * (1 - w) + np.array(c1)[None, :] * w
    return out


def base_map(terrain: dict, flip: bool = True) -> np.ndarray:
    """底图：晕渲 × 高程着色，北在上。"""
    from .terrain import hillshade
    dem = terrain["dem"]
    dem_n = np.clip((dem - dem.min()) / max(dem.max() - dem.min(), 1e-6), 0, 1)
    tint = _apply_cmap(dem_n, "temp") * 0.35 + _apply_cmap(dem_n * 0.6 + 0.2, "risk") * 0.15
    tint = tint * 0.5 + np.array([60, 62, 66])[None, None, :] * 0.5
    hs = hillshade(dem, C_RES()) / 255.0
    base = tint * (0.55 + 0.75 * hs[:, :, None])
    base = np.clip(base, 0, 255).astype(np.uint8)
    return base[::-1] if flip else base


def C_RES() -> float:
    import config as C
    return C.GRID_RES_M


def overlay(field: np.ndarray, vmin: float, vmax: float, cmap: str = "risk",
            alpha_max: float = 0.8, mask: np.ndarray | None = None,
            base: np.ndarray | None = None, flip: bool = True) -> np.ndarray:
    """把场叠加到底图上，返回 uint8 RGB。"""
    x = (np.clip(field, vmin, vmax) - vmin) / max(vmax - vmin, 1e-9)
    rgb = _apply_cmap(x, cmap)
    if base is None:
        base = np.full(field.shape + (3,), 32.0)
    a = np.full(field.shape, alpha_max, dtype=np.float64)
    if mask is not None:
        a = np.where(mask, a, a * 0.15)
    f = field.astype(float)
    valid = np.isfinite(f)
    img = np.where(valid[:, :, None], rgb * a[:, :, None] + base * (1 - a[:, :, None]), base)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if flip:
        img = img[::-1]
    return img


def draw_units(img: np.ndarray, terrain: dict, units: list[dict],
               color=(255, 236, 150), width: int = 2) -> np.ndarray:
    """在图像上勾画管理单元边界（经纬度 → 像素）。"""
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil)
    ext = terrain["extent"]
    ny, nx = img.shape[:2]
    for u in units:
        pts = []
        for lon, lat in u.get("polygon", []):
            px = (lon - ext["lon_min"]) / max(ext["lon_max"] - ext["lon_min"], 1e-9) * (nx - 1)
            py = (ext["lat_max"] - lat) / max(ext["lat_max"] - ext["lat_min"], 1e-9) * (ny - 1)
            pts.append((px, py))
        if len(pts) >= 3:
            dr.polygon(pts, outline=color, width=width)
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            r = 5
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))
    return np.asarray(pil)


def to_png_b64(img: np.ndarray, scale: int = 2, colors: int = 128) -> str:
    """放大 + 调色板量化 + base64（内嵌进 HTML，无需网络）。"""
    if scale != 1:
        img = np.asarray(Image.fromarray(img).resize(
            (img.shape[1] * scale, img.shape[0] * scale), Image.LANCZOS))
    pil = Image.fromarray(img).convert("RGB")
    if colors:
        pil = pil.quantize(colors=colors, method=Image.MEDIANCUT)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def grid_series_png(series: dict, key: str, color, w: int = 640, h: int = 160) -> str:
    """轻量 sparkline（部分环境不便用 JS 时作为备选）。"""
    v = np.array(series[key], dtype=float)
    if v.size == 0:
        return ""
    img = Image.new("RGB", (w, h), (26, 29, 34))
    dr = ImageDraw.Draw(img)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if hi - lo < 1e-6:
        hi = lo + 1
    pts = [((i / max(len(v) - 1, 1)) * (w - 8) + 4, h - 6 - (val - lo) / (hi - lo) * (h - 16))
           for i, val in enumerate(v)]
    dr.line(pts, fill=color, width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
