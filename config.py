"""
梅家坞气候智慧茶园 · 微气候风险预警与行动决策助手
全局配置：站点、网格、物候阈值、模型参数、风险分级标准。

所有阈值均可通过命令行覆盖，便于在实地校准后替换。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"
APP = ROOT / "app"

for _p in (RAW, PROC, OUT, DOCS, APP):
    _p.mkdir(parents=True, exist_ok=True)

TIMEZONE = "Asia/Shanghai"

# --------------------------------------------------------------------------
# 站点与网格
# --------------------------------------------------------------------------
SITE = {
    "name": "梅家坞气候智慧茶园示范站",
    "village": "浙江省杭州市西湖区转塘街道梅家坞村",
    "lat": 30.1830,          # 站点参考坐标（村域中心，腾讯地图 POI: 30.174663, 120.099143）
    "lon": 120.0990,
    "elev_note": "村域海拔约 40–340 m，为西湖龙井一级保护区内群山最高处",
    "climate_note": "北亚热带季风气候，年降水量约 1600 mm，昼夜温差 8–10 ℃，白沙壤 pH 4.5–5.5",
}

# 研究区外接矩形（约 3.1 km × 2.5 km），覆盖梅家坞村域溪谷与北侧茶山
BBOX = {
    "lat_min": 30.1720,
    "lat_max": 30.2000,
    "lon_min": 120.0860,
    "lon_max": 120.1120,
}

GRID_RES_M = 10.0            # 微气候网格分辨率（米）
DEM_Z = 15                   # Terrarium 地形瓦片层级（纬度 30° 处约 3.5–4 m/像素）

# --------------------------------------------------------------------------
# 茶树物候与胁迫阈值
#   依据：茶树生长最适温 15–25 ℃；高温伤害起点约 35 ℃；
#   适宜土壤含水量为田间持水量的 70–90%（相对有效水 0.55–0.75）；
#   早春采摘期 Tmin ≤ 4 ℃ 出现霜冻害。
#   这些是"文献先验值"，正式使用时应用小气候站观测做分位数校准。
# --------------------------------------------------------------------------
TEA = {
    "T_opt_min": 15.0,
    "T_opt_max": 25.0,
    "heat_mild": 35.0,        # ℃ 轻度热害起点（日最高温）
    "heat_moderate": 38.0,    # ℃ 中度热害
    "heat_severe": 40.0,      # ℃ 重度热害
    "heat_hours_base": 32.0,  # ℃ 度时累积基准（光合/蒸腾胁迫累积）
    "frost_mild": 4.0,        # ℃ 霜冻受害起点
    "frost_severe": 0.0,      # ℃ 重度霜冻
    "sm_rel_stress": 0.55,    # 相对有效水低于此值开始水分胁迫
    "sm_rel_wilt": 0.35,      # 接近萎蔫
    "sm_rel_optimal": 0.72,   # 灌溉目标上限
    "Kc": 0.95,               # 作物系数（成龄茶园、冠层覆盖度高）
    "TAW_mm": 90.0,           # 总有效水（0.5 m 根层，砂壤偏壤）
    "root_depth_m": 0.5,
    "dry_spell_mm": 5.0,      # 单日有效降水阈值
    "dry_spell_days": 7,      # 连续无有效降水日数 → 旱象
}

# --------------------------------------------------------------------------
# 微气候降尺度参数（物理内核；残差层由 ML 学习，见 src/microclimate.py）
# --------------------------------------------------------------------------
DOWNSCALE = {
    "lapse_rate": -0.60,      # ℃/100 m 环境温度直减率
    # 近地层气温对短波辐射差异的敏感度（℃ per W·m⁻²）。
    # 采用 0.012：对应"坡面短波相差 100 W·m⁻² → 气温相差约 1.2 ℃"，
    # 落在复杂地形近地层观测给出的 1–2 ℃/100 W·m⁻² 区间内；
    # 该系数是**待标定参数**，蒙特卡洛抽样已按 ±20% 传播其不确定性。
    "k_rad": 0.012,
    "rad_cap": 6.0,           # ℃ 辐射项截断
    "k_drain": 0.018,         # ℃ / m 夜间冷池（负 TPI）降温系数
    "drain_cap": 2.0,         # ℃ 冷池降温上限
    "k_wind_ridge": 0.35,     # 山脊/开阔地风速抬升系数
    "svf_rad_weight": 0.45,   # 天空可视因子对日间辐射的折减权重
}

# --------------------------------------------------------------------------
# 风险分级标准（0 低 / 1 中 / 2 高）
# 综合风险分 = max(热害分, 旱害分, 霜冻分) + 0.25 * second_max（复合放大）
# --------------------------------------------------------------------------
RISK = {
    "low_upper": 0.35,
    "high_lower": 0.60,
    "compound_gain": 0.25,
    "horizon_days": 3,        # 预警展望天数（72 h）
    "advice_lead_hours": 36,  # 行动建议提前量
}

# 热害分：度时累积（HSU, ℃·h）
HSI_SCALE = {"low": 6.0, "high": 24.0}
# 旱害分：相对有效水亏缺 × 连续干旱日数放大
DSI_SCALE = {"low": 0.35, "high": 0.75}
# 霜冻分
FSI_SCALE = {"low": 0.5, "high": 0.9}

# --------------------------------------------------------------------------
# 机器学习层
# --------------------------------------------------------------------------
ML = {
    "n_estimators": 400,
    "max_depth": 8,
    "min_samples_leaf": 8,
    "random_state": 42,
    "mc_perturb": 0.20,       # 弱标签生成时的参数蒙特卡洛扰动幅度（相对）
    "mc_draws": 24,           # 蒙特卡洛抽样次数
    "test_size": 0.25,
}

# --------------------------------------------------------------------------
# 天气数据源
# --------------------------------------------------------------------------
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

FORECAST_HOURLY = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "wind_speed_10m", "shortwave_radiation", "direct_normal_irradiance",
    "diffuse_radiation", "soil_moisture_3_to_9cm", "soil_temperature_6cm",
    "et0_fao_evapotranspiration", "cloud_cover",
]
ARCHIVE_HOURLY = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m",
    "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
    "soil_moisture_3_to_9cm", "et0_fao_evapotranspiration",
]

# 历史回溯:夏季（6/1–9/15）
BACKTEST_YEARS = [2013, 2022, 2024, 2025, 2026]
BACKTEST_START = "06-01"
BACKTEST_END = "09-15"

# 情景回放（历史极端事件，用于 Demo 演示与灾情复盘）
SCENARIOS = {
    "2022_drought": {
        "label": "2022-08-12 极端高温干旱",
        "date": "2022-08-12",
        "note": "2022 年夏长江流域伏旱。杭州是全国高温日数累计最多的省会城市（35 ℃ 以上 29 天，『满额』）；"
                "西湖区约 5000 亩茶树顶部灼伤，以『龙井43』（无性系扦插、根系浅）为主，"
                "群体种（根深）抗性较好；导致翌年春茶大幅减产（报道称『十年来产量最低』）。",
        "ref": "人民网浙江频道 2022-08-13；中国供销合作网《春茶中国行②》",
    },
    "2013_heat": {
        "label": "2013-08-06 罕见高温干旱",
        "date": "2013-08-06",
        "note": "2013 年盛夏罕见高温干旱（浙江省称 60 年同期最严重热浪）。全省茶园受灾约 40 万亩"
                "（杭州 20 万亩、湖州 7.8 万亩、丽水 10 万亩）；夏秋茶减产超 35%、翌年春茶减产 20%–30%，"
                "西湖龙井预计减产 40%。全省农业生产直接经济损失约 17 亿元（含粮食作物，非茶园单列）。",
        "ref": "浙江省农业厅通报，人民网/新华网 2013-08-09；中国经济网 2013-08-30",
    },
    "2025_summer": {
        "label": "2025-07-20 夏季常规高温",
        "date": "2025-07-20",
        "note": "用于对比的近年常规高温情景（非极端灾年）",
        "ref": "—",
    },
}

DISCLAIMER = (
    "本系统输出为气候风险评估与辅助建议，不替代茶农的实地判断与农技部门指导。"
    "请结合田间实际观测使用；系统不以预警结果自动判定茶叶品质或价格。"
)


def dump_config(path: Path) -> None:
    """把本次运行使用的全部参数落盘，保证结果可复现、可审计。"""
    payload = {
        "site": SITE, "bbox": BBOX, "grid_res_m": GRID_RES_M, "dem_z": DEM_Z,
        "tea": TEA, "downscale": DOWNSCALE, "risk": RISK,
        "hsi_scale": HSI_SCALE, "dsi_scale": DSI_SCALE, "fsi_scale": FSI_SCALE,
        "ml": ML, "scenarios": SCENARIOS, "backtest_years": BACKTEST_YEARS,
        "disclaimer": DISCLAIMER,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
