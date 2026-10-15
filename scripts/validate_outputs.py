"""校验 dashboard.html 的完整性与合规要点（不依赖浏览器）。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# 脚本既可能在项目根，也可能在 scripts/ 下；用 run_pipeline.py 作为项目根标记向上定位。
ROOT = _HERE if (_HERE / "run_pipeline.py").exists() else _HERE.parent
HTML = ROOT / "app" / "dashboard.html"
DATA = ROOT / "outputs" / "dashboard_data.json"

ok = True


def check(label: str, cond: bool, extra: str = "") -> None:
    global ok
    print(("[PASS] " if cond else "[FAIL] ") + label + (f"  {extra}" if extra else ""))
    ok = ok and cond


html = HTML.read_text(encoding="utf-8")

# 1) JSON 数据岛能被解析（防止截断 / 转义错误）
m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
check("HTML 内含 payload 数据岛", m is not None)
pack = None
if m:
    try:
        pack = json.loads(m.group(1))
        check("payload 是合法 JSON", True, f"{len(m.group(1))/1e6:.2f} MB")
    except Exception as e:  # noqa: BLE001
        check("payload 是合法 JSON", False, str(e))

# 2) 独立数据包
try:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    check("dashboard_data.json 合法", True,
          f"modes={list(d.get('modes', {}))} default={d.get('default_mode')}")
    check("含 blindspot 新指标", all(
        "blindspot" in mm.get("layer_stats", {}) for mm in d.get("modes", {}).values()))
    check("含可复现性声明", "reproducibility" in d)
except Exception as e:  # noqa: BLE001
    check("dashboard_data.json 合法", False, str(e))

if pack:
    check("模式数 = 4", len(pack["modes"]) == 4, str(list(pack["modes"])))
    check("地块数 = 14", len(pack["units"]) == 14)
    check("风险网格图为 base64 PNG", all(
        str(mm["images"]["risk"]).startswith("data:image") for mm in pack["modes"].values()))
    check("回溯含逐年指标", bool(pack.get("backtest")) and "summary" in pack["backtest"])
    md = pack["modes"][pack["default_mode"]]
    check("默认模式有梯度（高/中/低均出现）",
          md["summary"]["n_high"] > 0 and md["summary"]["n_mid"] > 0 and md["summary"]["n_low"] > 0,
          f"高{md['summary']['n_high']}/中{md['summary']['n_mid']}/低{md['summary']['n_low']}")

# 3) 合规要点
check("底图为腾讯位置服务", "map.qq.com" in html)
check("含 WGS-84 → GCJ-02 转换", ("gcj" in html.lower() or "GCJ" in html) and "outOfChina" in html)
check("未引用第三方瓦片服务",
      not re.search(r"https?://[^\"'\s]*(?:tile|mapbox|openstreetmap|arcgis)", html, re.I))
check("含免责声明", "不替代茶农的实地判断" in html)
check("风险网格为自绘（不依赖底图瓦片）", "自绘" in html or "self-rendered" in html.lower())
check("含腾讯地图加载失败回退", "__TMAP_FAILED" in html)

print("\n==> " + ("全部通过" if ok else "存在失败项"))
sys.exit(0 if ok else 1)
