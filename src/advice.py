"""行动建议生成层：从结构化风险数据 → 可执行的田间行动建议。

两种渲染路径（互为备份，保证在无网络/无 API Key 时系统仍可用）
----------------------------------------------------------------
A. 规则引擎（默认）：风险类型 × 地形 × 物候 的组合规则，输出结构化行动项；
   优点是可审计、可复现、不会"编造"。
B. LLM 解释层（可选开启）：把结构化风险数据 + 地块属性交给大模型，
   按 few-shot 约束生成差异化的、带优先级的自然语言建议，
   并可随茶农反馈迭代措辞（提示词与输出格式见 build_llm_prompt / parse_llm_output）。

设计原则：LLM 只负责"把已有结论说清楚、说具体"，不负责生成风险判断本身——
判断权始终在可审计的物理内核 + 分类器手中。
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta

import config as C

ACTION_LIB = {
    "irrigate_pre": {
        "action": "提前灌溉（滴灌/微喷）",
        "why": "未来窗口内坡面辐射强、蒸散需求高，土壤有效水将快速下降",
        "when": "次日 15:00 前完成",
        "how": "滴灌 25–35 min（湿润 0–20 cm 根层），以手握 10 cm 土层成团不散为达标",
    },
    "irrigate_dusk": {
        "action": "傍晚补灌",
        "why": "已出现明显水分亏缺，需在夜间恢复冠层水分状态",
        "when": "当日 17:30 后",
        "how": "滴灌 40 min；隔日清晨复测墒情，不足则追加 20 min",
    },
    "irrigate_high_eff": {
        "action": "优先灌溉（复合胁迫，灌溉边际效益最高）",
        "why": "高温与干旱叠加：蒸散需求大且供水不足，下午叶片温度显著高于气温",
        "when": "立即安排，24 h 内完成",
        "how": "滴灌 40–50 min，分两次（傍晚 + 次日清晨）避免地表径流损失",
    },
    "shade_check": {
        "action": "检查/补齐遮阳网",
        "why": "遮阳网可降低冠层温度 2–4 ℃，是最直接的物理防护",
        "when": "当日午后前",
        "how": "检查覆盖率与固定绳，阳坡地块优先；覆盖率不足 70% 的地块补齐",
    },
    "mist": {
        "action": "树冠间歇喷雾降温",
        "why": "当日最高气温将接近或超过 38 ℃，冠层存在灼伤风险",
        "when": "13:30–16:00 之间",
        "how": "每次 5–10 min、间隔 30 min，避免持续积水引发根区缺氧",
    },
    "mulch": {
        "action": "地表覆盖（稻草/无纺布）",
        "why": "降低土面蒸发，减缓土壤水下降速率",
        "when": "本周内",
        "how": "覆盖厚度 3–5 cm，距根颈保留 5 cm 空隙",
    },
    "pause_fieldwork": {
        "action": "暂停午间田间作业",
        "why": "高温时段人工作业中暑风险高，且扰动会加剧蒸腾",
        "when": "12:00–16:00",
        "how": "作业调整至 6:00–10:00 与 17:00 后",
    },
    "soil_check": {
        "action": "复测土壤墒情",
        "why": "模型判断存在水分亏缺，需实地确认",
        "when": "次日清晨",
        "how": "10 cm 与 20 cm 两层各取 3 点，记录后可反哺模型校准",
    },
    "frost_cover": {
        "action": "夜间覆盖防霜",
        "why": "窗口内日最低气温可能低于受害阈值",
        "when": "当日 20:00 前完成覆盖",
        "how": "蓬面覆盖无纺布/遮阳网，凌晨 2:00–5:00 为关键时段",
    },
    "frost_water": {
        "action": "霜前灌水",
        "why": "提高土壤热容量，减缓冠层降温速率",
        "when": "霜冻前一日下午",
        "how": "傍晚灌透一次，避免叶片带水过夜",
    },
    "pick_early": {
        "action": "采摘窗口前移",
        "why": "午后高温将导致鲜叶失水加快、品质下降",
        "when": "6:00–9:30",
        "how": "优先采摘阳坡与上坡位地块；鲜叶及时摊放散热，厚度 <3 cm",
    },
    "patrol_order": {
        "action": "调整巡园顺序",
        "why": "风险在空间上高度不均，有限人力应投向高风险地块",
        "when": "每日清晨",
        "how": "按下表顺序巡园，重点关注叶片卷曲、叶缘焦枯、新梢萎蔫",
    },
}

DOMINANT_ACTIONS = {
    "热害": ["shade_check", "irrigate_pre", "pause_fieldwork"],
    "旱害": ["irrigate_dusk", "soil_check", "mulch"],
    "高温干旱复合": ["irrigate_high_eff", "shade_check", "soil_check"],
    "霜冻": ["frost_cover", "frost_water"],
}


def _fmt_window(dates: list[str], i: int) -> str:
    return dates[i] if i < len(dates) else dates[-1]


def unit_advice(unit: dict, state: dict, dates: list[str]) -> dict:
    """单地块建议。state 为该单元代表单元的指标快照。"""
    cls = int(state["cls"])
    dom = state["dominant_cn"]
    keys = list(DOMINANT_ACTIONS.get(dom, []))
    if cls == 2:
        keys = keys[:3]
    elif cls == 1:
        keys = keys[:2]
    else:
        keys = []

    month = datetime.strptime(dates[0], "%Y-%m-%d").month
    season_note = ""
    if month in (3, 4):
        season_note = "当前为春茶采摘期，防霜与采摘时序同等重要。"
        if cls >= 1:
            keys.append("pick_early")
    elif month in (7, 8, 9):
        season_note = "当前为高温伏旱期，修剪后新梢对水分亏缺最敏感。"
    elif month in (10, 11):
        season_note = "当前为秋季生长末期，重点防止根系与冠层在干旱中受损、影响越冬。"

    if cls == 2 and state.get("hsu_sum", 0) > 20:
        keys.append("mist")

    actions = []
    for k in dict.fromkeys(keys):
        a = dict(ACTION_LIB[k])
        a["key"] = k
        a.setdefault("when", "按理由中的时机执行")
        a.setdefault("how", "")
        a.setdefault("why", "")
        actions.append(a)

    conf = float(state.get("agree", 1.0))
    level = {0: "低", 1: "中", 2: "高"}[cls]
    headline = {
        2: f"{unit['unit_id']}（{unit['aspect_label']}）未来 {C.RISK['horizon_days']} 天风险高，主导胁迫：{dom}。",
        1: f"{unit['unit_id']}（{unit['aspect_label']}）风险中等，需提前准备：{dom}。",
        0: f"{unit['unit_id']}（{unit['aspect_label']}）风险低，维持常规管理。",
    }[cls]

    return {
        "unit_id": unit["unit_id"],
        "aspect_label": unit["aspect_label"],
        "area_ha": unit["area_ha"],
        "elev_mean_m": unit["elev_mean_m"],
        "slope_mean_deg": unit["slope_mean_deg"],
        "risk_level": level,
        "risk_class": cls,
        "risk_score": round(float(state["score"]), 3),
        "dominant": dom,
        "confidence": round(conf, 2),
        "headline": headline,
        "season_note": season_note,
        "evidence": [
            f"窗口内累积热害度时（>32 ℃）{round(float(state.get('hsu_sum', 0)), 1)} ℃·h",
            f"日最高气温峰值 {round(float(state.get('tmax_peak', 0)), 1)} ℃",
            f"土壤相对有效水最低 {round(float(state.get('sm_rel_min', 0)), 2)}"
            f"（适宜 ≥{C.TEA['sm_rel_optimal']:.2f}）",
            f"高温日（≥35 ℃）{int(state.get('days_over_35', 0))} 天",
            f"模型参数集合一致度 {conf:.0%}（{state.get('draws', '-')} 次抽样）",
        ],
        "actions": actions,
        "window": _fmt_window(dates, 0),
        "disclaimer": C.DISCLAIMER,
    }


def build_all(units: list[dict], unit_states: dict, dates: list[str]) -> list[dict]:
    out = [unit_advice(u, unit_states[u["unit_id"]], dates) for u in units]
    out.sort(key=lambda x: (-x["risk_class"], -x["risk_score"]))
    for i, a in enumerate(out, start=1):
        a["priority"] = i
    return out


def patrol_plan(advices: list[dict], top: int = 3) -> dict:
    """巡园优先级表（把人力和风险对齐）。"""
    hot = [a for a in advices if a["risk_class"] >= 1][:top] or advices[:top]
    return {
        "action": ACTION_LIB["patrol_order"]["action"],
        "why": ACTION_LIB["patrol_order"]["why"],
        "when": ACTION_LIB["patrol_order"]["when"],
        "order": [{"rank": i + 1, "unit_id": a["unit_id"],
                   "aspect": a["aspect_label"], "level": a["risk_level"],
                   "focus": _patrol_focus(a["dominant"])} for i, a in enumerate(hot)],
    }


def _patrol_focus(dom: str) -> str:
    return {
        "热害": "叶片卷曲、叶缘焦枯、新梢萎蔫",
        "旱害": "叶片失去光泽、新梢伸展停滞、土层干燥",
        "高温干旱复合": "优先看阳坡上坡位：叶缘焦枯 + 表层干裂",
        "霜冻": "蓬面结霜、幼芽变褐",
    }.get(dom, "冠层长势与土壤墒情")


def overall_summary(advices: list[dict], mode_label: str, dates: list[str]) -> dict:
    n_high = sum(1 for a in advices if a["risk_class"] == 2)
    n_mid = sum(1 for a in advices if a["risk_class"] == 1)
    worst = advices[0] if advices else None
    if n_high:
        text = (f"{mode_label}：共 {len(advices)} 个管理单元中 {n_high} 个高风险、{n_mid} 个中等风险。"
                f"最高风险为 {worst['unit_id']}（{worst['aspect_label']}，{worst['dominant']}），"
                f"建议 24 h 内优先处置。")
    elif n_mid:
        text = (f"{mode_label}：无高风险地块，{n_mid} 个中等风险单元，建议提前检查灌溉与遮阳设施。")
    else:
        text = (f"{mode_label}：全部 {len(advices)} 个管理单元风险低，"
                f"维持常规管理，无需启动应急灌溉。")
    return {
        "text": text, "n_units": len(advices),
        "n_high": n_high, "n_mid": n_mid, "n_low": len(advices) - n_high - n_mid,
        "window": f"{dates[0]} → {dates[min(len(dates) - 1, C.RISK['horizon_days'] - 1)]}",
        "worst_unit": worst["unit_id"] if worst else None,
    }


def weekly_bulletin(summary: dict, advices: list[dict], backtest: dict, mode_label: str) -> str:
    """公众版"茶园气候风险周报"——把专业信息转成可传播的气候叙事。"""
    lines = [f"【梅家坞茶园气候风险周报】{summary['window']}",
             "",
             f"· 本期结论：{summary['text']}"]
    if summary["n_high"]:
        lines.append(f"· 高风险地块：{ '、'.join(a['unit_id'] + '(' + a['dominant'] + ')' for a in advices if a['risk_class'] == 2) }")
    lines.append("· 空间差异：同一谷地内，阳坡地块的累积热害度时显著高于阴坡，"
                 "统一阈值管理会在阳坡地块上滞后数日——这正是本系统提供地块级预警的原因。")
    if backtest:
        yrs = sorted(backtest.keys())[-3:]
        seg = "；".join(f"{y} 年夏季高温日（≥35 ℃）{backtest[y]['metrics']['days_ge35']} 天、"
                        f"极端高温日（≥38 ℃）{backtest[y]['metrics']['days_ge38']} 天" for y in yrs if y in backtest)
        lines.append(f"· 气候背景：{seg}。")
    lines.append("· 对消费者意味着什么：你杯中这一年的龙井，其香气与鲜爽度，与夏季"
                 "茶树是否经历高温干旱直接相关。")
    lines.append("")
    lines.append("（数据来源：Open-Meteo 区域气象再分析 + 公开 DEM；本简报为气候风险评估，"
                 "不构成茶叶品质或价格判断。）")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# LLM 解释层
# --------------------------------------------------------------------------
LLM_SYSTEM = (
    "你是西湖龙井核心产区（杭州梅家坞）的茶园气候风险管理助手。"
    "你会收到结构化的地块级气候风险数据。你的任务是把它转写成茶农能直接照做的行动建议。"
    "硬性要求：(1) 必须写清『哪块地、什么时间、做什么、做多少』；"
    "(2) 不得新增数据中没有的结论，不得夸大风险；"
    "(3) 必须保留『辅助参考，请结合实地观察』的表述；"
    "(4) 输出 JSON，字段与示例一致，不要输出多余文字。"
)

LLM_FEWSHOT = [
    {"role": "user", "content": json.dumps({
        "地块": "MDW-03", "坡向": "阳坡（南向）", "海拔_m": 128, "风险等级": "高",
        "主导胁迫": "高温干旱复合", "日最高气温峰值_℃": 40.2, "累积热害度时_℃h": 41.5,
        "土壤相对有效水": 0.38, "窗口": "2022-08-12 → 2022-08-14",
    }, ensure_ascii=False)},
    {"role": "assistant", "content": json.dumps({
        "地块": "MDW-03", "风险等级": "高", "一句话结论":
            "阳坡 MDW-03 未来两天下午土壤水分将明显下降，需在明天傍晚前完成一次补灌。",
        "行动": [
            {"序号": 1, "动作": "滴灌 40 min（湿润 0–20 cm 根层）",
             "时间": "2022-08-12 17:30 后", "理由": "累积热害度时已达 41.5 ℃·h，土壤有效水降至 0.38"},
            {"序号": 2, "动作": "检查遮阳网覆盖率并固定绳",
             "时间": "2022-08-12 中午前", "理由": "日最高气温将达 40.2 ℃，遮阳可降冠层温度 2–4 ℃"},
        ],
        "提示": "高温时段（12:00–16:00）暂停田间作业。辅助参考，请结合实地观察。",
    }, ensure_ascii=False)},
]


def build_llm_prompt(unit_advice_item: dict, mode_label: str) -> list[dict]:
    """构造 few-shot 提示词；返回 messages 列表，可直接发给任意 OpenAI 兼容接口。"""
    payload = {
        "地块": unit_advice_item["unit_id"],
        "坡向": unit_advice_item["aspect_label"],
        "面积_ha": unit_advice_item["area_ha"],
        "海拔_m": unit_advice_item["elev_mean_m"],
        "坡度_°": unit_advice_item["slope_mean_deg"],
        "风险等级": unit_advice_item["risk_level"],
        "主导胁迫": unit_advice_item["dominant"],
        "置信度": unit_advice_item["confidence"],
        "窗口": unit_advice_item["window"],
        "证据": unit_advice_item["evidence"],
        "情景": mode_label,
    }
    msgs = [{"role": "system", "content": LLM_SYSTEM}] + LLM_FEWSHOT
    msgs.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
    return msgs


def call_llm(messages: list[dict], model: str | None = None,
             timeout: int = 45) -> dict | None:
    """可选：调用 OpenAI 兼容接口（如 Qwen / DeepSeek）。

    需要环境变量 LLM_API_KEY；可选 LLM_BASE_URL、LLM_MODEL。
    未配置时返回 None，系统自动退回规则引擎渲染——保证 Demo 在任何环境下可运行。
    """
    key = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        return None
    base = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    mdl = model or os.environ.get("LLM_MODEL", "qwen-plus")
    try:
        import requests
        r = requests.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": mdl, "messages": messages, "temperature": 0.3,
                  "response_format": {"type": "json_object"}},
            timeout=timeout)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(txt)
        except Exception:  # noqa: BLE001
            return {"raw": txt}
    except Exception as e:  # noqa: BLE001
        print(f"[advice] LLM 调用失败，回退规则引擎: {e}")
        return None
