# 梅家坞微气候风险预警与行动决策助手

> 用公开气象数据 + 高分辨率地形 + 机器学习，把"明天 38 ℃"翻译成
> **"哪块地、什么时候、做什么、做多少"**。

一句话定位：**面向西湖龙井核心产区（杭州梅家坞）的微气候级"风险预警 → 行动决策"闭环工具**。

**在线 Demo**：<https://meijiawu-tea-dashboard.app.workbuddy.host/>

> 单文件离线看板，直接打开即可；也可在本地打开 `app/dashboard.html`（无需联网、无任何外部依赖）。
> 说明：线上为静态托管，腾讯位置服务的代理未注入，故区位底图不展示（页面会自动降级并给出说明）；
> 风险网格为自绘可视化，不依赖任何底图服务，功能完整。

它不只告诉你"有风险"，而是告诉你——**阳坡 MDW-04 明天傍晚该开 40 分钟滴灌，
阴坡 MDW-08 可以再等两天**。梅家坞三面环山、溪谷纵横，同一片谷地里阳坡与阴坡的
日最高气温差异可达数摄氏度；公里级天气预报说"明天 35 ℃"，茶农真正需要知道的是
"我那几垄阳坡的茶树下午要不要提前喷灌"。

---

## 一、这个仓库里有什么

| 层级 | 模块 | 作用 |
|---|---|---|
| 数据 | `src/fetch_weather.py` | Open-Meteo 预报 / 历史再分析（免密钥），区域强迫场 |
| 数据 | `src/fetch_dem.py` | Mapzen·AWS Terrarium 地形瓦片 → 10 m 规则网格 DEM |
| 地形 | `src/terrain.py` | 坡度、坡向、地形位置指数、汇流累积、TWI、晕渲 |
| 地形 | `src/solar.py` | 太阳高度/方位角、逐方位地平线遮蔽、坡面短波辐射 |
| AI ①| `src/microclimate.py` | **微气候降尺度**：高程直减 + 辐射项 + 夜间冷池 + FAO-56 水量平衡 |
| AI ②| `src/risk_model.py` | **参数不确定性蒙特卡洛 + 弱监督随机森林代理模型**，输出风险概率与特征归因 |
| 指标 | `src/stress.py` | 热害（度时累积）/ 旱害（有效水亏缺）/ 霜冻三指数与复合风险分级 |
| 决策 | `src/advice.py` | 规则引擎 + **LLM 提示词构建器**（无 API Key 时自动回退规则引擎） |
| 验证 | `src/backtest.py` | 用真实历史灾情复盘：预警提前量、命中率、漏报率 |
| 呈现 | `src/build_dashboard.py` | 单文件 HTML 仪表盘 + Markdown 报告 |
| 地块 | `src/plots.py` | 地形适宜性掩膜 + 连通域聚类 + KMeans 细分出候选管理单元 |

### 数据流

```
Open-Meteo 预报/再分析 ─┐
                        ├─► 区域强迫场 ─┐
Terrarium DEM (≈4 m) ───┘               │
        │                               ▼
        ├─► 坡度/坡向/TPI/TWI/遮蔽 ─► 微气候降尺度内核 ─► 10 m 地块级温湿/土壤水
        │                               │                        │
        │                               │                        ▼
        │                    参数蒙特卡洛 (N 次)         热害/旱害/霜冻指数
        │                               │                        │
        └─► 地形适宜性掩膜 ─► 管理单元 ──┴──► 风险分级 + 置信度 ──┤
                                                                ▼
                                       随机森林代理模型 ◄── 弱监督蒸馏
                                                │
                                                ▼
                                    行动建议（规则引擎 / LLM 解释层）
                                                │
                                     ┌──────────┴──────────┐
                                     ▼                     ▼
                              dashboard.html          risk_report.md
```

---

## 二、快速开始

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py                # 完整运行（历史回溯 + 代理模型训练）
python run_pipeline.py --fast         # 少抽样，快速出结果
python run_pipeline.py --refresh      # 强制重新抓取数据

# 打开 Demo
start app/dashboard.html              # Windows
open  app/dashboard.html              # macOS
```

**无需任何 API Key**。全部数据源均为免密钥公开接口，首次运行约需 2–5 分钟
（含 DEM 瓦片下载），之后走本地缓存。

可选：交互式调参版 Demo

```bash
pip install streamlit
streamlit run app/streamlit_app.py
```

可选：启用大模型解释层（不配置则自动使用规则引擎，功能不缺失）

```bash
export LLM_API_KEY=sk-xxx          # OpenAI 兼容接口，默认阿里云百炼
export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export LLM_MODEL=qwen-plus
```

### 产物

| 文件 | 内容 |
|---|---|
| `app/dashboard.html` | 单文件离线 Demo：风险网格、地块预警、72 h 曲线、回溯验证、模型说明 |
| `outputs/risk_report.md` | 完整分析报告（可直接作为参赛技术附件） |
| `outputs/dashboard_data.json` | 仪表盘数据包（供二次开发） |
| `outputs/plot_units.geojson` | 管理单元矢量边界（可导入 QGIS / 腾讯地图） |
| `outputs/plot_units_centroids.csv` | 地块质心（供 GEE 采样用） |
| `outputs/backtest.json` | 历年回溯验证明细 |
| `outputs/run_config.json` | 本次运行全部参数（可复现、可审计） |

---

## 三、AI 在方案里承担什么（为什么不用传统方法）

**任务 1 · 微气候降尺度 —— 传统方法不可行**
梅家坞茶园分布在三面环山的谷地坡面上，区域气象场是公里级，而坡向/坡位造成的
温差与湿度差异是地块级的。用传统物理降尺度（地表能量平衡）需要精确标定反照率、
粗糙度、热惯量等下垫面参数，在这样异质性强、观测稀疏的山区几乎无法完成。
本项目用**物理内核 + 可学习残差**的混合结构：物理部分（直减率、坡面辐射差异、
夜间冷池、水量平衡）给出可解释的场；残差部分（`src/models/downscaler_lstm.py`，
LSTM / Temporal Fusion Transformer）在小气候站观测到位后做数据驱动订正。

**任务 2 · 多源信号融合与风险分级 —— 人工阈值误报率高**
小尺度茶园存在云雾、修剪、施肥、采摘等大量干扰，单指标阈值法（例如"气温 ≥ 38 ℃ 就报警"）
在空间上必然失衡：阳坡可能早已达到胁迫阈值，而阴坡尚未。本项目用
**参数蒙特卡洛 + 随机森林代理模型**：先用物理内核的参数分布生成软标签与"预警置信度"，
再用随机森林学习「可观测特征 → 风险等级」的映射，可输出概率、可做特征归因，
并**保留一个换成真实田间标签即可再训练的接口**。

**任务 3 · 行动建议生成 —— 模板规则缺乏针对性**
从结构化风险数据到自然语言建议，`if risk=high then"建议灌溉"` 无法给出
"哪块地、什么时候、灌多久"。本项目规则引擎负责**保底与可审计**，
LLM 解释层负责**把结论说得更具体、更有优先级**，并可随茶农反馈迭代措辞。
LLM 只做语言转写，不产生风险判断——判断权始终在可审计的物理内核与分类器手中。

---

## 四、必须说明的局限（不回避）

1. **地块边界**为地形适宜性聚类得到的**候选管理单元**，尚未与《2022 年全国 10 m 分辨率
   茶园分布数据集》叠加校正；替换方式见 `src/plots.py::tea_suitability_mask()` 的说明。
2. **微气候系数**（`k_rad`、`k_drain`、直减率）为文献先验值，尚未用茶园小气候站观测标定；
   标定接口在 `config.DOWNSCALE`。
3. **代理模型**的精度是相对物理内核**弱标签**的一致性，**不是**田间真实预警精度；
   真实精度需在拿到小气候站或农情调查数据后重算。
4. **遥感胁迫信号**（Sentinel-2 NDVI/NDRE/NDWI）接口已就绪但当前运行**未接入**，
   因为需要 GEE 账号与影像导出流程（脚本见 `scripts/gee_sentinel2_export.js`）。
5. **灌溉时长**基于水量平衡估算，须结合实地灌溉设备流量二次校准。

> 本系统输出为气候风险评估与辅助建议，不替代茶农的实地判断与农技部门指导。
> 系统不以预警结果自动判定茶叶品质或价格。

---

## 五、合规说明

- 风险网格图是**自绘数据可视化**（由 DEM 与模型输出渲染），不依赖任何第三方瓦片；
- 区位底图使用**腾讯位置服务 GL JS**（官方代理模式，前端零密钥），
  且坐标已由 WGS-84 转换为 **GCJ-02**；
- 全部外部数据均为公开或免密钥来源，不采集、不存储任何个人信息与位置轨迹。

## 六、目录结构

```
meijiawu-tea-risk/
├── config.py                    全部参数（可复现、可审计）
├── run_pipeline.py              一键运行
├── requirements.txt
├── src/                         核心代码
│   ├── fetch_weather.py  fetch_dem.py
│   ├── terrain.py        solar.py
│   ├── microclimate.py   stress.py      plots.py
│   ├── risk_model.py     advice.py      backtest.py
│   └── models/downscaler_lstm.py        可选：LSTM / TFT 残差学习
├── app/
│   ├── dashboard.html           生成产物：单文件 Demo
│   └── streamlit_app.py         交互式版本
├── scripts/
│   └── gee_sentinel2_export.js  GEE：茶园提取 + 遥感指数导出
├── docs/
│   ├── 参赛方案.md
│   ├── AI技术说明.md
│   ├── 验证计划.md
│   └── 数据来源与待补清单.md
├── data/{raw,processed}/        数据缓存
└── outputs/                     运行产物
```
