"""仪表盘与分析报告生成：把管线产物变成单文件可运行 Demo + Markdown 报告。

Demo 设计要点（对应评审维度）
----------------------------
- 「零门槛」：单文件 HTML，双击即开，不需要装环境、不需要联网；
- 「可行动」：每个地块给出"何时/何地/做什么/做多少"，并标注置信度与依据；
- 「可核查」：风险判定依据（度时、土壤有效水、置信度）逐条列出，不隐藏不确定性；
- 「合规」：底图使用腾讯位置服务（代理模式，前端零密钥），风险网格为自绘数据可视化。
"""
from __future__ import annotations

import json

LEVEL_COLOR = {0: "#35a463", 1: "#e8b93b", 2: "#e5484d"}
LEVEL_CN = {0: "低", 1: "中", 2: "高"}


def blindspot_text(ls: dict) -> str:
    """把"统一阈值盲区"度量拼成一句人话。

    同时给出三个互补事实，避免单一比例被极端情景"抹平"：
      1) 区域场与最热地块的温差（始终有定义）；
      2) 区域未越 35 ℃ 的日数 + 其中仍有地块越 35 ℃ 的日数；
      3) 区域未越 32 ℃（度时累积起点）时，最热日受影响像元占比。
    """
    bs = ls.get("blindspot") or {}
    if not bs:
        return (f"在区域最高温未达 35 ℃ 的日子里，仍有 "
                f"{ls.get('region_underestimates_cells_frac', 0) * 100:.1f}% 的茶园像元微气候已达热害起点")

    parts = [f"地块级最高温 {bs['cell_peak_c']} ℃ 比区域气象场 {bs['region_peak_c']} ℃ 高 "
             f"**{bs['exceedance_c']} ℃**"]
    if bs["silent_days"]:
        if bs["silent_hot_days"]:
            parts.append(f"区域最高温未达 {bs['mild_threshold_c']:.0f} ℃ 的 {bs['silent_days']} 天里，"
                         f"仍有 {bs['silent_hot_days']} 天出现 ≥{bs['mild_threshold_c']:.0f} ℃ 的地块"
                         f"（最热日受影响像元占 {bs['hot_frac_max_pct']:.1f}%）")
        else:
            parts.append(f"区域最高温未达 {bs['mild_threshold_c']:.0f} ℃ 的 {bs['silent_days']} 天里，"
                         f"地块级尚未越 35 ℃（该阈值口径下无盲区）")
    else:
        parts.append(f"窗口内区域气象场本身已越 {bs['mild_threshold_c']:.0f} ℃，"
                     f"不存在『区域沉默』的日数（此时统一阈值同样能报警）")
    if bs["onset_silent_days"] and bs["onset_hot_frac_max_pct"] > 0:
        parts.append(f"若把口径收紧到度时累积起点 {bs['onset_threshold_c']:.0f} ℃，"
                     f"区域未越阈的 {bs['onset_silent_days']} 天里最热日已有 "
                     f"{bs['onset_hot_frac_max_pct']:.1f}% 的茶园像元越阈")
    return "；".join(parts)



def render_html(pack: dict) -> str:
    payload = json.dumps(pack, ensure_ascii=False, separators=(",", ":"))
    return _TEMPLATE.replace("__PAYLOAD__", payload)


# --------------------------------------------------------------------------
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>一垄一候 · 梅家坞茶园微气候风险预警</title>
<style>
:root{
  --bg:#0f1216; --panel:#171b21; --panel2:#1e232b; --line:#2a313b;
  --text:#e7ecf3; --muted:#94a0b1; --dim:#6b7688;
  --red:#e5484d; --amber:#e8b93b; --green:#35a463; --blue:#4c8dff; --violet:#9a7bff;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--text);
  font-family:"HarmonyOS Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif;
  font-size:14px;line-height:1.6}
a{color:var(--blue)}
.wrap{max-width:1560px;margin:0 auto;padding:20px 22px 60px}
header.top{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end;justify-content:space-between;
  padding-bottom:14px;border-bottom:1px solid var(--line);margin-bottom:18px}
h1{font-size:20px;margin:0 0 4px;letter-spacing:.3px}
h1 small{display:block;font-size:12px;font-weight:400;color:var(--muted);letter-spacing:0;margin-top:5px}
.modes{display:flex;flex-wrap:wrap;gap:8px}
.mode-btn{background:var(--panel2);border:1px solid var(--line);color:var(--muted);
  padding:7px 13px;border-radius:6px;cursor:pointer;font-size:12.5px;transition:.15s}
.mode-btn:hover{border-color:#3d4756;color:var(--text)}
.mode-btn.on{background:#22304a;border-color:#3a5c9a;color:#cfe0ff}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.kpi .v{font-size:24px;font-weight:600;letter-spacing:.5px;font-variant-numeric:tabular-nums}
.kpi .l{font-size:11.5px;color:var(--muted);margin-top:2px}
.kpi .s{font-size:11px;color:var(--dim);margin-top:3px}
.grid{display:grid;grid-template-columns:minmax(0,1.32fr) minmax(0,1fr);gap:16px;align-items:start}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;margin-top:16px;align-items:start}
@media(max-width:1140px){.grid,.grid2{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:14px 16px}
.panel h2{font-size:13.5px;margin:0 0 10px;color:#c8d2e0;font-weight:600;letter-spacing:.4px;
  display:flex;align-items:center;gap:8px;justify-content:space-between}
.panel h2 .hint{font-size:11px;color:var(--dim);font-weight:400}
.tabs{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}
.tab{font-size:12px;padding:5px 11px;border-radius:5px;border:1px solid var(--line);
  background:var(--panel2);color:var(--muted);cursor:pointer}
.tab.on{background:#1d3352;border-color:#365d94;color:#cfe0ff}
.mapwrap{position:relative;border-radius:7px;overflow:hidden;background:#0b0e12;border:1px solid var(--line)}
.mapwrap img{display:block;width:100%;height:auto}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--muted);margin-top:9px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;
  vertical-align:-1px}
.scale{display:flex;height:12px;border-radius:3px;overflow:hidden;margin-top:7px}
.scale div{flex:1}
.unit{background:var(--panel2);border:1px solid var(--line);border-left-width:3px;border-radius:7px;
  padding:11px 13px;margin-bottom:9px;cursor:pointer;transition:.15s}
.unit:hover{border-color:#3d4756;background:#232935}
.unit.on{border-color:#3f6ba8;background:#1d2531}
.unit .row1{display:flex;justify-content:space-between;align-items:center;gap:10px}
.unit .id{font-weight:600;font-size:13.5px}
.badge{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600;white-space:nowrap}
.b-high{background:rgba(229,72,77,.16);color:#ff8b8f;border:1px solid rgba(229,72,77,.45)}
.b-mid{background:rgba(232,185,59,.14);color:#f0cd6b;border:1px solid rgba(232,185,59,.42)}
.b-low{background:rgba(53,164,99,.14);color:#6fce9b;border:1px solid rgba(53,164,99,.4)}
.unit .meta{font-size:11.5px;color:var(--muted);margin-top:3px}
.unit .act{margin-top:9px;padding-top:9px;border-top:1px dashed var(--line);display:none}
.unit.on .act{display:block}
.act-item{padding:7px 0 7px 0;border-bottom:1px dotted #2b323d}
.act-item:last-child{border-bottom:0}
.act-item .a{font-weight:600;font-size:12.5px;color:#dbe4f0}
.act-item .w{font-size:11.5px;color:var(--amber)}
.act-item .r{font-size:11.5px;color:var(--muted)}
.evi{font-size:11.5px;color:var(--muted);margin-top:3px}
.evi code{background:#10141a;border:1px solid var(--line);border-radius:4px;padding:1px 5px;
  margin-right:5px;display:inline-block;margin-bottom:4px;font-size:11px;color:#a8b6c9}
.msg{background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--blue);
  border-radius:7px;padding:11px 14px;font-size:13px;margin-bottom:12px}
.msg.warn{border-left-color:var(--red)}
.msg.ok{border-left-color:var(--green)}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:500;font-size:11.5px}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}
svg{display:block;width:100%;height:auto}
.src li{font-size:12px;color:var(--muted);margin-bottom:6px}
.src b{color:#c2cddd;font-weight:600}
.lim{font-size:12px;color:#d9be8a;background:rgba(232,185,59,.06);border:1px solid rgba(232,185,59,.22);
  border-radius:7px;padding:10px 14px}
.lim li{margin:3px 0}
footer{margin-top:26px;padding-top:14px;border-top:1px solid var(--line);font-size:11.5px;color:var(--dim)}
#tmap{width:100%;height:330px;border-radius:7px;background:#0b0e12;border:1px solid var(--line)}
.fb{font-size:11.5px;color:var(--dim);padding:10px 12px;background:var(--panel2);border-radius:6px;
  border:1px solid var(--line);margin-top:8px}
.pill{display:inline-block;font-size:11px;padding:1px 7px;border-radius:4px;background:#22304a;
  color:#a9c3ec;margin-right:6px}
</style>
<!-- 合规地图：腾讯位置服务 GL JS，官方代理模式，前端不出现任何密钥 -->
<script type="text/javascript">
  window._TMapSecurityConfig = {
    serviceHost: 'http://127.0.0.1:__WB_HTTP_PORT__/_TMapService/_wbt/__WB_TMAP_SECRET__',
  };
  // 代理占位符未被宿主注入时（静态托管、离线打开、外链部署），腾讯底图必然不可用。
  // 主动置位失败标志，交由 initMap() 隐藏地图容器并展示自绘风险网格的降级说明，
  // 避免留下一个加载不出瓦片的空黑框。
  if (String(window._TMapSecurityConfig.serviceHost).indexOf('__WB_') !== -1) {
    window.__TMAP_FAILED = 1;
  }
</script>
<script src="https://map.qq.com/api/gljs?v=1.exp" onerror="window.__TMAP_FAILED=1"></script>
</head>
<body>
<div class="wrap">
<header class="top">
  <div>
    <h1>一垄一候 · 梅家坞茶园微气候风险预警
      <small id="subtitle"></small>
    </h1>
  </div>
  <div class="modes" id="modes"></div>
</header>

<div class="kpis" id="kpis"></div>

<div class="grid">
  <section class="panel">
    <h2>微气候风险网格（10 m）
      <span class="hint" id="mapHint"></span></h2>
    <div class="tabs" id="layerTabs">
      <button class="tab on" data-layer="risk">风险分级</button>
      <button class="tab" data-layer="deviation">微气候温差</button>
      <button class="tab" data-layer="moisture">土壤有效水</button>
    </div>
    <div class="mapwrap"><img id="mapImg" alt="风险网格"></div>
    <div class="legend" id="legend"></div>
    <div class="scale" id="scaleBar"></div>
  </section>

  <section class="panel">
    <h2>地块预警与行动建议 <span class="hint">点击地块查看依据</span></h2>
    <div id="overall"></div>
    <div id="patrol"></div>
    <div id="units"></div>
  </section>
</div>

<div class="grid2">
  <section class="panel">
    <h2>代表地块 72 小时微气候过程 <span class="hint" id="chartUnit"></span></h2>
    <div class="tabs" id="chartTabs">
      <button class="tab on" data-s="all">全部</button>
      <button class="tab" data-s="temp">气温</button>
      <button class="tab" data-s="sm_rel">土壤有效水</button>
      <button class="tab" data-s="vpd">饱和水汽压差</button>
    </div>
    <svg id="chart" viewBox="0 0 680 260" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="evi" id="chartNote"></div>
  </section>

  <section class="panel">
    <h2>历史灾情回溯验证 <span class="hint">真实再分析数据</span></h2>
    <div class="scroll"><table id="btTable"><thead><tr>
      <th>年份</th><th>最高温</th><th>≥35℃日</th><th>≥38℃日</th><th>夏季降水</th>
      <th>土壤缺水日</th><th>高风险日</th><th>预警提前量</th><th>预警准确率</th><th>漏报率</th>
    </tr></thead><tbody></tbody></table></div>
    <svg id="btChart" viewBox="0 0 680 200" preserveAspectRatio="xMidYMid meet" style="margin-top:10px"></svg>
  </section>
</div>

<div class="grid2">
  <section class="panel">
    <h2>点位区位（腾讯位置服务） <span class="hint">合规底图｜坐标已转换 GCJ-02</span></h2>
    <div id="tmap"></div>
    <div class="fb" id="mapFallback">区位底图未加载：当前为静态托管或离线打开，腾讯位置服务的代理注入不可用，因此不展示底图。地块级风险网格为自绘可视化，不依赖任何底图服务，功能不受影响；各地块经纬度见下方坐标表。</div>
    <div class="scroll" style="margin-top:10px"><table id="coordTable"><thead><tr>
      <th>地块</th><th>坡向</th><th>坡度</th><th>纬度</th><th>经度</th>
    </tr></thead><tbody></tbody></table></div>
  </section>

  <section class="panel">
    <h2>AI 层说明 <span class="hint">弱监督蒸馏 + 参数不确定性</span></h2>
    <div id="mlBox"></div>
    <h2 style="margin-top:16px">特征重要性（随机森林）</h2>
    <svg id="impChart" viewBox="0 0 680 300" preserveAspectRatio="xMidYMid meet"></svg>
  </section>
</div>

<section class="panel" style="margin-top:16px">
  <h2>公众版 · 茶园气候风险周报 <span class="hint">可直接用于传播</span></h2>
  <pre id="bulletin" style="white-space:pre-wrap;font-family:inherit;font-size:12.5px;color:#c6d1e0;margin:0"></pre>
</section>

<section class="panel" style="margin-top:16px">
  <h2>数据来源、模型链路与局限</h2>
  <div id="srcBox"></div>
  <div class="lim" id="limBox"></div>
</section>

<footer id="footer"></footer>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const P = JSON.parse(document.getElementById('payload').textContent);
const LV = {0:'低',1:'中',2:'高'};
const LC = {0:'#35a463',1:'#e8b93b',2:'#e5484d'};
let S = {mode: P.default_mode, unit: null, layer:'risk', series:'all'};

/* ---------------- WGS-84 → GCJ-02（腾讯地图使用国测局坐标） ---------------- */
const PI=3.1415926535897932384626, A=6378245.0, EE=0.00669342162296594323;
function outOfChina(lat,lng){return !(lng>73.66&&lng<135.05&&lat>3.86&&lat<53.55);}
function tfLat(x,y){let r=-100+2*x+3*y+0.2*y*y+0.1*x*y+0.2*Math.sqrt(Math.abs(x));
  r+=(20*Math.sin(6*x*PI)+20*Math.sin(2*x*PI))*2/3;r+=(20*Math.sin(y*PI)+40*Math.sin(y/3*PI))*2/3;
  r+=(160*Math.sin(y/12*PI)+320*Math.sin(y*PI/30))*2/3;return r;}
function tfLng(x,y){let r=300+x+2*y+0.1*x*x+0.1*x*y+0.1*Math.sqrt(Math.abs(x));
  r+=(20*Math.sin(6*x*PI)+20*Math.sin(2*x*PI))*2/3;r+=(20*Math.sin(x*PI)+40*Math.sin(x/3*PI))*2/3;
  r+=(150*Math.sin(x/12*PI)+300*Math.sin(x/30*PI))*2/3;return r;}
function wgs2gcj(lat,lng){
  if(outOfChina(lat,lng))return [lat,lng];
  let dLat=tfLat(lng-105,lat-35), dLng=tfLng(lng-105,lat-35);
  const radLat=lat/180*PI; let magic=Math.sin(radLat); magic=1-EE*magic*magic;
  const sqrtMagic=Math.sqrt(magic);
  dLat=(dLat*180)/((A*(1-EE))/(magic*sqrtMagic)*PI);
  dLng=(dLng*180)/(A/sqrtMagic*Math.cos(radLat)*PI);
  return [lat+dLat,lng+dLng];
}

/* ---------------- 渲染 ---------------- */
function fmt(n,d){return (n===null||n===undefined||isNaN(n))?'—':Number(n).toFixed(d===undefined?1:d);}
function M(){return P.modes[S.mode];}

function renderModes(){
  const box=document.getElementById('modes'); box.innerHTML='';
  Object.keys(P.modes).forEach(k=>{
    const b=document.createElement('button');
    b.className='mode-btn'+(k===S.mode?' on':'');
    const m=P.modes[k];
    b.innerHTML=m.label+`　<span style="color:${m.summary.n_high?'#ff8b8f':'#6b7688'}">高${m.summary.n_high}</span>`;
    b.onclick=()=>{S.mode=k; S.unit=null; renderAll();};
    box.appendChild(b);
  });
}

function renderHeader(){
  document.getElementById('subtitle').textContent =
    `${P.site.village}｜网格 ${P.grid.res_m} m · ${P.grid.shape[1]}×${P.grid.shape[0]} 像元`
    + `｜DEM ${fmt(P.grid.dem_stats.elev_min,0)}–${fmt(P.grid.dem_stats.elev_max,0)} m`
    + `｜生成于 ${P.generated_at}`;
}

function renderKPIs(){
  const m=M(), s=m.summary, L=m.layer_stats;
  const pair=L.max_aspect_pair?L.max_aspect_pair.join(' vs '):'坡向';
  const spread=(L.sunny_minus_shady!==null)?fmt(L.sunny_minus_shady,1)+' ℃':'—';
  const items=[
    {v:s.n_high,l:'高风险地块',s:`共 ${s.n_units} 个管理单元`,c:LC[2]},
    {v:s.n_mid,l:'中等风险地块',s:'需提前准备',c:LC[1]},
    {v:s.n_low,l:'低风险地块',s:'常规管理',c:LC[0]},
    {v:fmt(L.max_aspect_diff,1)+' ℃',l:`最大坡向温差（${pair}）`,
      s:`阳坡−阴坡 ${spread}｜昼间均温差 ${fmt(L.sunny_minus_shady_daytime,1)} ℃`,c:'#4c8dff'},
    {v:fmt(L.microclimate_spread_c,1)+' ℃',l:'适茶带全天最高温极差',s:'10 m 网格内',c:'#9a7bff'},
    {v:fmt(L.region_tmax_peak,1)+' ℃',l:'区域气象最高温',s:'公里级预报场',c:'#94a0b1'},
    {v:fmt(L.tmax_peak_max,1)+' ℃',l:'地块级最高温',s:'10 m 降尺度结果',c:'#e8b93b'},
    {v:'+'+fmt((L.blindspot||{}).exceedance_c,1)+' ℃',l:'最热地块高出区域场',
      s:`统一阈值在区域未越 ${fmt((L.blindspot||{}).mild_threshold_c,0)} ℃ 时失效`,c:'#e5484d'},
    {v:fmt(Math.max((L.blindspot||{}).hot_frac_max_pct||0,(L.blindspot||{}).onset_hot_frac_max_pct||0),1)+'%',
      l:'统一阈值盲区（最热日）',
      s:(L.blindspot&&L.blindspot.silent_days)?`区域未越阈的 ${L.blindspot.silent_days} 天中 ${L.blindspot.silent_hot_days} 天出现越阈地块`:'窗口内区域场本身已越阈',c:'#e5484d'},
  ];
  document.getElementById('kpis').innerHTML=items.map(i=>
    `<div class="kpi"><div class="v" style="color:${i.c}">${i.v}</div>
     <div class="l">${i.l}</div><div class="s">${i.s}</div></div>`).join('');
}

function renderMap(){
  const m=M();
  document.getElementById('mapImg').src=m.images[S.layer];
  const L=m.layer_stats;
  document.getElementById('mapHint').textContent =
    S.layer==='risk'?`高风险 ${m.summary.n_high} 个单元｜灰化区域为地形不适宜（坡度>25° / 过湿 / 高程越界）`
    :S.layer==='deviation'?`相对全域均值的日最高气温偏差｜阳坡 − 阴坡 ${fmt(L.sunny_minus_shady,1)} ℃`
    :`窗口内土壤相对有效水最低值｜适宜区间 ≥0.72，胁迫阈值 0.55`;
  const legend={
    risk:[['#2e8b57','低风险'],['#e8b93b','中等风险'],['#cd2a28','高风险'],['#fff','管理单元边界与代表点']],
    deviation:[['#315495','偏冷（阴坡/谷地）'],['#f5f5eb','接近均值'],['#b22222','偏热（阳坡/上坡位）'],['#fff','管理单元边界']],
    moisture:[['#965a28','严重亏缺'],['#d6aa5a','偏干'],['#b0d6a0','适宜'],['#184a8c','充足']]
  }[S.layer];
  document.getElementById('legend').innerHTML=legend.map(l=>
    `<span><i style="background:${l[0]}"></i>${l[1]}</span>`).join('');
  const stops={risk:['#2e8b57','#8cc63f','#e8b93b','#e97e2d','#cd2a28'],
    deviation:['#315495','#78b2d6','#f5f5eb','#f09e60','#b22222'],
    moisture:['#965a28','#d6aa5a','#b0d6a0','#4898be','#184a8c']}[S.layer];
  document.getElementById('scaleBar').innerHTML=stops.map(c=>`<div style="background:${c}"></div>`).join('');
}

function renderUnits(){
  const m=M();
  const s=m.summary;
  const cls = s.n_high? 'warn' : (s.n_mid? '' : 'ok');
  document.getElementById('overall').innerHTML =
    `<div class="msg ${cls}"><b>本期结论</b><br>${s.text}
      <div class="evi" style="margin-top:6px">窗口 ${s.window}｜参与评估地块 ${s.n_units} 个</div></div>`;

  document.getElementById('patrol').innerHTML = m.patrol? (
    `<div class="msg" style="border-left-color:var(--amber)">
      <b>${m.patrol.action}</b><span style="color:var(--muted)">　${m.patrol.when}</span><br>
      <span style="font-size:12px;color:var(--muted)">${m.patrol.why}</span>
      <div class="evi">` + m.patrol.order.map(o=>
        `第 ${o.rank} 位：<b style="color:#dbe4f0">${o.unit_id}</b>（${o.aspect}）— 关注 ${o.focus}`).join('<br>') +
      `</div></div>`) : '';

  const box=document.getElementById('units'); box.innerHTML='';
  m.advices.forEach(a=>{
    const d=document.createElement('div');
    d.className='unit'+(S.unit===a.unit_id?' on':'');
    d.style.borderLeftColor=LC[a.risk_class];
    const badge=`<span class="badge b-${['low','mid','high'][a.risk_class]}">${a.risk_level}风险</span>`;
    d.innerHTML=`
      <div class="row1"><span class="id">#${a.priority} ${a.unit_id}
        <span style="font-size:11.5px;color:var(--muted);font-weight:400">${a.aspect_label}</span></span>${badge}</div>
      <div class="meta">${a.area_ha} ha｜海拔 ${a.elev_mean_m} m｜坡度 ${a.slope_mean_deg}°｜
        综合风险分 ${a.risk_score}｜置信度 ${(a.confidence*100).toFixed(0)}%</div>
      <div class="meta" style="color:#b9c6d8">${a.headline}</div>
      <div class="act">
        <div class="evi"><span class="pill">判定依据</span>${a.evidence.map(e=>`<code>${e}</code>`).join('')}</div>
        ${a.actions.length? a.actions.map((x,i)=>`
          <div class="act-item">
            <div class="a">${i+1}. ${x.action}</div>
            <div class="w">时间：${x.when}</div>
            <div class="r">做法：${x.how}</div>
            <div class="r">理由：${x.why}</div>
          </div>`).join('')
        : `<div class="act-item"><div class="r">当前风险低，无需应急措施；维持常规巡园与墒情观察。</div></div>`}
        ${a.season_note?`<div class="evi" style="margin-top:6px">物候提示：${a.season_note}</div>`:''}
        ${a.disclaimer?`<div class="evi" style="margin-top:6px;color:#d9be8a">${a.disclaimer}</div>`:''}
      </div>`;
    d.onclick=()=>{S.unit=(S.unit===a.unit_id?null:a.unit_id); renderUnits(); renderChart();};
    box.appendChild(d);
  });
  if(!S.unit && m.advices.length) S.unit=m.advices[0].unit_id;
}

/* ---------------- 折线图 ---------------- */
function renderChart(){
  const m=M(), svg=document.getElementById('chart');
  if(!S.unit) S.unit=m.advices[0]?.unit_id;
  const se=m.series[S.unit];
  if(!se||!se.t||!se.t.length){svg.innerHTML='<text x="20" y="40" fill="#6b7688">无序列数据</text>';return;}
  document.getElementById('chartUnit').textContent=S.unit+'｜'+m.advices.find(a=>a.unit_id===S.unit)?.aspect_label;
  const W=680,H=260,L=44,R=52,T=16,B=34;
  const x=i=>L+(i/(se.t.length-1))*(W-L-R);
  const temp=se.temp, sm=se.sm_rel, vpd=se.vpd;
  const t0=Math.min(...temp)-1, t1=Math.max(...temp)+1;
  const yT=v=>H-B-((v-t0)/(t1-t0))*(H-T-B);
  const smin=0, smax=1;
  const yS=v=>H-B-((v-smin)/(smax-smin))*(H-T-B);
  const show=(S.series==='all'||S.series==='temp');
  let g='';
  // 高温区底纹
  temp.forEach((v,i)=>{ if(v>=35){ g+=`<rect x="${x(i)-3}" y="${T}" width="6" height="${H-T-B}" fill="#e5484d" opacity="0.10"/>`;}});
  // 坐标轴
  g+=`<line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="#2a313b"/>`;
  g+=`<line x1="${L}" y1="${T}" x2="${L}" y2="${H-B}" stroke="#2a313b"/>`;
  g+=`<line x1="${W-R}" y1="${T}" x2="${W-R}" y2="${H-B}" stroke="#2a313b"/>`;
  for(let k=0;k<=4;k++){const v=t0+(t1-t0)*k/4;
    g+=`<text x="${L-6}" y="${yT(v)+4}" fill="#6b7688" font-size="10" text-anchor="end">${v.toFixed(0)}</text>`;}
  for(let k=0;k<=4;k++){const v=k/4;
    g+=`<text x="${W-R+6}" y="${yS(v)+4}" fill="#4c8dff" font-size="10">${v.toFixed(2)}</text>`;}
  const path=(arr,yf)=>arr.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(1)},${yf(v).toFixed(1)}`).join(' ');
  if(show) g+=`<path d="${path(temp,yT)}" fill="none" stroke="#e5484d" stroke-width="2"/>`;
  if(S.series==='all'||S.series==='sm_rel') g+=`<path d="${path(sm,yS)}" fill="none" stroke="#4c8dff" stroke-width="1.8" stroke-dasharray="5 3"/>`;
  if(S.series==='vpd'){
    const v0=0, v1=Math.max(...vpd)+0.5; const yV=v=>H-B-((v-v0)/(v1-v0))*(H-T-B);
    g+=`<path d="${path(vpd,yV)}" fill="none" stroke="#9a7bff" stroke-width="2"/>`;
    g+=`<text x="${L-6}" y="${T+10}" fill="#9a7bff" font-size="10" text-anchor="end">${v1.toFixed(1)}kPa</text>`;}
  // 35℃ 阈值线
  if(show&&t0<35&&t1>35) g+=`<line x1="${L}" y1="${yT(35)}" x2="${W-R}" y2="${yT(35)}" stroke="#e5484d" stroke-dasharray="4 4" opacity="0.6"/>
    <text x="${W-R-4}" y="${yT(35)-4}" fill="#e5484d" font-size="10" text-anchor="end">35 ℃ 热害起点</text>`;
  // 时间刻度
  const step=Math.max(1,Math.floor(se.t.length/5));
  for(let i=0;i<se.t.length;i+=step) g+=`<text x="${x(i)}" y="${H-B+16}" fill="#6b7688" font-size="10" text-anchor="middle">${se.t[i].slice(6,11)}</text>`;
  svg.innerHTML=g;
  const a=m.advices.find(u=>u.unit_id===S.unit);
  document.getElementById('chartNote').innerHTML = a ?
    `<code>${S.unit}</code><code>坡向 ${a.aspect_label}</code><code>${a.dominant}</code>`+
    `<code>最高温 ${fmt(Math.max(...temp),1)} ℃</code><code>最低有效水 ${fmt(Math.min(...sm),2)}</code>` : '';
}

/* ---------------- 回溯 ---------------- */
function renderBacktest(){
  const tb=document.querySelector('#btTable tbody'); tb.innerHTML='';
  const ys=Object.keys(P.backtest).filter(k=>/^\d+$/.test(k)).sort();
  ys.forEach(y=>{
    const r=P.backtest[y], k=r.metrics, de=r.day_eval||{};
    tb.innerHTML+=`<tr><td><b>${y}</b></td><td class="n">${k.tmax_peak}</td>
      <td class="n">${k.days_ge35}</td><td class="n">${k.days_ge38}</td>
      <td class="n">${k.precip_total_mm}</td><td class="n">${k.days_sm_stressed}</td>
      <td class="n" style="color:${r.high_risk_days?'#ff8b8f':'#6fce9b'}">${r.high_risk_days}</td>
      <td class="n">${r.lead_days_heat===null?'—':r.lead_days_heat+' 天'}</td>
      <td class="n">${de.precision===undefined?'—':(de.precision*100).toFixed(0)+'%'}</td>
      <td class="n">${de.miss_rate===undefined?'—':(de.miss_rate*100).toFixed(0)+'%'}</td></tr>`;
  });
  const sm=P.backtest.summary;
  if(sm){
    let el=document.getElementById('btSummary');
    if(!el){el=document.createElement('div'); el.id='btSummary'; el.className='evi';
      el.style.marginTop='8px'; document.getElementById('btTable').parentElement.appendChild(el);}
    el.innerHTML=`<span class="pill">五年汇总</span>平均预警提前量 <b style="color:#f0cd6b">`
      +`${sm.lead_days_mean} 天</b>（最长 ${sm.lead_days_max} 天）｜逐日预警准确率 `
      +`<b style="color:#6fce9b">${(sm.precision_mean*100).toFixed(0)}%</b>｜对胁迫日的召回率 `
      +`<b style="color:#6fce9b">${(sm.recall_mean*100).toFixed(0)}%</b>`
      +`<div style="margin-top:4px">${sm.note}</div>`;
  }
  const svg=document.getElementById('btChart'); const W=680,H=200,L=44,R=16,T=16,B=42;
  if(!ys.length){svg.innerHTML='';return;}
  const maxd=Math.max(...ys.map(y=>P.backtest[y].metrics.days_ge35),4);
  const bw=(W-L-R)/ys.length;
  let g='';
  ys.forEach((y,i)=>{
    const k=P.backtest[y].metrics;
    const h1=(k.days_ge35/maxd)*(H-T-B), h2=(k.days_ge38/maxd)*(H-T-B);
    const x=L+i*bw+bw*0.18;
    g+=`<rect x="${x}" y="${H-B-h1}" width="${bw*0.29}" height="${h1}" fill="#e8b93b" rx="2"/>`;
    g+=`<rect x="${x+bw*0.33}" y="${H-B-h2}" width="${bw*0.29}" height="${h2}" fill="#e5484d" rx="2"/>`;
    g+=`<text x="${x+bw*0.3}" y="${H-B-h1-4}" fill="#94a0b1" font-size="10" text-anchor="middle">${k.days_ge35}</text>`;
    g+=`<text x="${x+bw*0.47}" y="${H-B-h2-4}" fill="#ff8b8f" font-size="10" text-anchor="middle">${k.days_ge38}</text>`;
    g+=`<text x="${x+bw*0.3}" y="${H-B+16}" fill="#94a0b1" font-size="11" text-anchor="middle">${y}</text>`;
  });
  g+=`<line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="#2a313b"/>`;
  g+=`<rect x="${L}" y="${T-4}" width="10" height="10" fill="#e8b93b"/><text x="${L+15}" y="${T+5}" fill="#94a0b1" font-size="11">≥35 ℃ 天数</text>`;
  g+=`<rect x="${L+120}" y="${T-4}" width="10" height="10" fill="#e5484d"/><text x="${L+135}" y="${T+5}" fill="#94a0b1" font-size="11">≥38 ℃ 天数</text>`;
  svg.innerHTML=g;
}

/* ---------------- AI 层 ---------------- */
function renderML(){
  const ml=P.ml.metrics; const box=document.getElementById('mlBox');
  if(!ml){box.innerHTML='<div class="evi">模型未训练。</div>';return;}
  const cm=ml.confusion_matrix||[];
  box.innerHTML=`
  <div class="msg" style="border-left-color:var(--violet)">
    <b>两层结构</b>：① 物理内核（地形降尺度 + 水量平衡）产出地块级微气候场与风险等级；
    ② 随机森林<b>代理模型</b>以物理内核结果为弱标签，学习「可观测信号 → 风险等级」的映射，
    用于快速推理、概率校准与特征归因，并可在拿到田间真实标签后直接再训练。
  </div>
  <div class="scroll"><table>
    <tr><th>弱监督样本数</th><td class="n">${ml.n_samples}</td>
        <th>类别分布</th><td class="n">${JSON.stringify(ml.class_balance)}</td></tr>
    <tr><th>对照弱标签准确率</th><td class="n">${(ml.test_accuracy*100).toFixed(1)}%</td>
        <th>宏平均 F1</th><td class="n">${ml.test_macro_f1}</td></tr>
    <tr><th>风险分回归 R²</th><td class="n">${ml.score_r2===undefined?'—':ml.score_r2}</td>
        <th>参数集合一致度</th><td class="n">${(ml.mean_ensemble_agreement*100).toFixed(0)}%</td></tr>
  </table></div>
  <div class="evi" style="margin-top:8px">
    <span class="pill">混淆矩阵</span>${cm.map(r=>`<code>[${r.join(' , ')}]</code>`).join('')}
    <div style="margin-top:6px;color:#d9be8a">注意：${ml.labels_note}。该指标衡量的是"代理模型复现物理内核判断的能力"，
    不是田间预警精度；遥感特征（NDVI/NDRE/NDWI）接口已预留，待接入 GEE 导出后可一键启用。</div>
  </div>`;
  const svg=document.getElementById('impChart');
  const imp=P.ml.feature_importance.slice(0,14);
  const maxv=Math.max(...imp.map(i=>i[1]));
  const W=680, rowH=19, L=150, R=46;
  let g='';
  imp.forEach((it,i)=>{
    const y=8+i*rowH, w=(it[1]/maxv)*(W-L-R);
    g+=`<text x="${L-8}" y="${y+13}" fill="#94a0b1" font-size="11" text-anchor="end">${it[0]}</text>
        <rect x="${L}" y="${y+2}" width="${w}" height="12" rx="3" fill="#4c8dff" opacity="${0.45+0.55*(it[1]/maxv)}"/>
        <text x="${L+w+6}" y="${y+13}" fill="#6b7688" font-size="10">${it[1].toFixed(3)}</text>`;
  });
  svg.setAttribute('viewBox',`0 0 ${W} ${Math.max(120, imp.length*rowH+12)}`);
  svg.innerHTML=g;
}

/* ---------------- 周报 / 来源 / 页脚 ---------------- */
function renderBulletin(){
  const m=M();
  const bs=m.layer_stats.blindspot||{};
  const lines=[`【一垄一候 · 梅家坞茶园气候风险周报】${m.window}`,'',`· 本期结论：${m.summary.text}`];
  if(m.summary.n_high){
    lines.push('· 高风险地块：'+m.advices.filter(a=>a.risk_class===2).map(a=>`${a.unit_id}(${a.dominant})`).join('、'));
  }
  // 统一阈值盲区：三个互补事实拼成一句，任何情景下都可读
  const bsTxt=(function(){
    if(bs.mild_threshold_c===undefined) return '';
    const mp=fmt(bs.mild_threshold_c,0), op=fmt(bs.onset_threshold_c,0);
    let s;
    if(!bs.silent_days){ s=`窗口内区域气象场本身已越 ${mp} ℃，不存在『区域沉默』的日数。`; }
    else if(bs.silent_hot_days){
      s=`区域最高温未达 ${mp} ℃ 的 ${bs.silent_days} 天里，仍有 ${bs.silent_hot_days} 天出现 ≥${mp} ℃ 的地块`
        +`（最热日受影响像元占 ${fmt(bs.hot_frac_max_pct,1)}%）。`;
    } else {
      s=`区域最高温未达 ${mp} ℃ 的 ${bs.silent_days} 天里，地块级尚未越 ${mp} ℃。`;
    }
    if(bs.onset_silent_days&&bs.onset_hot_frac_max_pct>0){
      s+=`若收紧到度时累积起点 ${op} ℃，区域未越阈的 ${bs.onset_silent_days} 天里最热日已有 `
        +`${fmt(bs.onset_hot_frac_max_pct,1)}% 的茶园像元越阈。`;
    }
    return s+'统一阈值管理会在这些地块上滞后。';
  })();
  lines.push(`· 空间差异：适茶带内最大坡向温差为 ${fmt(m.layer_stats.max_aspect_diff,1)} ℃`
    +`（${m.layer_stats.max_aspect_pair?m.layer_stats.max_aspect_pair.join(' vs '):'坡向对比'}），`
    +`全域最高气温极差 ${fmt(m.layer_stats.microclimate_spread_c,1)} ℃。区域气象报出的最高温为 `
    +`${fmt(m.layer_stats.region_tmax_peak,1)} ℃，而最热地块的降尺度最高温达 ${fmt(bs.cell_peak_c,1)} ℃`
    +`（高出 ${fmt(bs.exceedance_c,1)} ℃）。`+bsTxt);
  const ys=Object.keys(P.backtest).filter(k=>/^\d+$/.test(k)).sort();
  if(ys.length){
    lines.push('· 气候背景：'+ys.map(y=>`${y} 年夏季高温日（≥35 ℃）${P.backtest[y].metrics.days_ge35} 天`).join('、')+'。');
  }
  if(P.backtest.summary){
    const sb=P.backtest.summary;
    lines.push(`· 预警性能：五年平均在胁迫致害前 ${sb.lead_days_mean} 天发出预警，`
      +`逐日预警准确率 ${(sb.precision_mean*100).toFixed(0)}%。`);
  }
  lines.push('· 对消费者意味着什么：你杯中这一年的龙井，其香气与鲜爽度，与夏季茶树是否经历高温干旱直接相关。');
  lines.push('');
  lines.push('（数据来源：Open-Meteo 区域气象再分析 + Mapzen/AWS 公开 DEM；本简报为气候风险评估，不构成茶叶品质或价格判断。）');
  document.getElementById('bulletin').textContent=lines.join('\n');
}

function renderSources(){
  document.getElementById('srcBox').innerHTML=
    '<ul class="src">'+P.data_sources.map(d=>
      `<li><b>${d.name}</b>（${d.provider}）— ${d.use}　<span class="pill">${d.access}</span></li>`).join('')
    +`<li><b>站点</b>：${P.site.village}｜${P.site.elev_note}</li>`
    +'</ul>';
  document.getElementById('limBox').innerHTML=
    '<b>局限与待校准项（不回避）</b><ul>'+P.limits.map(l=>`<li>${l}</li>`).join('')+'</ul>';
  document.getElementById('footer').textContent=
    `${P.disclaimer}｜本页为单文件离线 Demo，风险网格为自绘数据可视化，区位底图由腾讯位置服务提供。`;
}

/* ---------------- 腾讯地图 ---------------- */
let tmap=null, circles=null, labels=null;
function initMap(){
  if(window.__TMAP_FAILED || typeof TMap==='undefined'){
    document.getElementById('tmap').style.display='none'; return;
  }
  try{
    document.getElementById('mapFallback').style.display='none';
    tmap=new TMap.Map('tmap',{center:new TMap.LatLng(...wgs2gcj(P.site.lat,P.site.lon)),
      zoom:14});
    circles=new TMap.MultiCircle({map:tmap, styles:{r:new TMap.CircleStyle({color:'#e5484d',showBorder:true,borderColor:'#ffffff',borderWidth:2})}});
    labels=new TMap.MultiLabel({map:tmap, styles:{l:new TMap.LabelStyle({color:'#ffffff',size:11,offset:{x:0,y:-4}})}});
  }catch(e){ document.getElementById('tmap').style.display='none'; }
}
function updateMap(){
  if(!circles) return;
  const m=M();
  const geos=[], labs=[];
  P.units.forEach(u=>{
    const st=m.units[u.unit_id]; const c=wgs2gcj(u.center_lat,u.center_lon);
    const rad=[70,120,190][st.cls];
    geos.push({id:u.unit_id,styleId:'r',center:new TMap.LatLng(c[0],c[1]),radius:rad,
      color:LC[st.cls]});
    labs.push({id:u.unit_id,styleId:'l',position:new TMap.LatLng(c[0],c[1]),
      content:`${u.unit_id} ${u.aspect_label.slice(0,2)} ${LV[st.cls]}`});
  });
  try{ circles.setGeometries(geos); labels.setGeometries(labs); }catch(e){}
  const tb=document.querySelector('#coordTable tbody'); tb.innerHTML='';
  P.units.forEach(u=>{
    const st=m.units[u.unit_id];
    tb.innerHTML+=`<tr><td><b>${u.unit_id}</b></td><td>${u.aspect_label}</td>
      <td class="n">${u.slope_mean_deg}°</td><td class="n">${u.center_lat.toFixed(5)}</td>
      <td class="n">${u.center_lon.toFixed(5)}</td></tr>`;
  });
}

function renderAll(){
  renderModes(); renderKPIs(); renderMap(); renderUnits(); renderChart();
  renderBacktest(); renderML(); renderBulletin(); renderSources(); updateMap();
}

document.getElementById('layerTabs').addEventListener('click',e=>{
  if(!e.target.dataset.layer)return;
  S.layer=e.target.dataset.layer;
  [...e.currentTarget.children].forEach(b=>b.classList.toggle('on',b===e.target));
  renderMap();
});
document.getElementById('chartTabs').addEventListener('click',e=>{
  if(!e.target.dataset.s)return;
  S.series=e.target.dataset.s;
  [...e.currentTarget.children].forEach(b=>b.classList.toggle('on',b===e.target));
  renderChart();
});
renderHeader();
initMap();
renderAll();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
def write_report(pack: dict, path) -> None:
    """生成 Markdown 分析报告（可直接作为参赛文档的技术附件）。"""
    L = []
    A = L.append
    A(f"# 一垄一候 · 梅家坞茶园微气候风险运行报告\n")
    A(f"生成时间：{pack['generated_at']}　|　站点：{pack['site']['name']}"
      f"（{pack['site']['village']}）\n")
    g = pack["grid"]
    A("## 1 研究区与网格\n")
    A(f"- 网格分辨率：{g['res_m']} m；范围 {g['shape'][1]} × {g['shape'][0]} 像元")
    A(f"- 高程：{g['dem_stats']['elev_min']:.0f} – {g['dem_stats']['elev_max']:.0f} m"
      f"（均值 {g['dem_stats']['elev_mean']:.0f} m）")
    A(f"- 平均坡度：{g['dem_stats']['slope_mean']:.1f}°（最大 {g['dem_stats']['slope_max']:.1f}°）")
    A(f"- 经纬度范围：{g['extent']['lon_min']:.4f}–{g['extent']['lon_max']:.4f} E，"
      f"{g['extent']['lat_min']:.4f}–{g['extent']['lat_max']:.4f} N\n")

    A("## 2 管理单元（候选地块）\n")
    A("| 单元 | 面积(ha) | 坡向 | 平均坡度 | 海拔(m) | 天空可视因子 | 阳坡占比 |")
    A("|---|---|---|---|---|---|---|")
    for u in pack["units"]:
        A(f"| {u['unit_id']} | {u['area_ha']} | {u['aspect_label']} | {u['slope_mean_deg']}° | "
          f"{u['elev_mean_m']} | {u['svf_mean']} | {u['frac_sunny']} |")
    A("")

    A("## 3 各模式风险评估结果\n")
    for key, m in pack["modes"].items():
        A(f"### 3.{key} {m['label']}\n")
        A(f"- 窗口：{m['window']}")
        A(f"- 总体结论：{m['summary']['text']}")
        ls = m["layer_stats"]
        q = ls.get("aspect_quadrant_tmax", {})
        qd = ls.get("aspect_quadrant_tday", {})
        A(f"- 坡向日最高气温均值：" + "｜".join(f"{k} {v} ℃" for k, v in q.items()))
        A(f"- 坡向昼间（10–16 时）均温：" + "｜".join(f"{k} {v} ℃" for k, v in qd.items()))
        A(f"- 阳坡 − 阴坡：日最高气温差 **{ls['sunny_minus_shady']} ℃**，"
          f"昼间均温差 **{ls['sunny_minus_shady_daytime']} ℃**")
        A(f"- 最大坡向温差：**{ls['max_aspect_diff']} ℃**（{ ' vs '.join(ls['max_aspect_pair']) if ls.get('max_aspect_pair') else '—' }）；"
          f"适茶带全域极差 **{ls['microclimate_spread_c']} ℃**")
        A(f"- 区域气象最高温 {ls['region_tmax_peak']} ℃；地块级最高温 {ls['tmax_peak_max']} ℃")
        A(f"- **统一阈值盲区**：{blindspot_text(ls)}\n")
        A("| 优先级 | 单元 | 风险 | 主导胁迫 | 风险分 | 置信度 | 最关键行动 |")
        A("|---|---|---|---|---|---|---|")
        for a in m["advices"]:
            act = a["actions"][0]["action"] if a["actions"] else "常规管理"
            A(f"| {a['priority']} | {a['unit_id']} | {a['risk_level']} | {a['dominant']} | "
              f"{a['risk_score']} | {a['confidence']} | {act} |")
        A("")

    A("## 4 逐地块行动建议（默认模式）\n")
    dm = pack["modes"][pack["default_mode"]]
    A(f"默认展示模式：**{dm['label']}**\n")
    for a in dm["advices"]:
        A(f"**{a['unit_id']}（{a['aspect_label']}，{a['area_ha']} ha，海拔 {a['elev_mean_m']} m）"
          f"—— {a['risk_level']}风险，主导胁迫 {a['dominant']}**\n")
        A(f"{a['headline']}\n")
        A("判定依据：" + "；".join(a["evidence"]) + "\n")
        if a["actions"]:
            for i, x in enumerate(a["actions"], 1):
                A(f"{i}. **{x['action']}**｜时间：{x['when']}｜做法：{x['how']}｜理由：{x['why']}")
        else:
            A("当前风险低，无需应急措施。")
        A("")

    if pack.get("backtest"):
        A("## 5 历史灾情回溯验证\n")
        A("| 年份 | 最高气温 | ≥35℃日 | ≥38℃日 | ≥40℃日 | 夏季降水 | 土壤缺水累积 | "
          "最大连续无雨 | 缺水日数 | 高风险日 | 热害提前量 | 预警准确率 | 漏报率 |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for y, r in sorted((k, v) for k, v in pack["backtest"].items() if isinstance(k, int)):
            k = r["metrics"]
            de = r["day_eval"]
            lead = "—" if r["lead_days_heat"] is None else f"{r['lead_days_heat']} 天"
            A(f"| {y} | {k['tmax_peak']} ℃ | {k['days_ge35']} | {k['days_ge38']} | {k['days_ge40']} | "
              f"{k['precip_total_mm']} mm | {k['cum_drought_severity']} | {k['max_dry_streak']} 天 | "
              f"{k['days_sm_stressed']} | {r['high_risk_days']} | {lead} | "
              f"{de['precision']*100:.0f}% | {de['miss_rate']*100:.0f}% |")
        A("")
        sm = pack["backtest"].get("summary", {})
        if sm:
            A(f"**五年汇总**：平均预警提前量 **{sm['lead_days_mean']} 天**"
              f"（最长 {sm['lead_days_max']} 天），逐日预警准确率均值 "
              f"**{sm['precision_mean']*100:.0f}%**，对胁迫日的召回率均值 "
              f"**{sm['recall_mean']*100:.0f}%**。")
            A(f"\n> {sm['note']}\n")
        for y, r in sorted((k, v) for k, v in pack["backtest"].items() if isinstance(k, int)):
            if r["events"]:
                A(f"- **{y} 年预警事件段**：" + "；".join(
                    f"{e['start']}~{e['end']}（{e['days']} 天，峰值 {e['peak_cls']} 级，"
                    f"最高温 {e['peak_tmax']} ℃）" for e in r["events"][:4]))
            if r["damage_day_heat"]:
                A(f"- {y} 年累积热害度时跨过致害阈值：**{r['damage_day_heat']}**"
                  + (f"，首次预警提前 **{r['lead_days_heat']} 天**"
                     if r["lead_days_heat"] is not None else ""))
        A("")

    if pack["ml"].get("metrics"):
        A("## 6 机器学习层评估\n")
        ml = pack["ml"]["metrics"]
        A(f"- 弱监督样本数：{ml['n_samples']}；类别分布：{ml['class_balance']}")
        A(f"- 相对弱标签的测试准确率：{ml['test_accuracy']*100:.1f}%，宏平均 F1：{ml['test_macro_f1']}")
        A(f"- 风险分回归 R²：{ml.get('score_r2', '—')}；参数集合一致度："
          f"{ml['mean_ensemble_agreement']*100:.0f}%")
        A(f"- 说明：{ml['labels_note']}")
        A("\n特征重要性（前 10）：\n")
        A("| 特征 | 重要性 |")
        A("|---|---|")
        for n, v in pack["ml"]["feature_importance"][:10]:
            A(f"| {n} | {v} |")
        A("")

    A("## 7 数据来源\n")
    for d in pack["data_sources"]:
        A(f"- **{d['name']}**（{d['provider']}）— {d['use']}｜获取方式：{d['access']}")
    A("")

    A("## 8 局限与下一步\n")
    for l in pack["limits"]:
        A(f"- {l}")
    A("")
    A(f"> {pack['disclaimer']}")

    path.write_text("\n".join(L), encoding="utf-8")
