"""生成自包含静态 HTML 报告（Plotly + 数据内联）。

数据诚实性呈现：
- 历史价格缺口 connectgaps=False（断开，不用线"补"出不存在的观测）；
- 每个当前值标注真实观测日期；不可用周期/标的显示原因而非假数字；
- 顶部数据谱系表逐字段列出来源、末次观测、样本数与质量门状态；
- demo 模式全程红色「合成数据」水印。
"""
from __future__ import annotations

import json
from typing import List, Optional

import shutil
from pathlib import Path

import plotly
import plotly.graph_objects as go
from jinja2 import Template

from ..config import get_config
from .narratives import FACTOR_CN, STATUS_CN


def ensure_plotly_asset() -> str:
    """把 plotly bundle 复制到 reports/assets，HTML 以相对路径引用。

    相比外网 CDN：国内/受限网络打开即渲染、无第三方追踪；相比逐份内联 4.5MB，
    30 份历史报告共享同一份 JS，仓库体积可控。返回相对 HTML 文件的引用路径。
    """
    src = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    assets = Path(get_config()["storage"]["latest_dir"]).parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    dst = assets / "plotly.min.js"
    if not dst.exists():
        shutil.copy(src, dst)
    return "../assets/plotly.min.js"   # latest/ 与 archive/ 同级，相对路径一致

HORIZON_CN = {"short": "短期·两周", "mid": "中期·三个月", "long": "长期·十二个月"}
H_COLOR = {"short": "#1f6feb", "mid": "#e08a00", "long": "#b03434"}
STATUS_COLOR = {"ok": "#2e7d32", "stale": "#e08a00", "insufficient": "#e08a00",
                "unavailable": "#b03434"}


def _is_ok(item: Optional[dict]) -> bool:
    return isinstance(item, dict) and item.get("status", "ok") == "ok" and "path" in item


def price_figure(report: dict, target: str, title: str, unit: str) -> go.Figure:
    hist = report["history"]
    hx = [r["date"] for r in hist]
    hy = [r.get(target) for r in hist]
    meta = report["current_meta"].get(target, {})
    fig = go.Figure()
    # 历史真实观测：缺口保持断开（connectgaps=False）
    fig.add_trace(go.Scatter(x=hx, y=hy, name="历史真实观测", connectgaps=False,
                             line=dict(color="#2c3e50", width=2)))
    anchor_x, anchor_y = meta.get("observed_date"), meta.get("value")
    for hz in ("short", "mid", "long"):
        item = report["forecasts"][hz].get(target)
        if not _is_ok(item):
            continue
        recs = item["path"]
        x = [anchor_x] + [r["date"] for r in recs]
        c = H_COLOR[hz]
        for lo_, hi_, alpha, lbl in (("q05", "q95", 0.08, "95%区间"),
                                     ("q25", "q75", 0.14, "50%区间")):
            yb = [anchor_y] + [r[hi_] for r in recs]
            ya = [anchor_y] + [r[lo_] for r in recs]
            fig.add_trace(go.Scatter(
                x=x + x[::-1], y=yb + ya[::-1],
                fill="toself", fillcolor=_hex_alpha(c, alpha), line=dict(width=0),
                hoverinfo="skip", showlegend=False, name=f"{HORIZON_CN[hz]}{lbl}"))
        fig.add_trace(go.Scatter(x=x, y=[anchor_y] + [r["mean"] for r in recs],
                                 name=f"{HORIZON_CN[hz]}预测均值",
                                 line=dict(color=c, width=2, dash="dash")))
    obs = f"，末次真实观测 {anchor_x}" if anchor_x else "（无真实观测）"
    if anchor_x is None:
        fig.add_annotation(text="该标的暂无可核验真实数据，按数据原则不展示预测",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=15, color="#b03434"))
        fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.update_layout(
        title=f"{title} 历史走势与多周期预测（{unit}）{obs}",
        height=430, margin=dict(l=50, r=20, t=56, b=40),
        plot_bgcolor="white", hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
        font=dict(family="-apple-system,Segoe UI,Microsoft YaHei", size=12))
    fig.update_xaxes(showgrid=True, gridcolor="#eef1f5")
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f5")
    return fig


def weights_figure(report: dict) -> Optional[go.Figure]:
    def _is_num(v):
        return v is not None and v == v  # 排除 None 与 float('nan')
    usable = sorted((r for r in report["weights"] if _is_num(r.get("weight"))),
                    key=lambda r: r["weight"])
    missing = [r for r in report["weights"] if not _is_num(r.get("weight"))]
    if not usable:
        return None
    y = [FACTOR_CN.get(r["factor"], r["factor"]) for r in usable]
    fig = go.Figure(go.Bar(
        x=[r["weight"] * 100 for r in usable], y=y, orientation="h",
        marker_color="#1f6feb",
        text=[f"{r['weight']*100:.1f}%" for r in usable], textposition="outside"))
    if missing:  # 缺失因素：灰色零长条 + “不可用”，绝不显示 nan%
        fig.add_trace(go.Bar(
            x=[0.0] * len(missing),
            y=[FACTOR_CN.get(r["factor"], r["factor"]) for r in missing],
            orientation="h", marker_color="#d9d9d9", showlegend=False,
            text=["不可用（无真实数据）"] * len(missing), textposition="outside"))
    fig.update_layout(title="多因素权重排名（灰色＝无真实数据、不参与归一化）",
                      height=380, margin=dict(l=210, r=150, t=56, b=30),
                      xaxis_title="归一化权重 (%)", plot_bgcolor="white",
                      font=dict(size=12), showlegend=False)
    return fig


def scenario_figure(report: dict) -> Optional[go.Figure]:
    item = report["forecasts"]["long"].get("wti")
    if not _is_ok(item):
        return None
    probs = item["endpoint"].get("scenario_probs", {})
    label = {"bull": "高油价", "base": "基准", "bear": "低油价"}
    fig = go.Figure(go.Pie(labels=[label.get(k, k) for k in probs],
                           values=[v * 100 for v in probs.values()],
                           marker=dict(colors=["#b03434", "#1f6feb", "#2e7d32"]),
                           hole=0.55, textinfo="label+percent"))
    fig.update_layout(title="长期情景概率", height=320, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _hex_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _fig_json(fig) -> str:
    return json.loads(fig.to_json()) if fig is not None else None


def _cards(report: dict) -> List[dict]:
    rows = []
    for hz in ("short", "mid", "long"):
        item = report["forecasts"][hz]["wti"]
        if not _is_ok(item):
            rows.append({"horizon": HORIZON_CN[hz], "unavailable": True,
                         "reason": item.get("reason", "真实数据不可用")})
            continue
        ep = item["endpoint"]
        rows.append({"horizon": HORIZON_CN[hz], "unavailable": False, "mean": ep["mean"],
                     "range95": f"{ep['q05']} ~ {ep['q95']}", "pct": ep["pct_mean"],
                     "prob_up": ep["prob_up"] * 100, "prob_down": ep["prob_down"] * 100,
                     "target": ep["target_date"]})
    return rows


def _lineage_rows(report: dict) -> List[dict]:
    out = []
    for f, L in report.get("lineage", {}).items():
        out.append({"field": f, "display": L.get("display", ""),
                    "status": L.get("status", "unavailable"),
                    "status_cn": STATUS_CN.get(L.get("status"), L.get("status")),
                    "source": L.get("source_name", ""), "url": L.get("url", ""),
                    "last": L.get("last_observed") or "—", "n": L.get("n_obs", 0),
                    "note": L.get("note", ""), "tried": L.get("tried_sources", "")})
    return out


def render_static_html(report: dict) -> str:
    figs = {
        "fig_wti": _fig_json(price_figure(report, "wti", "WTI原油", "美元/桶")),
        "fig_brent": _fig_json(price_figure(report, "brent", "布伦特原油", "美元/桶")),
        "fig_diesel": _fig_json(price_figure(report, "diesel", "国内0#柴油批发价", "元/吨")),
        "fig_weights": _fig_json(weights_figure(report)),
        "fig_scenario": _fig_json(scenario_figure(report)),
    }
    plotly_src = ensure_plotly_asset()
    return Template(TEMPLATE).render(
        report=report, FACTOR_CN=FACTOR_CN, figs_json=json.dumps(figs, ensure_ascii=False),
        cards=_cards(report), lineage_rows=_lineage_rows(report), plotly_src=plotly_src)


TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>多因素油价智能分析报告 · {{ report.report_date }}</title>
<script src="{{ plotly_src }}" charset="utf-8"></script>
<style>
 :root{--bd:#2c3e50;--blue:#1f6feb;--bg:#f5f7fa;--card:#fff;--mut:#6b7785;}
 *{box-sizing:border-box} body{margin:0;font-family:-apple-system,Segoe UI,"Microsoft YaHei",sans-serif;
   background:var(--bg);color:#1f2733;line-height:1.6}
 .wrap{max-width:1180px;margin:0 auto;padding:24px}
 header.top{background:linear-gradient(120deg,#173a5e,#2c5f8a);color:#fff;padding:28px 32px;border-radius:14px}
 header.top h1{margin:0 0 6px;font-size:26px} .mut{color:var(--mut);font-size:13px}
 header.top .mut{color:#c9d8e8}
 .demo-banner{background:#b03434;color:#fff;font-weight:700;text-align:center;padding:10px;
   border-radius:10px;margin-bottom:14px;letter-spacing:1px}
 .grid{display:grid;gap:16px;margin:20px 0}
 .cards{grid-template-columns:repeat(3,1fr)}
 .card{background:var(--card);border-radius:12px;padding:18px 20px;box-shadow:0 1px 4px rgba(20,40,70,.08)}
 .card h3{margin:0 0 8px;font-size:15px;color:var(--bd)}
 .big{font-size:30px;font-weight:700;color:var(--bd)} .up{color:#b03434}.down{color:#2e7d32}
 .kv{display:flex;justify-content:space-between;font-size:13px;color:var(--mut);padding:2px 0}
 .unavail{background:#fbf1f1;border:1px dashed #b03434;border-radius:10px;padding:14px;color:#8a2b2b;font-size:13px}
 .obsline{font-size:13px;color:var(--mut);margin:6px 0 0}
 .two{grid-template-columns:2fr 1fr}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{padding:8px 10px;border-bottom:1px solid #edf0f4;text-align:left;vertical-align:top}
 th{background:#f0f4f9;color:var(--bd);font-weight:600}
 .narr{background:#eef4fb;border-left:4px solid var(--blue);padding:10px 14px;border-radius:0 8px 8px 0;
   margin:8px 0;font-size:14px}
 h2{font-size:19px;margin:26px 0 10px;color:var(--bd);border-left:5px solid var(--blue);padding-left:10px}
 .probbar{display:flex;height:8px;border-radius:6px;overflow:hidden;margin-top:6px}
 .probbar .u{background:#b03434}.probbar .d{background:#2e7d32}
 .st{font-weight:700;padding:1px 8px;border-radius:10px;color:#fff;font-size:12px}
 footer{color:var(--mut);font-size:12px;margin:26px 0 10px}
 a{color:var(--blue)}
 @media(max-width:860px){.cards,.two{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
 {% if report.mode == 'demo' %}
 <div class="demo-banner">⚠ 演示模式：本报告全部数据为人工合成，非真实观测，严禁用于任何真实判断 ⚠</div>
 {% endif %}
 <header class="top">
   <h1>多因素油价智能分析与预测报告{% if report.mode == 'demo' %}（演示·合成数据）{% endif %}</h1>
   <div class="mut">报告日期 {{ report.report_date }} ｜ 生成于 {{ report.generated_at }}（北京时间）｜
     数据模式：{{ '演示合成' if report.mode=='demo' else '严格真实（缺失不补齐）' }}</div>
 </header>

 <h2>数据谱系与质量门（每个数字都可追溯到来源与观测日期）</h2>
 <div class="card" style="overflow-x:auto">
 <table>
   <tr><th>字段</th><th>含义</th><th>状态</th><th>命中来源</th><th>末次观测</th><th>样本数</th><th>数据源优先级尝试链（✓采用 / ✗失败原因）</th></tr>
   {% for L in lineage_rows %}
   <tr>
     <td>{{ L.field }}</td><td>{{ L.display }}</td>
     <td><span class="st" style="background:{{ {'ok':'#2e7d32','stale':'#e08a00','insufficient':'#e08a00','unavailable':'#b03434'}[L.status] }}">{{ L.status_cn }}</span></td>
     <td>{% if L.url %}<a href="{{ L.url }}" target="_blank" rel="noopener">{{ L.source }}</a>{% else %}{{ L.source }}{% endif %}</td>
     <td>{{ L.last }}</td><td>{{ L.n }}</td><td class="mut" style="font-size:12px">{{ L.tried or L.note }}</td>
   </tr>
   {% endfor %}
 </table>
 </div>

 <h2>当前真实观测</h2>
 <div class="grid cards">
 {% for t in ['wti','brent','diesel'] %}
  <div class="card">
    <h3>{{ report.names[t] }}（{{ report.units[t] }}）</h3>
    {% if report.current_meta[t].status == 'ok' %}
      <div class="big">{{ report.current_meta[t].value }}</div>
      <div class="obsline">真实观测日：{{ report.current_meta[t].observed_date }} ｜ {{ report.current_meta[t].source }}</div>
    {% else %}
      <div class="unavail">不可用（{{ report.current_meta[t].status }}）：{{ report.current_meta[t].reason }}</div>
    {% endif %}
  </div>
 {% endfor %}
 </div>

 <h2>预测卡片（WTI 原油）</h2>
 <div class="grid cards">
 {% for c in cards %}
  <div class="card">
    <h3>{{ c.horizon }}{% if not c.unavailable %} <span class="mut">→ {{ c.target }}</span>{% endif %}</h3>
    {% if c.unavailable %}
      <div class="unavail">暂不预测<br>{{ c.reason }}</div>
    {% else %}
      <div class="big {{ 'up' if c.pct>=0 else 'down' }}">{{ c.mean }} <span style="font-size:15px">美元/桶</span></div>
      <div class="{{ 'up' if c.pct>=0 else 'down' }}" style="font-size:14px">
        {{ '▲' if c.pct>=0 else '▼' }} {{ c.pct }}%（相对观测日）</div>
      <div class="kv"><span>95%概率区间</span><span>{{ c.range95 }}</span></div>
      <div class="kv"><span>看涨/看跌</span><span>{{ '%.0f'|format(c.prob_up) }}% / {{ '%.0f'|format(c.prob_down) }}%</span></div>
      <div class="probbar"><div class="u" style="width:{{c.prob_up}}%"></div><div class="d" style="width:{{c.prob_down}}%"></div></div>
    {% endif %}
  </div>
 {% endfor %}
 </div>

 <h2>价格走势与预测区间（缺口=当时无真实观测，不作连线）</h2>
 <div class="card" id="fig_wti"></div>
 <div class="card" id="fig_brent" style="margin-top:16px"></div>
 <div class="card" id="fig_diesel" style="margin-top:16px"></div>

 <h2>模型解释：当前哪些因素主导油价</h2>
 <div class="grid two">
   <div class="card" id="fig_weights"></div>
   <div class="card" id="fig_scenario"></div>
 </div>
 <div class="narr">{{ report.narratives.weights }}</div>
 <div class="narr">{{ report.narratives.long_wti }}</div>

 <h2>近期关键事件与量化影响（真实新闻，规则打分）</h2>
 <div class="card">
 {% if report.events %}
 <table>
   <tr><th>日期</th><th>事件</th><th>来源</th><th>因素主题</th><th>强度</th><th>估算影响(美元/桶)</th></tr>
   {% for e in report.events[:20] %}
   <tr><td>{{ e.date }}</td><td>{% if e.url %}<a href="{{e.url}}" target="_blank" rel="noopener">{{ e.title }}</a>{% else %}{{ e.title }}{% endif %}</td>
       <td>{{ e.source }}</td><td>{{ FACTOR_CN.get(e.theme, e.theme) }}</td>
       <td>{{ '%.0f'|format(e.intensity*100) }}%</td>
       <td class="{{ 'up' if e.est_price_impact>0 else 'down' }}">{{ '%+.2f'|format(e.est_price_impact) }}</td></tr>
   {% endfor %}
 </table>
 {% else %}<div class="unavail">本期事件源不可达或无有效真实条目，按数据原则不列举任何模拟事件。</div>{% endif %}
 </div>
 {% for t in report.narratives.events %}<div class="narr">{{ t }}</div>{% endfor %}

 <h2>机构观点（真实抽取）</h2>
 <div class="card">
 {% if report.views %}
 <table>
   <tr><th>日期</th><th>机构</th><th>WTI目标价</th><th>方向</th><th>摘要</th></tr>
   {% for v in report.views[:12] %}
   <tr><td>{{ v.date }}</td><td>{{ v.institution }}</td><td>{{ v.target_wti if v.target_wti else '—' }}</td>
       <td>{{ v.stance }}</td><td>{{ v.note }}</td></tr>
   {% endfor %}
 </table>
 {% else %}<div class="unavail">机构观点源当前不可达，保持空缺，不生成模拟观点。</div>{% endif %}
 </div>

 <h2>走势与预测文字解读</h2>
 {% for t,name in [('wti','WTI原油'),('brent','布伦特原油'),('diesel','国内柴油')] %}
   <div class="narr"><b>{{ name}}｜历史：</b>{{ report.narratives.trends[t] }}</div>
 {% endfor %}
 {% for t in ['wti','brent','diesel'] %}
   <div class="narr"><b>两周：</b>{{ report.narratives.short[t] }}</div>
   <div class="narr"><b>三个月：</b>{{ report.narratives.mid[t] }}</div>
 {% endfor %}
 {% if report.narratives.backtest %}<div class="narr">{{ report.narratives.backtest }}</div>{% endif %}

 <h2>数据原则与免责声明</h2>
 <div class="card">
   <p class="mut">{{ report.narratives.sources }}</p>
   <p class="mut"><b>数据原则：</b>模型只建立在真实、可追溯、带观测日期的数据之上；
   缺失、过期或无法验证的数据一律保持缺失并在谱系表标明状态，绝不用插值、外推或合成值"补齐"。
   月频指标在两次发布之间沿用最近一次真实发布值，并在底层数据中保留其原始发布日期（vintage）。</p>
   <p class="mut">模型包括 Direct 多步梯度提升（短期）、VAR 向量自回归（中期）与情景蒙特卡洛（长期），
   权重由随机森林与 LASSO 融合、向人工先验收缩并跨日 EMA 自适应。预测区间反映历史波动与模型不确定性，
   不构成任何投资建议；第三方数据版权归原方所有。</p>
 </div>
 <footer>OilCast v1.1 · 每日 UTC 02:00（北京 10:00）由 GitHub Actions 自动更新</footer>
</div>
<script>
const FIGS = {{ figs_json | safe }};
for (const [id, fig] of Object.entries(FIGS)) {
  if (fig) Plotly.newPlot(id, fig.data, fig.layout, {responsive:true, displaylogo:false});
}
</script>
</body></html>
"""
