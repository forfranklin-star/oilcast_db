"""生成自包含静态 HTML 报告（Plotly.js CDN + 数据内联）。

GitHub Pages 直接托管 reports/latest/index.html 即可公开访问；
历史报告 reports/archive/YYYY-MM-DD.html 同步保留。
"""
from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template

from .narratives import FACTOR_CN

HORIZON_CN = {"short": "短期·两周", "mid": "中期·三个月", "long": "长期·十二个月"}
H_COLOR = {"short": "#1f6feb", "mid": "#e08a00", "long": "#b03434"}


def _series_from_records(records: List[dict], col: str):
    return [r["date"] for r in records], [r.get(col) for r in records]


def price_figure(report: dict, target: str, title: str, unit: str) -> go.Figure:
    hist = report["history"]
    hx, hy = _series_from_records(hist, target)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hx, y=hy, name="历史价格",
                             line=dict(color="#2c3e50", width=2)))
    anchor_x, anchor_y = hx[-1], hy[-1]
    for hz in ("short", "mid", "long"):
        pack = report["forecasts"][hz]
        if target not in pack:
            continue
        recs = pack[target]["path"]
        x = [anchor_x] + [r["date"] for r in recs]
        mean = [anchor_y] + [r["mean"] for r in recs]
        q05 = [anchor_y] + [r["q05"] for r in recs]
        q95 = [anchor_y] + [r["q95"] for r in recs]
        q25 = [anchor_y] + [r["q25"] for r in recs]
        q75 = [anchor_y] + [r["q75"] for r in recs]
        c = H_COLOR[hz]
        fig.add_trace(go.Scatter(x=x + x[::-1], y=q95 + q05[::-1],
                                 fill="toself", fillcolor=_hex_alpha(c, 0.08),
                                 line=dict(width=0), hoverinfo="skip",
                                 name=f"{HORIZON_CN[hz]} 95%区间", legendgroup=hz,
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x + x[::-1], y=q75 + q25[::-1],
                                 fill="toself", fillcolor=_hex_alpha(c, 0.14),
                                 line=dict(width=0), hoverinfo="skip",
                                 name=f"{HORIZON_CN[hz]} 50%区间", legendgroup=hz,
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=mean, name=f"{HORIZON_CN[hz]}预测均值",
                                 line=dict(color=c, width=2, dash="dash"),
                                 legendgroup=hz))
    fig.update_layout(
        title=f"{title} 历史走势与多周期预测（{unit}）",
        height=430, margin=dict(l=50, r=20, t=50, b=40),
        plot_bgcolor="white", hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
        font=dict(family="-apple-system,Segoe UI,Microsoft YaHei", size=12))
    fig.update_xaxes(showgrid=True, gridcolor="#eef1f5")
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f5")
    return fig


def weights_figure(report: dict) -> go.Figure:
    w = sorted(report["weights"], key=lambda r: r["weight"])
    y = [FACTOR_CN.get(r["factor"], r["factor"]) for r in w]
    fig = go.Figure(go.Bar(
        x=[r["weight"] * 100 for r in w], y=y, orientation="h",
        marker_color="#1f6feb",
        text=[f"{r['weight']*100:.1f}%" for r in w], textposition="outside"))
    fig.update_layout(title="多因素权重排名（模型学习 + 先验收缩 + 跨日EMA）",
                      height=380, margin=dict(l=200, r=40, t=50, b=30),
                      xaxis_title="归一化权重 (%)", plot_bgcolor="white",
                      font=dict(size=12))
    return fig


def scenario_figure(report: dict) -> go.Figure:
    probs = report["forecasts"]["long"]["wti"]["endpoint"].get("scenario_probs", {})
    label = {"bull": "高油价", "base": "基准", "bear": "低油价"}
    fig = go.Figure(go.Pie(
        labels=[label.get(k, k) for k in probs],
        values=[v * 100 for v in probs.values()],
        marker=dict(colors=["#b03434", "#1f6feb", "#2e7d32"]), hole=0.55,
        textinfo="label+percent"))
    fig.update_layout(title="长期情景概率", height=320,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _hex_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _fig_json(fig: go.Figure) -> str:
    return json.loads(fig.to_json())


def render_static_html(report: dict) -> str:
    figs = {
        "fig_wti": _fig_json(price_figure(report, "wti", "WTI原油", "美元/桶")),
        "fig_brent": _fig_json(price_figure(report, "brent", "布伦特原油", "美元/桶")),
        "fig_diesel": _fig_json(price_figure(report, "diesel", "国内0#柴油批发价", "元/吨")),
        "fig_weights": _fig_json(weights_figure(report)),
        "fig_scenario": _fig_json(scenario_figure(report)),
    }
    html = Template(TEMPLATE).render(
        report=report, HORIZON_CN=HORIZON_CN, FACTOR_CN=FACTOR_CN,
        figs_json=json.dumps(figs, ensure_ascii=False),
        cards=_cards(report),
    )
    return html


def _cards(report: dict) -> List[dict]:
    rows = []
    for hz in ("short", "mid", "long"):
        ep = report["forecasts"][hz]["wti"]["endpoint"]
        rows.append({
            "horizon": HORIZON_CN[hz], "mean": ep["mean"],
            "range95": f"{ep['q05']} ~ {ep['q95']}",
            "pct": ep["pct_mean"], "prob_up": ep["prob_up"] * 100,
            "prob_down": ep["prob_down"] * 100,
            "target": ep["target_date"],
        })
    return rows


TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>多因素油价智能分析报告 · {{ report.report_date }}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
 :root{--bd:#2c3e50;--blue:#1f6feb;--bg:#f5f7fa;--card:#fff;--mut:#6b7785;}
 *{box-sizing:border-box} body{margin:0;font-family:-apple-system,Segoe UI,"Microsoft YaHei",sans-serif;
   background:var(--bg);color:#1f2733;line-height:1.6}
 .wrap{max-width:1180px;margin:0 auto;padding:24px}
 header.top{background:linear-gradient(120deg,#173a5e,#2c5f8a);color:#fff;padding:28px 32px;border-radius:14px}
 header.top h1{margin:0 0 6px;font-size:26px} .mut{color:var(--mut);font-size:13px}
 header.top .mut{color:#c9d8e8}
 .badges{margin-top:10px} .badge{display:inline-block;background:rgba(255,255,255,.16);border-radius:20px;
   padding:3px 12px;font-size:12px;margin:3px 6px 3px 0}
 .grid{display:grid;gap:16px;margin:20px 0}
 .cards{grid-template-columns:repeat(3,1fr)}
 .card{background:var(--card);border-radius:12px;padding:18px 20px;box-shadow:0 1px 4px rgba(20,40,70,.08)}
 .card h3{margin:0 0 8px;font-size:15px;color:var(--bd)}
 .big{font-size:30px;font-weight:700;color:var(--bd)} .up{color:#b03434}.down{color:#2e7d32}
 .kv{display:flex;justify-content:space-between;font-size:13px;color:var(--mut);padding:2px 0}
 .two{grid-template-columns:2fr 1fr}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{padding:8px 10px;border-bottom:1px solid #edf0f4;text-align:left;vertical-align:top}
 th{background:#f0f4f9;color:var(--bd);font-weight:600}
 .narr{background:#eef4fb;border-left:4px solid var(--blue);padding:10px 14px;border-radius:0 8px 8px 0;
   margin:8px 0;font-size:14px}
 h2{font-size:19px;margin:26px 0 10px;color:var(--bd);border-left:5px solid var(--blue);padding-left:10px}
 .probbar{display:flex;height:8px;border-radius:6px;overflow:hidden;margin-top:6px}
 .probbar .u{background:#b03434}.probbar .d{background:#2e7d32}
 footer{color:var(--mut);font-size:12px;margin:26px 0 10px}
 @media(max-width:860px){.cards,.two{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
 <header class="top">
   <h1>多因素油价智能分析与预测报告</h1>
   <div class="mut">报告日期 {{ report.report_date }} ｜ 生成于 {{ report.generated_at }}（北京时间）</div>
   <div class="badges">
     {% for k,v in report.provenance.items() %}<span class="badge">{{ k }}：{{ v }}</span>{% endfor %}
   </div>
 </header>

 <h2>预测卡片（WTI 原油）</h2>
 <div class="grid cards">
 {% for c in cards %}
  <div class="card">
    <h3>{{ c.horizon }} <span class="mut">→ {{ c.target }}</span></h3>
    <div class="big {{ 'up' if c.pct>=0 else 'down' }}">{{ c.mean }} <span style="font-size:15px">美元/桶</span></div>
    <div class="{{ 'up' if c.pct>=0 else 'down' }}" style="font-size:14px">
      {{ '▲' if c.pct>=0 else '▼' }} {{ c.pct }}%（相对当前）</div>
    <div class="kv"><span>95%概率区间</span><span>{{ c.range95 }}</span></div>
    <div class="kv"><span>看涨/看跌</span><span>{{ '%.0f'|format(c.prob_up) }}% / {{ '%.0f'|format(c.prob_down) }}%</span></div>
    <div class="probbar"><div class="u" style="width:{{c.prob_up}}%"></div><div class="d" style="width:{{c.prob_down}}%"></div></div>
  </div>
 {% endfor %}
 </div>

 <h2>价格走势与预测区间</h2>
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

 <h2>近期关键事件与量化影响</h2>
 <div class="card">
 <table>
   <tr><th>日期</th><th>事件</th><th>来源</th><th>因素主题</th><th>强度</th><th>估算影响(美元/桶)</th></tr>
   {% for e in report.events[:20] %}
   <tr><td>{{ e.date }}</td><td>{{ e.title }}</td><td>{{ e.source }}</td>
       <td>{{ FACTOR_CN.get(e.theme, e.theme) }}</td>
       <td>{{ '%.0f'|format(e.intensity*100) }}%</td>
       <td class="{{ 'up' if e.est_price_impact>0 else 'down' }}">
         {{ '%+.2f'|format(e.est_price_impact) }}</td></tr>
   {% endfor %}
 </table>
 </div>
 {% for t in report.narratives.events %}<div class="narr">{{ t }}</div>{% endfor %}

 <h2>机构观点</h2>
 <div class="card"><table>
   <tr><th>日期</th><th>机构</th><th>WTI目标价</th><th>方向</th><th>摘要</th></tr>
   {% for v in report.views[:12] %}
   <tr><td>{{ v.date }}</td><td>{{ v.institution }}</td><td>{{ v.target_wti if v.target_wti else '—' }}</td>
       <td>{{ v.stance }}</td><td>{{ v.note }}</td></tr>
   {% endfor %}
 </table></div>

 <h2>走势与预测文字解读</h2>
 {% for t,name in [('wti','WTI原油'),('brent','布伦特原油'),('diesel','国内柴油')] %}
   <div class="narr"><b>{{ name}}｜历史：</b>{{ report.narratives.trends[t] }}</div>
 {% endfor %}
 {% for t in ['wti','brent','diesel'] %}
   <div class="narr"><b>两周：</b>{{ report.narratives.short[t] }}</div>
   <div class="narr"><b>三个月：</b>{{ report.narratives.mid[t] }}</div>
 {% endfor %}
 {% if report.narratives.backtest %}<div class="narr">{{ report.narratives.backtest }}</div>{% endif %}

 <h2>数据来源与免责声明</h2>
 <div class="card">
   <p class="mut">{{ report.narratives.sources }}</p>
   <p class="mut">本报告由多因素量化模型自动生成，模型包括 Direct 多步梯度提升（短期）、
   VAR 向量自回归（中期）与情景蒙特卡洛（长期），权重由随机森林与 LASSO 融合并跨日自适应。
   预测区间反映历史波动与模型不确定性，不构成任何投资建议；接入的第三方数据版权归原方所有。</p>
 </div>
 <footer>OilCast v1.0 · 每日 UTC 02:00（北京 10:00）由 GitHub Actions 自动更新</footer>
</div>

<script>
const FIGS = {{ figs_json | safe }};
for (const [id, fig] of Object.entries(FIGS)) {
  Plotly.newPlot(id, fig.data, fig.layout, {responsive:true, displaylogo:false});
}
</script>
</body></html>
"""
