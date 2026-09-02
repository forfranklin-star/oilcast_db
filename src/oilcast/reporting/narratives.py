"""把模型输出转成中文文字解读：每个数字都来自上游计算，不写空话。"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

FACTOR_CN = {
    "supply_disruption": "供给中断与OPEC+产量政策",
    "geopolitical_risk": "地缘政治风险",
    "usd_index": "美元指数",
    "us_treasury_10y": "美国10年期国债收益率",
    "cpi_surprise": "美国通胀(CPI)超预期程度",
    "jobs_surprise": "非农就业数据",
    "fed_policy_expectation": "美联储降息/加息预期",
    "demand_outlook": "全球需求前景",
    "institutional_view": "机构观点与目标价调整",
}


def pct_word(x: float) -> str:
    if x > 0.05:
        return "上涨"
    if x < -0.05:
        return "下跌"
    return "基本持平"


def trend_narrative(prices: pd.Series, name: str, unit: str) -> str:
    p = prices.dropna()
    cur = float(p.iloc[-1])
    def chg(k):
        return (cur / float(p.iloc[-1 - k]) - 1) * 100 if len(p) > k else np.nan
    c5, c20, c60 = chg(5), chg(20), chg(60)
    vol_ann = float(p.pct_change().tail(20).std() * np.sqrt(252) * 100)
    hi, lo = float(p.tail(252).max()), float(p.tail(252).min())
    pos = (cur - lo) / max(hi - lo, 1e-9) * 100
    txt = (f"{name}最新收于 {cur:.2f} {unit}，近5个交易日{pct_word(c5)}{abs(c5):.2f}%、"
           f"近20日{pct_word(c20)}{abs(c20):.2f}%、近60日{pct_word(c60)}{abs(c60):.2f}%；"
           f"近20日年化波动率约 {vol_ann:.1f}%。当前价格处于近一年区间 "
           f"[{lo:.2f}, {hi:.2f}] 的 {pos:.0f}% 分位。")
    return txt


def forecast_narrative(ep: dict, name: str, unit: str, horizon_cn: str) -> str:
    direction = "上涨" if ep["pct_mean"] > 0 else "下跌"
    prob = ep["prob_up"] * 100 if ep["pct_mean"] >= 0 else ep["prob_down"] * 100
    txt = (f"未来{horizon_cn}，模型对{name}的预测均值为 {ep['mean']:.2f} {unit}"
           f"（较当前{direction} {abs(ep['pct_mean']):.2f}%），"
           f"80%概率区间 [{ep['q10'] if 'q10' in ep else ep['q25']}, {ep['q75']}]，"
           f"95%概率区间 [{ep['q05']}, {ep['q95']}]；"
           f"方向概率：看涨 {ep['prob_up']*100:.0f}% / 看跌 {ep['prob_down']*100:.0f}%，"
           f"模型对{('上行' if ep['pct_mean'] >= 0 else '下行')}方向赋予的置信度约 {prob:.0f}%。")
    return txt


def weights_narrative(weights: pd.DataFrame) -> str:
    top = weights.head(3)
    parts = []
    for _, r in top.iterrows():
        parts.append(f"{FACTOR_CN.get(r['factor'], r['factor'])}（权重 {r['weight']*100:.1f}%）")
    leader = FACTOR_CN.get(weights.iloc[0]["factor"], weights.iloc[0]["factor"])
    return ("当前主导油价的前三大因素依次为 " + "、".join(parts) +
            f"；其中「{leader}」是边际定价的核心变量，"
            "权重由随机森林特征重要性与LASSO系数融合、再向人工先验收缩后得到，"
            "并随每日新增数据自适应更新。")


def events_narrative(events: pd.DataFrame, top_n: int = 5) -> List[str]:
    if events is None or events.empty:
        return ["最近一周未捕捉到达到强度阈值的重要事件。"]
    ev = events.head(top_n)
    out = []
    for _, r in ev.iterrows():
        impact = float(r.get("est_price_impact", 0) or 0)
        direction = "利多" if impact > 0 else ("利空" if impact < 0 else "中性")
        out.append(
            f"{pd.Timestamp(r['date']).strftime('%m-%d')}｜{r['title']}"
            f"（主题：{FACTOR_CN.get(r['theme'], r['theme'])}，强度 {float(r['intensity'])*100:.0f}%，"
            f"估算{direction}油价约 {abs(impact):.2f} 美元/桶）")
    return out


def scenario_narrative(long_ep: dict) -> str:
    p = long_ep["scenario_probs"]
    label = {"bull": "高油价（供给冲击）", "base": "基准（供需再平衡）", "bear": "低油价（需求衰退）"}
    seg = "、".join(f"{label[k]} {v*100:.0f}%" for k, v in p.items())
    anchor = long_ep.get("institution_anchor")
    extra = f"主要机构目标价中位数约 {anchor:.1f} 美元/桶，可作为模型分布的外部锚点。" if anchor else ""
    return (f"长期（12个月）情景概率：{seg}；模型分布均值 {long_ep['mean']:.2f}，"
            f"95%区间 [{long_ep['q05']}, {long_ep['q95']}]。{extra}")


def backtest_narrative(bt: dict) -> Optional[str]:
    if not bt or not bt.get("available"):
        return None
    beat = "优于" if bt["mae_pct"] < bt["benchmark_mae_pct"] else "暂未跑赢"
    return (f"近期滚动回测（{bt['n_origins']} 个样本外原点、{bt['horizon_td']} 交易日视野）："
            f"平均绝对误差 {bt['mae_pct']}%，方向命中率 {bt['direction_accuracy']*100:.0f}%，"
            f"相对随机游走基准（{bt['benchmark_mae_pct']}%）{beat}。")


def source_narrative(provenance: Dict[str, str]) -> str:
    return "数据来源标注：" + "；".join(f"{k}={v}" for k, v in provenance.items()) + \
        "。simulated/estimated 表示当前为合成或成本估算数据，接入真实数据源后自动替换。"
