"""把模型输出转成中文文字解读：每个数字都来自上游真实计算，不写空话；
数据不可用时明确说明，不编造趋势。"""
from __future__ import annotations

from typing import List, Optional

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
STATUS_CN = {"ok": "可用", "stale": "已过期", "insufficient": "样本不足",
             "unavailable": "不可用"}


def pct_word(x: float) -> str:
    if pd.isna(x):
        return "数据不足"
    if x > 0.05:
        return "上涨"
    if x < -0.05:
        return "下跌"
    return "基本持平"


def trend_narrative(prices: pd.Series, name: str, unit: str) -> str:
    p = prices.dropna()
    cur = float(p.iloc[-1])
    obs = p.index[-1].strftime("%Y-%m-%d")

    def chg(k):
        return (cur / float(p.iloc[-1 - k]) - 1) * 100 if len(p) > k else np.nan
    c5, c20, c60 = chg(5), chg(20), chg(60)
    vol_ann = float(p.pct_change().tail(20).std() * np.sqrt(252) * 100)
    hi, lo = float(p.tail(252).max()), float(p.tail(252).min())
    pos = (cur - lo) / max(hi - lo, 1e-9) * 100
    return (f"截至真实观测日 {obs}，{name}收于 {cur:.2f} {unit}，"
           f"近5个交易日{pct_word(c5)}{abs(c5):.2f}%、近20日{pct_word(c20)}{abs(c20):.2f}%、"
           f"近60日{pct_word(c60)}{abs(c60):.2f}%；近20日年化波动率约 {vol_ann:.1f}%，"
           f"价格处于近一年真实区间 [{lo:.2f}, {hi:.2f}] 的 {pos:.0f}% 分位。")


def forecast_narrative(ep: dict, name: str, unit: str, horizon_cn: str) -> str:
    direction = "上涨" if ep["pct_mean"] > 0 else "下跌"
    prob = ep["prob_up"] * 100 if ep["pct_mean"] >= 0 else ep["prob_down"] * 100
    return (f"未来{horizon_cn}，模型对{name}的预测均值为 {ep['mean']:.2f} {unit}"
            f"（较观测日{direction} {abs(ep['pct_mean']):.2f}%），"
            f"50%概率区间 [{ep['q25']}, {ep['q75']}]，95%概率区间 [{ep['q05']}, {ep['q95']}]；"
            f"方向概率：看涨 {ep['prob_up']*100:.0f}% / 看跌 {ep['prob_down']*100:.0f}%，"
            f"模型对{('上行' if ep['pct_mean'] >= 0 else '下行')}方向置信度约 {prob:.0f}%。"
            f"预测仅基于截至观测日的真实数据，不构成投资建议。")


def weights_narrative(weights: pd.DataFrame) -> str:
    usable = weights.dropna(subset=["weight"])
    if usable.empty:
        return "当前没有任何因素具备充分真实数据，本期不计算因素权重。"
    top = usable.head(3)
    parts = [f"{FACTOR_CN.get(r['factor'], r['factor'])}（权重 {r['weight']*100:.1f}%）"
             for _, r in top.iterrows()]
    leader = FACTOR_CN.get(usable.iloc[0]["factor"], usable.iloc[0]["factor"])
    missing = [FACTOR_CN.get(f, f) for f in weights[weights["weight"].isna()]["factor"]]
    tail = f"；因素【{'、'.join(missing)}】因真实数据缺失未参与权重归一化" if missing else ""
    return ("当前主导油价的前三大因素依次为 " + "、".join(parts) +
            f"；其中「{leader}」是边际定价的核心变量。权重由随机森林重要性与"
            f"LASSO系数融合、向人工先验收缩并与历史权重EMA平滑得到{tail}。")


def events_narrative(events: pd.DataFrame, top_n: int = 5) -> List[str]:
    if events is None or events.empty:
        return ["本期事件源不可达或未捕捉到达到强度阈值的真实事件，不列举模拟事件。"]
    out = []
    for _, r in events.head(top_n).iterrows():
        impact = float(r.get("est_price_impact", 0) or 0)
        direction = "利多" if impact > 0 else ("利空" if impact < 0 else "中性")
        out.append(
            f"{pd.Timestamp(r['date']).strftime('%m-%d')}｜{r['title']}"
            f"（主题：{FACTOR_CN.get(r['theme'], r['theme'])}，强度 {float(r['intensity'])*100:.0f}%，"
            f"规则估算{direction}约 {abs(impact):.2f} 美元/桶；来源：{r.get('source','')}）")
    return out


def scenario_narrative(long_ep: dict) -> str:
    p = long_ep["scenario_probs"]
    label = {"bull": "高油价（供给冲击）", "base": "基准（供需再平衡）", "bear": "低油价（需求衰退）"}
    seg = "、".join(f"{label[k]} {v*100:.0f}%" for k, v in p.items())
    anchor = long_ep.get("institution_anchor")
    extra = f"真实抓取的机构目标价中位数约 {anchor:.1f} 美元/桶，作为外部锚点并列展示。" if anchor else ""
    return (f"长期（12个月）情景概率：{seg}；模型分布均值 {long_ep['mean']:.2f}，"
            f"95%区间 [{long_ep['q05']}, {long_ep['q95']}]。{extra}")


def backtest_narrative(bt: dict) -> Optional[str]:
    if not bt or not bt.get("available"):
        return None
    beat = "优于" if bt["mae_pct"] < bt["benchmark_mae_pct"] else "暂未跑赢"
    return (f"近期滚动回测（{bt['n_origins']} 个样本外原点、{bt['horizon_td']} 交易日视野，"
            f"全部基于真实历史）：平均绝对误差 {bt['mae_pct']}%，方向命中率 "
            f"{bt['direction_accuracy']*100:.0f}%，相对随机游走基准（{bt['benchmark_mae_pct']}%）{beat}。")


def lineage_narrative(bundle) -> str:
    """逐字段说明真实来源、观测日期、质量门状态（替代旧的 provenance 说明）。"""
    parts = []
    for f, L in bundle.lineage.items():
        status = STATUS_CN.get(L.get("status"), L.get("status"))
        src = L.get("source_name", "")
        last = L.get("last_observed") or "无观测"
        parts.append(f"{f}[{status}|末次观测 {last}|{src}]")
    head = "演示模式：以下全部为合成数据，严禁用于真实判断。" if bundle.mode == "demo" \
        else "数据谱系（字段[状态|末次真实观测|来源]）："
    return head + "；".join(parts)
