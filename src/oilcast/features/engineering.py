"""特征工程。

设计原则：
1. **无未来泄漏**：第 t 行特征只使用 ≤t 的信息；标签为 t+1..t+h 累计收益；
2. **方向统一**：所有因素特征处理成"数值越大 → 越利多油价"，
   这样模型系数/重要性聚合后符号可直接解释；
3. **两层结构**：细粒度特征进模型，按 FACTOR_GROUPS 聚合为九大因素权重，
   与 config.prior_weights 的键一一对应。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# 细特征 -> 九大因素（技术面特征归为 _technical_，不参与因素权重展示）
FACTOR_GROUPS: Dict[str, list] = {
    "supply_disruption": ["ev_supply_5d"],
    "geopolitical_risk": ["gpr_level_z", "gpr_chg_5d", "ev_geo_5d"],
    "usd_index": ["dxy_chg_5d_dir"],
    "us_treasury_10y": ["us10y_chg_5d_dir"],
    "cpi_surprise": ["cpi_surprise_dir", "ev_cpi_5d"],
    "jobs_surprise": ["nfp_surprise_dir", "ev_jobs_5d"],
    "fed_policy_expectation": ["fed_exp_level", "fed_exp_chg", "ev_fed_5d"],
    "demand_outlook": ["demand_chg_5d", "ev_demand_5d"],
    "institutional_view": ["inst_bias", "ev_view_5d"],
    "_technical_": ["ret_1d", "ret_5d", "ma_gap", "vol_20"],
}

THEME_TO_COL = {
    "supply_disruption": "ev_supply_5d",
    "geopolitical_risk": "ev_geo_5d",
    "demand_outlook": "ev_demand_5d",
    "cpi_surprise": "ev_cpi_5d",
    "jobs_surprise": "ev_jobs_5d",
    "fed_policy_expectation": "ev_fed_5d",
    "institutional_view": "ev_view_5d",
}


def _zscore(s: pd.Series, win: int = 120) -> pd.Series:
    mu = s.rolling(win, min_periods=20).mean()
    sd = s.rolling(win, min_periods=20).std().replace(0, np.nan)
    return (s - mu) / sd


def aggregate_events(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """把事件表按主题聚合到日频，再做 5 日滚动强度和。"""
    out = pd.DataFrame(0.0, index=index, columns=list(THEME_TO_COL.values()))
    if events is None or events.empty:
        return out
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    daily = ev.groupby(["date", "theme"])["intensity"].sum().unstack(fill_value=0)
    daily = daily.reindex(index).fillna(0.0)
    for theme, col in THEME_TO_COL.items():
        if theme in daily.columns:
            out[col] = daily[theme].rolling(5, min_periods=1).sum()
    return out


def institutional_bias(views: pd.DataFrame, price_index: pd.Series,
                       index: pd.DatetimeIndex) -> pd.Series:
    """机构情绪：目标价相对现价的平均偏离 + 净看涨比例，合成 -1~1 指标。"""
    bias = pd.Series(0.0, index=index)
    if views is None or views.empty:
        return bias
    v = views.copy()
    v["date"] = pd.to_datetime(v["date"])
    score_map = {"看涨": 1.0, "看跌": -1.0, "中性": 0.0}
    v["score"] = v["stance"].map(score_map).fillna(0)
    daily = v.set_index("date")["score"].sort_index()
    # 30 个自然日滚动均值，前向填充到价格工作日索引
    roll = daily.rolling("30D").mean().reindex(index, method="ffill").fillna(0)
    return roll.clip(-1, 1).rename("inst_bias")


def build_features(prices: pd.DataFrame, macro: pd.DataFrame,
                   events: pd.DataFrame, views: pd.DataFrame,
                   target: str = "wti") -> pd.DataFrame:
    """返回对齐后的特征矩阵（index 与 prices 相同）。

    target: 预测目标列名（wti / brent / diesel），技术面特征随目标切换，
    九大外生因素对所有目标保持一致。
    """
    idx = prices.index
    feats = pd.DataFrame(index=idx)

    # ---------- 目标价格自身的技术面特征 ----------
    log_t = np.log(prices[target].ffill())
    feats["ret_1d"] = log_t.diff(1)
    feats["ret_5d"] = log_t.diff(5)
    feats["ma_gap"] = (log_t - log_t.rolling(20, min_periods=5).mean())
    feats["vol_20"] = log_t.diff().rolling(20, min_periods=5).std()

    m = macro.reindex(idx).ffill()
    # ---------- 九大因素（统一为"利多为正"） ----------
    # 美元：走强利空 → 取负向变化
    feats["dxy_chg_5d_dir"] = -m["dxy"].pct_change(5) * 100
    # 美债10Y：收益率上行压制风险资产 → 取负
    feats["us10y_chg_5d_dir"] = -m["us10y"].diff(5)
    # CPI：相对自身趋势超预期 → 紧缩利空 → 取负 z 分
    feats["cpi_surprise_dir"] = -_zscore(m["cpi_yoy"]).fillna(0)
    # 非农意外：超强劲 → 紧缩利空 → 取负
    feats["nfp_surprise_dir"] = -_zscore(m["nonfarm_surprise"]).fillna(0)
    # 降息预期：越偏降息越利多
    feats["fed_exp_level"] = m["fed_expectation"].fillna(0)
    feats["fed_exp_chg"] = m["fed_expectation"].diff(5).fillna(0)
    # 需求景气变化
    feats["demand_chg_5d"] = m["demand_proxy"].diff(5).fillna(0)
    # GPR 地缘风险：水平 z 分 + 5 日变化
    gpr = m["gpr_index"] if "gpr_index" in m.columns else None
    if gpr is None or gpr.notna().sum() < 30:
        # 真实 GPR 缺失时，用事件强度自建（缩放到 ~100 量纲）
        gpr = 100 + aggregate_events(events, idx).sum(axis=1) * 20
    feats["gpr_level_z"] = _zscore(gpr).fillna(0)
    feats["gpr_chg_5d"] = gpr.diff(5).fillna(0) / 50

    # 事件主题聚合
    ev_agg = aggregate_events(events, idx)
    for col in ev_agg.columns:
        feats[col] = ev_agg[col]
    # 机构情绪
    feats["inst_bias"] = institutional_bias(views, prices["wti"], idx)
    feats["ev_view_5d"] = feats.get("ev_view_5d", 0.0)

    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return feats


def make_supervised(features: pd.DataFrame, target_price: pd.Series,
                    horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """构造 (X_t, y_t)：用 t 日特征预测未来 horizon 日累计对数收益。"""
    fwd_ret = np.log(target_price).shift(-horizon) - np.log(target_price)
    y = fwd_ret.loc[features.index]
    valid = y.replace([np.inf, -np.inf], np.nan).notna()
    return features.loc[valid], y.loc[valid]


def all_feature_columns() -> list:
    cols = []
    for v in FACTOR_GROUPS.values():
        cols.extend(v)
    return cols
