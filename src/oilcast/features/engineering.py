"""特征工程。

设计原则：
1. **无未来泄漏**：第 t 行特征只使用 ≤t 的信息；标签为 t+1..t+h 累计收益；
2. **方向统一**：所有因素特征处理成"数值越大 → 越利多油价"，
   模型系数/重要性聚合后符号可直接解释；
3. **缺失不造假**：源不可达导致的缺失保留 NaN，绝不 fillna(0) 伪装成"中性"；
   训练时只使用真实观测齐全的样本，整列缺失的因素在权重层显式剔除；
4. **两层结构**：细粒度特征进模型，按 FACTOR_GROUPS 聚合为九大因素权重。
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
# 因素整列非空率低于该阈值 → 判定该因素数据缺失，不参与权重归一化
MIN_FACTOR_COVERAGE = 0.20


def _zscore(s: pd.Series, win: int = 120) -> pd.Series:
    mu = s.rolling(win, min_periods=20).mean()
    sd = s.rolling(win, min_periods=20).std().replace(0, np.nan)
    return (s - mu) / sd


def aggregate_events(events: pd.DataFrame, index: pd.DatetimeIndex,
                     source_available: bool = True) -> pd.DataFrame:
    """事件按主题聚合到日频，再做 5 日滚动强度和。

    事件源不可达时返回全 NaN（"没抓到"不等于"当天零事件"）；
    源可达但当天确无事件，才是真实的 0。
    """
    cols = list(THEME_TO_COL.values())
    if not source_available:
        return pd.DataFrame(np.nan, index=index, columns=cols)
    out = pd.DataFrame(np.nan, index=index, columns=cols)
    if events is None or events.empty:
        return out
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    daily = ev.groupby(["date", "theme"])["intensity"].sum().unstack(fill_value=0)
    daily = daily.reindex(index).fillna(0.0)
    for theme, col in THEME_TO_COL.items():
        if theme in daily.columns:
            # 该主题至少真实出现过：无事件日记真实 0，滚动 5 日强度；
            # 从未出现的主题保持 NaN（没有该类信号 ≠ 该因素恒为 0 可建模）
            out[col] = daily[theme].rolling(5, min_periods=1).sum()
    return out


def institutional_bias(views: pd.DataFrame, index: pd.DatetimeIndex,
                       source_available: bool = True) -> pd.Series:
    """机构情绪：净看涨比例的 30 日滚动均值（-1~1）。源不可达返回全 NaN。"""
    if not source_available or views is None or views.empty:
        return pd.Series(np.nan if not source_available else 0.0, index=index, name="inst_bias")
    v = views.copy()
    v["date"] = pd.to_datetime(v["date"])
    v["score"] = v["stance"].map({"看涨": 1.0, "看跌": -1.0, "中性": 0.0}).fillna(0)
    daily = v.set_index("date")["score"].sort_index()
    roll = daily.rolling("30D").mean().reindex(index, method="ffill", limit=22)
    return roll.clip(-1, 1).rename("inst_bias")


def build_features(prices: pd.DataFrame, macro: pd.DataFrame,
                   events: pd.DataFrame, views: pd.DataFrame,
                   target: str = "wti",
                   events_available: bool = True,
                   views_available: bool = True) -> pd.DataFrame:
    """返回对齐后的特征矩阵；缺失保持 NaN，不做零值填充。"""
    idx = prices.index
    feats = pd.DataFrame(index=idx)

    # ---------- 目标价格技术面（仅允许节假日级 3 工作日短填充）----------
    log_t = np.log(prices[target].ffill(limit=3))
    feats["ret_1d"] = log_t.diff(1)
    feats["ret_5d"] = log_t.diff(5)
    feats["ma_gap"] = log_t - log_t.rolling(20, min_periods=5).mean()
    feats["vol_20"] = log_t.diff().rolling(20, min_periods=5).std()

    # 宏观列在采集层已完成节假日短填充与月频 vintage 对齐，这里不再无限 ffill
    m = macro.reindex(idx)

    feats["dxy_chg_5d_dir"] = -m["dxy"].pct_change(5, fill_method=None) * 100 if "dxy" in m else np.nan
    feats["us10y_chg_5d_dir"] = -m["us10y"].diff(5) if "us10y" in m else np.nan
    feats["cpi_surprise_dir"] = -_zscore(m["cpi_yoy"]) if "cpi_yoy" in m else np.nan
    feats["nfp_surprise_dir"] = _zscore(m["nonfarm_surprise"]) if "nonfarm_surprise" in m else np.nan
    # 非农超强劲→紧缩利空，方向取负
    feats["nfp_surprise_dir"] = -feats["nfp_surprise_dir"]
    if "fed_expectation" in m:
        feats["fed_exp_level"] = m["fed_expectation"]
        feats["fed_exp_chg"] = m["fed_expectation"].diff(5)
    else:
        feats["fed_exp_level"] = np.nan
        feats["fed_exp_chg"] = np.nan
    feats["demand_chg_5d"] = m["demand_proxy"].diff(5) if "demand_proxy" in m else np.nan

    # GPR：真实指数优先；不可达时用【真实事件】构造代理；两者皆无则保持缺失
    gpr = m["gpr_index"] if "gpr_index" in m else None
    if gpr is None or gpr.notna().sum() < 30:
        ev_proxy = aggregate_events(events, idx, source_available=events_available).sum(axis=1)
        if events_available:
            gpr = 100 + ev_proxy * 20
        else:
            gpr = pd.Series(np.nan, index=idx)
    feats["gpr_level_z"] = _zscore(gpr)
    feats["gpr_chg_5d"] = gpr.diff(5) / 50

    ev_agg = aggregate_events(events, idx, source_available=events_available)
    for col in ev_agg.columns:
        feats[col] = ev_agg[col]
    feats["inst_bias"] = institutional_bias(views, idx, source_available=views_available)
    if "ev_view_5d" not in feats:
        feats["ev_view_5d"] = ev_agg.get("ev_view_5d",
                                         pd.Series(np.nan if not views_available else 0.0, index=idx))

    # 只做 3 工作日短填充（衔接节假日），长缺口保留 NaN，绝不填 0 冒充中性
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill(limit=3)
    return feats


def factor_availability(features: pd.DataFrame) -> Dict[str, bool]:
    """逐因素判断真实数据覆盖率是否达标（整列缺失的因素不参与建模/权重）。"""
    avail = {}
    n = len(features)
    for factor, cols in FACTOR_GROUPS.items():
        if factor == "_technical_":
            continue
        present = [c for c in cols if c in features.columns]
        if not present:
            avail[factor] = False
            continue
        cov = features[present].notna().any(axis=1).mean() if n else 0.0
        avail[factor] = bool(cov >= MIN_FACTOR_COVERAGE)
    return avail


TECHNICAL_COLS = ["ret_1d", "ret_5d", "ma_gap", "vol_20"]


def make_supervised(features: pd.DataFrame, target_price: pd.Series,
                    horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """构造 (X_t, y_t)：用 t 日特征预测未来 horizon 日累计对数收益。

    样本有效性只要求【标签】与【技术面特征】真实非空；外生因素列允许 NaN——
    HistGradientBoosting 原生处理缺失，这样某因素暂时缺数据不会浪费其余真实样本，
    也不会靠填零制造样本。
    """
    fwd_ret = np.log(target_price).shift(-horizon) - np.log(target_price)
    y = fwd_ret.loc[features.index]
    tech = [c for c in TECHNICAL_COLS if c in features.columns]
    valid = (y.replace([np.inf, -np.inf], np.nan).notna()
             & features[tech].notna().all(axis=1))
    return features.loc[valid], y.loc[valid]


def all_feature_columns() -> list:
    cols = []
    for v in FACTOR_GROUPS.values():
        cols.extend(v)
    return cols
