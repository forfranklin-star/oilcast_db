"""中期（三个月 / 66 个交易日）预测：向量自回归 VAR。

内生变量（全部平稳化）：
    dlwti     WTI 日对数收益
    dxy_chg   美元指数日变化
    u10y_chg  10Y 美债日变化
    gpr_chg   地缘风险日变化
    fed_chg   降息预期日变化
VAR 同时刻画油价与宏观变量的领先/滞后联动；预测区间用**保留同期相关性的
残差块 bootstrap**：每一步整行抽取历史残差冲击，递归模拟数百条路径，
再按情景概率叠加年化漂移，得到分布与涨跌概率。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils import get_logger, seeded_rng

LOG = get_logger(__name__)


class MidTermVAR:
    COLS = ["dlwti", "dxy_chg", "u10y_chg", "gpr_chg", "fed_chg"]

    def fit(self, prices: pd.DataFrame, macro: pd.DataFrame, max_lags: int = 4):
        from statsmodels.tsa.api import VAR

        df = pd.DataFrame(index=prices.index)
        df["dlwti"] = np.log(prices["wti"].ffill()).diff()
        m = macro.reindex(prices.index).ffill()
        df["dxy_chg"] = m["dxy"].pct_change()
        df["u10y_chg"] = m["us10y"].diff()
        df["gpr_chg"] = m["gpr_index"].diff().fillna(0) / 50 if "gpr_index" in m else 0.0
        df["fed_chg"] = m["fed_expectation"].diff().fillna(0)
        df = df.dropna()
        # 稳健化：按列做 0.5%/99.5% 分位缩尾，防止极端值导致 VAR 伴随矩阵爆炸
        lo = df.quantile(0.005)
        hi = df.quantile(0.995)
        self.data = df.clip(lower=lo, upper=hi, axis=1)
        self.last_price = float(prices["wti"].ffill().iloc[-1])

        try:
            sel = VAR(self.data).select_order(maxlags=max_lags)
            cand = int(np.clip(sel.aic, 1, max_lags))
        except Exception:
            cand = 2
        # statsmodels 稳定性判据：过程特征根全部位于单位圆【外】(is_stable)；
        # 若 AIC 阶数不稳定则逐级降阶兜底
        lags = cand
        while lags >= 1:
            fitted = VAR(self.data).fit(lags)
            if fitted.is_stable(verbose=False):
                break
            lags -= 1
        self.model = VAR(self.data).fit(max(lags, 1))
        self.lags = max(lags, 1)
        self.resid = self.model.resid
        LOG.info("VAR 拟合完成，lags=%d，样本=%d，稳定=%s，最近特征根模长=%.3f",
                 self.lags, len(self.data), self.model.is_stable(verbose=False),
                 float(np.min(np.abs(self.model.roots))))
        return self

    def predict(self, future_dates: pd.DatetimeIndex,
                scenario_probs: Dict[str, float], scenarios_cfg,
                n_sims: int = 500) -> PathResultLike:
        steps = len(future_dates)
        rng = seeded_rng()
        last_y = self.data.values[-self.lags:]
        coefs = self.model.coefs          # (lags, k, k)
        intercept = getattr(self.model, "intercept", np.zeros(len(self.COLS)))
        resid = self.resid.values
        k = len(self.COLS)

        # 每个情景的日漂移（仅作用于油价方程，第 0 列）
        daily_drift = {name: np.log1p(float(s["drift_annual"])) / 252
                       for name, s in scenarios_cfg.items()}
        names = list(scenario_probs)
        probs = np.array([scenario_probs[n] for n in names], dtype=float)
        probs = probs / probs.sum()    # 归一化保险
        expected_drift = float((probs * np.array([daily_drift[n] for n in names])).sum())
        draws = rng.choice(len(names), size=n_sims, p=probs)

        sim_endpoints, sim_paths = np.zeros(n_sims), np.zeros((n_sims, steps))
        mean_path = np.zeros(steps)
        history = last_y.copy()
        for t in range(steps):                       # 条件均值路径（含期望情景漂移）
            yhat = intercept.copy()
            for L in range(self.lags):
                yhat += coefs[L] @ history[-(L + 1)]
            yhat[0] += expected_drift
            history = np.vstack([history, yhat])
            mean_path[t] = yhat[0]
        mean_cum = np.cumsum(mean_path)

        for s in range(n_sims):
            hist = last_y.copy()
            drift = daily_drift[names[draws[s]]]
            cum = np.zeros(steps)
            for t in range(steps):
                yhat = intercept.copy()
                for L in range(self.lags):
                    yhat += coefs[L] @ hist[-(L + 1)]
                shock = resid[rng.integers(0, len(resid))]
                yhat = yhat + shock
                yhat[0] += drift                    # 情景漂移
                hist = np.vstack([hist, yhat])
                cum[t] = cum[t - 1] + yhat[0] if t else yhat[0]
                # 数值保险：单条路径相对当前价偏离超 ±ln(3.5) 即截断，
                # 防止罕见爆炸路径污染分位数（不改变绝大多数路径）
                cum[t] = np.clip(cum[t], np.log(0.3), np.log(3.5))
            sim_paths[s] = cum
            sim_endpoints[s] = cum[-1]

        q = np.quantile(sim_paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        path = pd.DataFrame({
            "mean": self.last_price * np.exp(mean_cum),
            "q05": self.last_price * np.exp(q[0]), "q25": self.last_price * np.exp(q[1]),
            "q50": self.last_price * np.exp(q[2]), "q75": self.last_price * np.exp(q[3]),
            "q95": self.last_price * np.exp(q[4]),
        }, index=future_dates)

        prob_up = float((sim_endpoints > 0).mean())
        endpoint = {
            "target_date": pd.Timestamp(future_dates[-1]).strftime("%Y-%m-%d"),
            "mean": round(float(path["mean"].iloc[-1]), 2),
            "q05": round(float(path["q05"].iloc[-1]), 2),
            "q25": round(float(path["q25"].iloc[-1]), 2),
            "q75": round(float(path["q75"].iloc[-1]), 2),
            "q95": round(float(path["q95"].iloc[-1]), 2),
            "pct_mean": round((float(path["mean"].iloc[-1]) / self.last_price - 1) * 100, 2),
            "prob_up": round(prob_up, 3), "prob_down": round(1 - prob_up, 3),
        }
        return {"path": path, "endpoint": endpoint}


# 仅用于类型提示的别名（避免循环导入 short_term 的 dataclass）
PathResultLike = dict
