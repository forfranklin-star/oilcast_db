"""短期（两周 / 10 个交易日）预测。

主模型：Direct multi-step 梯度提升回归 —— 对每个预测步长 h 单独训练一个模型，
目标是未来 h 日累计对数收益（外生变量即特征工程的多因素矩阵）。
区间：滚动原点（rolling-origin）样本外残差的经验分位数，避免用 in-sample
残差造成的过度乐观。
基准：ARIMA(2,0,2) 拟合日对数收益，作为无外生变量的时间序列对照。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

from ..utils import get_logger

LOG = get_logger(__name__)
RESIDUAL_HORIZONS = (2, 5, 10)


@dataclass
class PathResult:
    path: pd.DataFrame           # date, mean,q05,q25,q50,q75,q95
    endpoint: dict               # 期末均值/区间/涨跌概率
    benchmark_endpoint: Optional[dict] = None


class ShortTermForecaster:
    def __init__(self, horizon: int = 10, window: int = 180,
                 compute_residuals: bool = True) -> None:
        self.horizon = horizon
        self.window = window
        self.compute_residuals = compute_residuals
        self.models: Dict[int, HistGradientBoostingRegressor] = {}
        self.resid_quantiles: Dict[int, np.ndarray] = {}
        self.resid_std: Dict[int, float] = {}

    # ------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, price: pd.Series) -> "ShortTermForecaster":
        X = X.tail(max(self.window * 3, 400))
        price = price.loc[X.index].ffill()
        log_p = np.log(price)
        self._train_direct(X, log_p)
        if self.compute_residuals:
            self._rolling_residuals(X, log_p)
            self._arima_benchmark(log_p)
        return self

    def _train_direct(self, X: pd.DataFrame, log_p: pd.Series) -> None:
        for h in range(1, self.horizon + 1):
            y = (log_p.shift(-h) - log_p).dropna()
            Xh = X.loc[y.index]
            model = HistGradientBoostingRegressor(
                max_depth=4, max_iter=250, learning_rate=0.05,
                min_samples_leaf=15, random_state=42)
            model.fit(Xh, y)
            self.models[h] = model

    def _rolling_residuals(self, X: pd.DataFrame, log_p: pd.Series) -> None:
        """滚动原点样本外残差：origin 每 4 个交易日取一个，回看 window 训练。"""
        n = len(X)
        # 残差锚点步长不超过预测 horizon，保证 t+h 不会越界
        hs = tuple(h for h in RESIDUAL_HORIZONS if h <= self.horizon) \
            or tuple(range(1, self.horizon + 1))
        max_h = max(hs)
        origins = list(range(self.window, n - max_h, 5))[-12:]   # 最多 12 个 origin
        resid: Dict[int, list] = {h: [] for h in hs}
        for t in origins:
            tr_X, tr_log = X.iloc[:t], log_p.iloc[:t]
            for h in hs:
                y = (tr_log.shift(-h) - tr_log).dropna()
                m = HistGradientBoostingRegressor(
                    max_depth=4, max_iter=150, learning_rate=0.05,
                    min_samples_leaf=15, random_state=42)
                m.fit(tr_X.loc[y.index], y)
                pred = m.predict(X.iloc[[t]])[0]
                actual = float(log_p.iloc[t + h] - log_p.iloc[t])
                resid[h].append(actual - pred)
        # 插值补齐全部 h；样本不足以做滚动原点时，用日收益正态分位 ×√h 兜底
        self.ret_std = float(log_p.diff().std())
        z = {"q05": -1.645, "q25": -0.674, "q75": 0.674, "q95": 1.645}
        anchor_h = np.array(hs)
        for h in range(1, self.horizon + 1):
            if origins:
                qs, stds = [], []
                for q in (0.05, 0.25, 0.75, 0.95):
                    vals = [np.quantile(resid[hh], q) for hh in anchor_h]
                    qs.append(float(np.interp(h, anchor_h, vals)))
                stds = [np.std(resid[hh], ddof=1) for hh in anchor_h]
                self.resid_quantiles[h] = np.array(qs)
                self.resid_std[h] = float(np.interp(h, anchor_h, stds))
            else:
                sd = self.ret_std * np.sqrt(h)
                self.resid_quantiles[h] = np.array([z["q05"] * sd, z["q25"] * sd,
                                                    z["q75"] * sd, z["q95"] * sd])
                self.resid_std[h] = float(sd)

    def _arima_benchmark(self, log_p: pd.Series) -> None:
        """ARIMA 基准：阶数由高到低自动降级，收敛警告视为失败并重试更简模型。"""
        import warnings
        from statsmodels.tsa.arima.model import ARIMA
        rets = log_p.diff().dropna().tail(250)
        for order in ((2, 0, 2), (1, 0, 1), (1, 0, 0)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(rets, order=order).fit()
                if not getattr(model, "mle_retvals", {}).get("converged", True):
                    continue
                fc = model.forecast(steps=self.horizon)
                self.arima_cumret = float(fc.cumsum().iloc[-1])
                return
            except Exception as exc:
                LOG.warning("ARIMA%s 基准拟合失败：%s", order, exc)
        LOG.warning("全部 ARIMA 阶数均未收敛，跳过基准对比")
        self.arima_cumret = None

    # --------------------------------------------------------- predict
    def predict(self, latest_X: pd.DataFrame, last_price: float,
                future_dates: pd.DatetimeIndex) -> PathResult:
        x_now = latest_X.iloc[[-1]]
        rows = []
        for h, dt in enumerate(future_dates, start=1):
            # 点预测向随机游走收缩（James-Stein 思想）：小样本 ML 远端外推容易过激，
            # 0.7 的收缩系数在回测 MAE 与方向命中率之间更稳健；区间仍由样本外残差决定
            cum = 0.7 * float(self.models[h].predict(x_now)[0])
            q05, q25, q75, q95 = cum + self.resid_quantiles[h]
            rows.append({
                "date": dt, "mean": last_price * np.exp(cum),
                "q50": last_price * np.exp(cum),
                "q05": last_price * np.exp(q05), "q25": last_price * np.exp(q25),
                "q75": last_price * np.exp(q75), "q95": last_price * np.exp(q95),
                "_cumret": cum,
            })
        path = pd.DataFrame(rows).set_index("date")

        last = path.iloc[-1]
        std = self.resid_std[self.horizon]
        prob_up = float(norm.cdf(last["_cumret"] / max(std, 1e-6)))
        endpoint = {
            "target_date": pd.Timestamp(path.index[-1]).strftime("%Y-%m-%d"),
            "mean": round(float(last["mean"]), 2),
            "q05": round(float(last["q05"]), 2), "q25": round(float(last["q25"]), 2),
            "q75": round(float(last["q75"]), 2), "q95": round(float(last["q95"]), 2),
            "pct_mean": round((float(last["mean"]) / last_price - 1) * 100, 2),
            "prob_up": round(prob_up, 3), "prob_down": round(1 - prob_up, 3),
        }
        bench = None
        if self.arima_cumret is not None:
            bench = {"mean": round(last_price * np.exp(self.arima_cumret), 2),
                     "pct_mean": round((np.exp(self.arima_cumret) - 1) * 100, 2)}
        return PathResult(path=path.drop(columns=["_cumret"]), endpoint=endpoint,
                          benchmark_endpoint=bench)
