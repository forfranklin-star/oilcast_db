"""滚动原点回测：量化短期模型的近期样本外表现，并与随机游走基准对比。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .short_term import ShortTermForecaster


def backtest_short(features: pd.DataFrame, price: pd.Series,
                   horizon: int = 10, n_origins: int = 8, gap: int = 5) -> dict:
    """在最近 n_origins 个 origin 上做样本外 10 日预测评估。

    返回 RMSE/MAE（百分比价格误差）、方向命中率与随机游走基准误差。
    """
    n = len(features)
    if n < 260:
        return {"available": False, "reason": f"样本不足({n}<260)"}
    origins = list(range(n - horizon - n_origins * gap, n - horizon, gap))
    errs, bench_errs, hits, pred_dirs = [], [], 0, 0
    for t in origins:
        tr_X, tr_p = features.iloc[:t], price.iloc[:t]
        try:
            # 回测只需要 direct 模型的点预测，残差区间/ARIMA 基准不必重复计算
            fc = ShortTermForecaster(horizon=horizon, window=180,
                                     compute_residuals=False).fit(tr_X, tr_p)
            x_now = tr_X.iloc[[-1]]
            cum_pred = float(fc.models[horizon].predict(x_now)[0])
        except Exception:
            continue
        actual_ret = float(np.log(price.iloc[t + horizon]) - np.log(price.iloc[t]))
        errs.append(abs(np.exp(cum_pred) - np.exp(actual_ret)) / np.exp(actual_ret) * 100)
        bench_errs.append(abs(1 - np.exp(actual_ret)) * 100)   # random walk: 预测=当前价
        if np.sign(cum_pred) == np.sign(actual_ret) and actual_ret != 0:
            hits += 1
        pred_dirs += 1
    if not errs:
        return {"available": False, "reason": "回测全部失败"}
    return {
        "available": True,
        "mae_pct": round(float(np.mean(errs)), 2),
        "rmse_pct": round(float(np.sqrt(np.mean(np.square(errs)))), 2),
        "benchmark_mae_pct": round(float(np.mean(bench_errs)), 2),
        "direction_accuracy": round(hits / pred_dirs, 3),
        "n_origins": pred_dirs,
        "horizon_td": horizon,
    }
