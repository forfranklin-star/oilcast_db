"""因素权重学习。

三路融合，兼顾数据驱动与可解释稳定性：
  1. RandomForest 特征重要性（非线性、捕捉交互）；
  2. LASSO 标准化系数绝对值（线性、稀疏，抑制噪声特征）；
  3. config.prior_weights 人工先验，做贝叶斯式收缩；
再与上一轮权重做指数平滑（EMA），实现"每天追加真实数据后渐进调整"。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from ..config import get_config
from ..features.engineering import FACTOR_GROUPS
from ..utils import get_logger

LOG = get_logger(__name__)


def _feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """RF 重要性 与 LASSO |系数| 等权平均，返回归一化细特征重要性。"""
    X = X.tail(1500)          # 控制训练成本
    y = y.loc[X.index]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                               max_depth=6, n_jobs=-1, random_state=42)
    rf.fit(Xs, y)
    imp_rf = pd.Series(rf.feature_importances_, index=X.columns)

    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso.fit(Xs, y)
    imp_lasso = pd.Series(np.abs(lasso.coef_), index=X.columns)
    if imp_lasso.sum() < 1e-8:    # LASSO 全稀疏时退化为等权
        imp_lasso = pd.Series(1 / X.shape[1], index=X.columns)

    fused = 0.6 * imp_rf / imp_rf.sum() + 0.4 * imp_lasso / imp_lasso.sum()
    return fused


def aggregate_to_factors(feature_imp: pd.Series) -> pd.Series:
    """细特征重要性聚合到九大因素（剔除技术面后归一化）。"""
    scores: Dict[str, float] = {}
    for factor, cols in FACTOR_GROUPS.items():
        if factor == "_technical_":
            continue
        scores[factor] = float(sum(feature_imp.get(c, 0.0) for c in cols))
    s = pd.Series(scores)
    if s.sum() < 1e-12:     # 模型信号过弱时均匀分配
        s = pd.Series(1 / len(s), index=s.index)
    return s / s.sum()


def learn_factor_weights(X: pd.DataFrame, y: pd.Series,
                         prev_weights: Optional[pd.Series] = None) -> pd.DataFrame:
    """主入口：返回 DataFrame[factor, weight, model_importance, prior]（权重和=1）。"""
    cfg = get_config()
    priors = pd.Series(dict(cfg["prior_weights"]), dtype=float)
    priors = priors / priors.sum()

    feature_imp = _feature_importance(X, y)
    model_w = aggregate_to_factors(feature_imp).reindex(priors.index).fillna(0)
    model_w = model_w / model_w.sum()

    # 先验收缩：数据不足/信号不稳时向人工先验收缩
    shrink = float(cfg["model"]["prior_shrinkage"])
    blended = shrink * model_w + (1 - shrink) * priors
    blended = blended / blended.sum()

    # 与历史权重 EMA，实现自适应平滑
    if prev_weights is not None:
        alpha = float(cfg["model"]["weight_ema_alpha"])
        prev = prev_weights.reindex(blended.index).fillna(0)
        if prev.sum() > 0:
            blended = alpha * prev / prev.sum() + (1 - alpha) * blended
            blended = blended / blended.sum()

    out = pd.DataFrame({
        "factor": blended.index,
        "weight": blended.round(4).values,
        "model_importance": model_w.round(4).values,
        "prior": priors.round(4).values,
    }).sort_values("weight", ascending=False).reset_index(drop=True)
    LOG.info("因素权重更新完成，主导因素：%s", out.iloc[0]["factor"])
    return out
