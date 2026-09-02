"""端到端冒烟测试：保证数据→特征→模型→报告主链路在干净环境可跑通。

运行：pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from oilcast.data_sources.collector import collect
from oilcast.data_sources.synthetic import build_synthetic_bundle
from oilcast.features.engineering import FACTOR_GROUPS, build_features, make_supervised
from oilcast.models.short_term import ShortTermForecaster
from oilcast.models.weights import learn_factor_weights
from oilcast.reporting.static_html import render_static_html

AS_OF = pd.Timestamp("2026-09-02 10:00")


@pytest.fixture(scope="module")
def bundle():
    return collect(as_of=AS_OF, prefer_live=False)


def test_synthetic_structure():
    b = build_synthetic_bundle(AS_OF, 730)
    assert set(["wti", "brent", "diesel"]).issubset(b["prices"].columns)
    assert len(b["prices"]) > 400
    assert b["macro"].notna().sum().sum() > 0
    assert not b["events"].empty and not b["views"].empty


def test_collector_offline_bundle(bundle):
    assert len(bundle.prices.columns) == 3
    for col in ["dxy", "us10y", "cpi_yoy", "nonfarm_surprise",
                "fed_expectation", "gpr_index"]:
        assert col in bundle.macro.columns
    assert all(v == "simulated" for v in bundle.provenance.values())


def test_features_clean(bundle):
    feats = build_features(bundle.prices, bundle.macro, bundle.events,
                           bundle.views, target="wti")
    assert not np.isinf(feats.values).any()
    # 九大因素 + 技术面特征都应存在
    for group_cols in FACTOR_GROUPS.values():
        for c in group_cols:
            assert c in feats.columns, c


def test_weights_sum_to_one(bundle):
    feats = build_features(bundle.prices, bundle.macro, bundle.events, bundle.views)
    X, y = make_supervised(feats, bundle.prices["wti"], 10)
    w = learn_factor_weights(X, y)
    assert abs(w["weight"].sum() - 1.0) < 1e-3
    assert len(w) == 9
    assert (w["weight"] >= 0).all()


def test_short_forecast_shape(bundle):
    feats = build_features(bundle.prices, bundle.macro, bundle.events, bundle.views)
    fc = ShortTermForecaster(horizon=3, window=120).fit(feats, bundle.prices["wti"])
    future = pd.bdate_range(feats.index[-1] + pd.Timedelta(days=1), periods=3)
    res = fc.predict(feats.iloc[[-1]], float(bundle.prices["wti"].iloc[-1]), future)
    assert len(res.path) == 3
    ep = res.endpoint
    assert ep["q05"] <= ep["mean"] <= ep["q95"]
    assert 0 <= ep["prob_up"] <= 1


def test_full_pipeline_and_html():
    """完整流水线（小历史窗口加速）+ HTML 渲染。"""
    from oilcast.config import get_config
    import oilcast.pipeline.main as pm
    # 用 730 天历史以保证模型样本足够
    report = pm.run(as_of=AS_OF, offline=True)
    assert report["report_date"] == "2026-09-02"
    for hz in ("short", "mid", "long"):
        assert "wti" in report["forecasts"][hz]
    html = render_static_html(report)
    assert "<html" in html and "Plotly" in html
