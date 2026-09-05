"""端到端与数据原则测试。
运行：pytest -q
覆盖：demo 主链路、strict 数据原则（质量门/缺失不填零/样本不足显式失败/
月频 vintage 可追溯）、HTML 渲染。
"""
import numpy as np
import pandas as pd
import pytest

from oilcast.data_sources.collector import collect
from oilcast.data_sources.quality import (align_monthly_with_vintage,
                                          assess_daily, assess_monthly)
from oilcast.data_sources.synthetic import build_synthetic_bundle
from oilcast.features.engineering import (FACTOR_GROUPS, build_features,
                                          factor_availability, make_supervised)
from oilcast.models.errors import InsufficientData
from oilcast.models.short_term import ShortTermForecaster
from oilcast.models.weights import learn_factor_weights
from oilcast.reporting.static_html import render_static_html

AS_OF = pd.Timestamp("2026-09-02 10:00")


@pytest.fixture(scope="module")
def bundle():
    return collect(as_of=AS_OF, mode="demo")


# ----------------------------------------------------------- demo 主链路
def test_synthetic_structure():
    b = build_synthetic_bundle(AS_OF, 730)
    assert {"wti", "brent", "diesel"}.issubset(b["prices"].columns)
    assert len(b["prices"]) > 400
    assert not b["events"].empty and not b["views"].empty


def test_demo_bundle_is_labeled(bundle):
    assert bundle.mode == "demo"
    assert len(bundle.prices.columns) == 3
    for col in ["dxy", "us10y", "cpi_yoy", "nonfarm_surprise",
                "fed_expectation", "gpr_index"]:
        assert col in bundle.macro.columns
    # demo 必须逐字段打上合成水印，绝不允许伪装成真实源
    assert all("SYNTHETIC-DEMO" in L["source_name"] for L in bundle.lineage.values())


def test_features_clean(bundle):
    feats = build_features(bundle.prices, bundle.macro, bundle.events,
                           bundle.views, target="wti",
                           events_available=True, views_available=True)
    assert not np.isinf(feats.values).any()
    for group_cols in FACTOR_GROUPS.values():
        for c in group_cols:
            assert c in feats.columns, c


def test_weights_sum_to_one_and_flag(bundle):
    feats = build_features(bundle.prices, bundle.macro, bundle.events, bundle.views)
    X, y = make_supervised(feats, bundle.prices["wti"], 10)
    w = learn_factor_weights(X, y, available={f: True for f in FACTOR_GROUPS if f != "_technical_"})
    assert abs(w.dropna(subset=["weight"])["weight"].sum() - 1.0) < 1e-3
    assert len(w) == 9 and (w["available"]).all()


def test_short_forecast_shape(bundle):
    feats = build_features(bundle.prices, bundle.macro, bundle.events, bundle.views)
    fc = ShortTermForecaster(horizon=3, window=120).fit(feats, bundle.prices["wti"])
    future = pd.bdate_range(feats.index[-1] + pd.Timedelta(days=1), periods=3)
    res = fc.predict(feats.iloc[[-1]], float(bundle.prices["wti"].iloc[-1]), future)
    assert len(res.path) == 3
    ep = res.endpoint
    assert ep["q05"] <= ep["mean"] <= ep["q95"] and 0 <= ep["prob_up"] <= 1


def test_full_pipeline_demo_and_html():
    import oilcast.pipeline.main as pm
    report = pm.run(as_of=AS_OF, demo=True)
    assert report["mode"] == "demo"
    for hz in ("short", "mid", "long"):
        assert "wti" in report["forecasts"][hz]
    html = render_static_html(report)
    assert "<html" in html and "演示模式" in html


# ----------------------------------------------------------- strict 数据原则
def test_quality_gate_flags_stale_and_insufficient():
    idx = pd.bdate_range("2025-01-02", "2026-09-02")
    fresh = pd.Series(np.linspace(70, 80, len(idx)), index=idx)
    gate = {"min_obs": 120, "max_stale_bdays": 7}
    lin_ok = assess_daily("x", "x", fresh, AS_OF, "src", "u", gate)
    assert lin_ok.status == "ok"
    # 最后观测停在 8 月初 → stale
    stale = fresh.loc[:"2026-08-03"]
    assert assess_daily("x", "x", stale, AS_OF, "src", "u", gate).status == "stale"
    # 样本不足
    short = fresh.tail(60)
    assert assess_daily("x", "x", short, AS_OF, "src", "u", gate).status == "insufficient"
    # 源不可达
    assert assess_daily("x", "x", None, AS_OF, "src", "u", gate).status == "unavailable"


def test_monthly_vintage_is_traceable():
    # 月频发布值对齐到工作日时，必须逐点携带真实发布日期
    monthly = pd.Series([100.0, 101.0, 102.0],
                        index=pd.to_datetime(["2026-06-01", "2026-07-01", "2026-08-01"]))
    bdays = pd.bdate_range("2026-07-15", "2026-08-20")
    aligned, vint = align_monthly_with_vintage(monthly, bdays)
    # 7/15~7/31 对齐的是 7 月发布值，vintage 必须是 2026-07-01，不能写成当天
    assert vint.loc[pd.Timestamp("2026-07-15")] == "2026-07-01"
    assert aligned.loc[pd.Timestamp("2026-07-15")] == 101.0
    assert vint.loc[pd.Timestamp("2026-08-19")] == "2026-08-01"


def test_missing_factor_is_never_zero_filled():
    # 无 GPR、事件源不可达：地缘因素列必须保持 NaN，availability=False
    b = build_synthetic_bundle(AS_OF, 730)
    macro = b["macro"].drop(columns=["gpr_index"])
    feats = build_features(b["prices"], macro, events=pd.DataFrame(),
                           views=pd.DataFrame(), target="wti",
                           events_available=False, views_available=False)
    assert feats["gpr_level_z"].isna().all()
    avail = factor_availability(feats)
    assert avail["geopolitical_risk"] is False


def test_missing_factor_excluded_from_weights():
    b = build_synthetic_bundle(AS_OF, 730)
    feats = build_features(b["prices"], b["macro"], b["events"], b["views"])
    X, y = make_supervised(feats, b["prices"]["wti"], 10)
    avail = {f: True for f in FACTOR_GROUPS if f != "_technical_"}
    avail["geopolitical_risk"] = False   # 模拟该因素无真实数据
    w = learn_factor_weights(X, y, available=avail)
    row = w[w["factor"] == "geopolitical_risk"].iloc[0]
    assert pd.isna(row["weight"]) and row["available"] == 0
    usable = w.dropna(subset=["weight"])
    assert abs(usable["weight"].sum() - 1.0) < 1e-3   # 只在可用因素间归一化


def test_model_refuses_insufficient_real_data():
    # 真实样本不足时必须显式失败，而不是硬算出预测
    b = build_synthetic_bundle(AS_OF, 730)
    feats = build_features(b["prices"], b["macro"], b["events"], b["views"]).tail(100)
    with pytest.raises(InsufficientData):
        ShortTermForecaster(horizon=5).fit(feats, b["prices"]["wti"].loc[feats.index])


# ----------------------------------------------------- 多数据源优先级链
def test_chain_failover_to_second_source():
    from oilcast.data_sources.sources import run_chain
    import pandas as pd
    calls = []
    def dead():
        calls.append("A"); return None
    def alive():
        calls.append("B"); return (pd.Series([1.0, 2.0, 3.0]), {"k": 1})
    res = run_chain([("源A", dead), ("源B", alive)], field_name="x", min_obs=2)
    assert res.ok and res.used == "源B" and calls == ["A", "B"]
    assert res.attempts[0].ok is False and res.attempts[1].ok is True
    assert "源A✗" in res.trail_text() and "源B✓" in res.trail_text()


def test_chain_all_sources_fail():
    from oilcast.data_sources.sources import run_chain
    res = run_chain([("A", lambda: None), ("B", lambda: None)], field_name="x")
    assert not res.ok and res.used is None and len(res.attempts) == 2


def test_chain_insufficient_obs_falls_through():
    from oilcast.data_sources.sources import run_chain
    import pandas as pd
    thin = lambda: (pd.Series([1.0]), {})
    full = lambda: (pd.Series([1., 2., 3.]), {})
    res = run_chain([("薄源", thin), ("足源", full)], field_name="x", min_obs=3)
    assert res.used == "足源" and "薄源" in res.trail_text()


def test_chain_provider_exception_isolated():
    from oilcast.data_sources.sources import run_chain
    import pandas as pd
    def boom():
        raise RuntimeError("网络中断")
    res = run_chain([("崩源", boom), ("好源", lambda: (pd.Series([1., 2.]), {}))],
                    field_name="x", min_obs=1)
    assert res.ok and res.used == "好源"
    assert "RuntimeError" in res.attempts[0].reason

# ------------------------------------- 新鲜度优先选源 / CNBC解析 / 预测复盘
def test_chain_freshest_picks_latest_even_if_not_first():
    """现货主源滞后时，freshest 必须选末次观测更新的后位期货源。"""
    from oilcast.data_sources.sources import run_chain
    import pandas as pd
    old = lambda: (pd.Series([80., 81.], index=pd.to_datetime(["2026-08-29", "2026-09-01"])),
                   {"last_observed": "2026-09-01"})
    new = lambda: (pd.Series([80., 81., 82.], index=pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-04"])),
                   {"last_observed": "2026-09-04"})
    res = run_chain([("滞后现货", old), ("新鲜期货", new)], field_name="wti",
                    min_obs=2, select="freshest")
    assert res.used == "新鲜期货"
    assert res.attempts[0].ok and not res.attempts[0].chosen
    assert res.attempts[1].chosen and "采用" in res.attempts[1].line()


class _FakeResp:
    def __init__(self, payload):
        self._p = payload
    def json(self):
        return self._p


class _FakeSess:
    def __init__(self, payload):
        self._p = payload
    def get(self, url):
        return _FakeResp(self._p)


def test_cnbc_client_parses_bars():
    import pandas as pd
    from oilcast.data_sources.cnbc_client import fetch_cnbc_bars
    payload = {"barData": {"priceBars": [
        {"close": "90.0", "tradeTimeinMills": int(pd.Timestamp("2026-09-03").timestamp() * 1000)},
        {"close": "91.5", "tradeTimeinMills": int(pd.Timestamp("2026-09-04").timestamp() * 1000)},
    ]}}
    got = fetch_cnbc_bars("@CL.1", pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-05"),
                          sess=_FakeSess(payload))
    assert got is not None
    s, meta = got
    assert len(s) == 2 and abs(s.iloc[-1] - 91.5) < 1e-9
    assert meta["last_observed"] == "2026-09-04" and "期货" in meta["caliber"]


def test_review_scores_past_forecasts_against_realized():
    import pandas as pd
    from oilcast.storage.database import OilCastDB
    from oilcast.models.review import review_predictions
    db = OilCastDB()  # conftest 已把 OILCAST_HOME 关到临时目录
    import sqlite3 as _sq
    from oilcast.config import get_config as _gc
    _c = _sq.connect(_gc()["storage"]["sqlite_path"])
    _c.execute("DELETE FROM forecasts"); _c.commit(); _c.close()  # 清掉同 session demo 写入，保证隔离
    idx = pd.bdate_range("2026-08-25", "2026-09-04")
    prices = {"wti": pd.Series([88.0, 88.5, 89.0, 89.5, 90.0, 90.5, 91.0, 91.48],
                               index=pd.to_datetime(["2026-08-25", "2026-08-26", "2026-08-27",
                                                     "2026-08-28", "2026-08-31", "2026-09-01",
                                                     "2026-09-02", "2026-09-04"]))}
    # 8-25 发布(发布价88)、目标 9-04：预测 90(看涨)，实际 91.48(看涨)→方向命中；区间[89,92]覆盖
    db.save_forecasts("2026-08-25", [{"horizon": "short", "instrument": "wti",
                                      "target_date": "2026-09-04", "mean": 90.0,
                                      "q05": 89.0, "q25": 89.5, "q50": 90.0, "q75": 91.0,
                                      "q95": 92.0, "prob_up": 0.6, "prob_down": 0.4}])
    rv = review_predictions(prices, pd.Timestamp("2026-09-05"))
    assert rv["available"] and rv["summary"]["n"] == 1
    row = rv["detail"][0]
    assert row["actual"] == 91.48 and row["covered"] is True and row["dir_hit"] is True
    assert row["pred_ret_pct"] > 0 and row["actual_ret_pct"] > 0
