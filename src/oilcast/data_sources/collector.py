"""数据采集总调度。

strict（默认）：只采集真实数据，逐字段过质量门并生成 lineage；
    任何源不可达/过期/样本不足都如实记录状态，缺口保留 NaN，绝不合成补齐。
demo（显式 --demo）：全部使用 synthetic，mode=demo，报告强制演示水印。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from ..config import get_config
from ..utils import get_logger, now_beijing, run_with_timeout
from . import events as ev_mod
from . import institutional as inst_mod
from . import macro as macro_mod
from . import prices as price_mod
from . import synthetic
from .base import DataBundle
from .diesel import fetch_diesel_live
from .quality import FieldLineage, assess_daily, assess_monthly

LOG = get_logger(__name__)

DISPLAY = {
    "wti": "WTI原油现货", "brent": "布伦特原油现货", "diesel": "国内0#柴油批发价",
    "usd_index": "美元广义指数", "us_treasury_10y": "美国10年期国债收益率",
    "cpi_surprise": "美国CPI同比(月频)", "jobs_surprise": "非农就业意外(月频)",
    "fed_policy_expectation": "美联储政策预期(短端利率推导)",
    "demand_outlook": "工业产出同比/需求代理(月频)", "geopolitical_risk": "地缘风险指数GPRD",
    "events": "地缘与宏观事件", "institutional_view": "机构观点",
}
DAILY_MACRO = {"usd_index": "dxy", "us_treasury_10y": "us10y",
               "fed_policy_expectation": "fed_expectation", "geopolitical_risk": "gpr_index"}
MONTHLY_MACRO = {"cpi_surprise": "cpi_yoy", "jobs_surprise": "nonfarm_surprise",
                 "demand_outlook": "demand_proxy"}


def _empty_events():
    return pd.DataFrame(columns=["date", "title", "source", "theme", "sentiment",
                                 "intensity", "est_price_impact", "url"])


def collect_strict(as_of: pd.Timestamp) -> DataBundle:
    cfg = get_config()
    history_days = int(cfg["model"]["history_days"])
    qg = cfg["quality_gate"]
    lineage = {}
    idx = pd.bdate_range(as_of - pd.Timedelta(days=history_days), as_of)

    # 1) 真实油价（FRED/EIA 现货为主，yfinance 备份）
    got = run_with_timeout(price_mod.fetch_prices, (as_of, history_days), timeout_sec=120)
    prices = got[0] if got else pd.DataFrame(index=idx)
    pmeta = got[1] if got else {}
    prices = prices.reindex(idx)
    for f in ("wti", "brent"):
        m = pmeta.get(f, {"source_name": "UNAVAILABLE", "url": "", "frequency": "daily_business"})
        lineage[f] = assess_daily(f, DISPLAY[f], prices.get(f), as_of,
                                  m["source_name"], m.get("url", ""),
                                  dict(qg["price_daily"])).to_dict()

    # 国内柴油：只认真实钩子，拿不到就是 unavailable（绝不按国际油价推算冒充）
    dgot = run_with_timeout(fetch_diesel_live, (as_of, history_days), timeout_sec=60)
    diesel_s, dmeta = dgot if dgot else (None, {})
    if diesel_s is not None and len(diesel_s.dropna()) >= 20:
        prices["diesel"] = diesel_s.reindex(idx)
        dm = dmeta or {"source_name": "live", "url": "", "frequency": "daily_business"}
        lineage["diesel"] = assess_daily("diesel", DISPLAY["diesel"], prices["diesel"],
                                         as_of, dm["source_name"], dm.get("url", ""),
                                         dict(qg["price_daily"])).to_dict()
    else:
        prices["diesel"] = np.nan
        lineage["diesel"] = FieldLineage(
            "diesel", DISPLAY["diesel"],
            source_name=(dmeta or {}).get("source_name", "UNAVAILABLE"),
            status="unavailable",
            note="未接入可核验的国内柴油真实源；按数据原则不以国际油价推算冒充").to_dict()

    # 2) 真实宏观（FRED + GPRD）
    mgot = run_with_timeout(macro_mod.fetch_macro, (as_of, history_days), timeout_sec=150)
    if mgot:
        macro, mmeta, vintage = mgot
    else:
        macro, mmeta, vintage = pd.DataFrame(index=idx), {}, {}
    macro = macro.reindex(idx)
    for f, col in DAILY_MACRO.items():
        m = mmeta.get(f, {"source_name": "UNAVAILABLE", "url": "", "frequency": "daily_business"})
        lineage[f] = assess_daily(f, DISPLAY[f], macro.get(col), as_of,
                                  m["source_name"], m.get("url", ""),
                                  dict(qg["macro_daily"])).to_dict()
    for f, col in MONTHLY_MACRO.items():
        m = mmeta.get(f, {"source_name": "UNAVAILABLE", "url": "", "frequency": "monthly"})
        lineage[f] = assess_monthly(f, DISPLAY[f], macro.get(col), as_of,
                                    m["source_name"], m.get("url", ""),
                                    dict(qg["macro_monthly"])).to_dict()

    # 3) 真实事件（不可达即空表，绝不伪造）
    events = run_with_timeout(ev_mod.fetch_events,
                              (as_of, int(qg["events"]["lookback_days"])), timeout_sec=60)
    if events is None or len(events) == 0:
        events = _empty_events()
        lineage["events"] = FieldLineage(
            "events", DISPLAY["events"], source_name="UNAVAILABLE", status="unavailable",
            note="事件RSS当前不可达或无有效条目，保持空缺、不生成模拟事件").to_dict()
    else:
        lineage["events"] = FieldLineage(
            "events", DISPLAY["events"], source_name="GoogleNews RSS + 规则打分",
            url=str(cfg["data_sources"]["google_news_rss"]), frequency="event",
            n_obs=len(events),
            first_observed=pd.to_datetime(events["date"]).min().strftime("%Y-%m-%d"),
            last_observed=pd.to_datetime(events["date"]).max().strftime("%Y-%m-%d"),
            status="ok").to_dict()

    # 4) 真实机构观点（不可达即空表）
    views = run_with_timeout(inst_mod.fetch_institutional_views, (as_of,), timeout_sec=60)
    if views is None or len(views) == 0:
        views = pd.DataFrame(columns=["date", "institution", "target_wti", "stance", "note"])
        lineage["institutional_view"] = FieldLineage(
            "institutional_view", DISPLAY["institutional_view"],
            source_name="UNAVAILABLE", status="unavailable",
            note="机构观点源当前不可达，保持空缺").to_dict()
    else:
        lineage["institutional_view"] = FieldLineage(
            "institutional_view", DISPLAY["institutional_view"],
            source_name="Google News RSS + 结构化抽取",
            url=str(cfg["data_sources"]["google_news_rss"]), frequency="event",
            n_obs=len(views),
            first_observed=pd.to_datetime(views["date"]).min().strftime("%Y-%m-%d"),
            last_observed=pd.to_datetime(views["date"]).max().strftime("%Y-%m-%d"),
            status="ok").to_dict()

    prices = prices.loc[prices.index <= as_of]
    macro = macro.loc[macro.index <= as_of]
    return DataBundle(prices, macro, events, views, lineage=lineage,
                      vintage=vintage, mode="strict")


def collect_demo(as_of: pd.Timestamp) -> DataBundle:
    """显式演示：全部合成，lineage 强制标注 SYNTHETIC-DEMO。"""
    cfg = get_config()
    synth = synthetic.build_synthetic_bundle(as_of, int(cfg["model"]["history_days"]))
    lineage = {}
    for f, display in DISPLAY.items():
        lineage[f] = FieldLineage(f, display, source_name="SYNTHETIC-DEMO（合成，非真实观测）",
                                  url="", frequency="demo", status="ok",
                                  note="仅用于功能演示，严禁用于任何真实判断").to_dict()
    return DataBundle(synth["prices"], synth["macro"], synth["events"], synth["views"],
                      lineage=lineage, mode="demo")


def collect(as_of: Optional[datetime] = None, mode: Optional[str] = None,
            prefer_live: Optional[bool] = None) -> DataBundle:
    """采集入口。mode: 'strict'（默认）/ 'demo'；prefer_live=False 等价 demo（兼容旧参数）。"""
    as_of = pd.Timestamp(as_of or now_beijing())
    if as_of.tzinfo is not None:
        as_of = as_of.tz_convert("Asia/Shanghai").tz_localize(None)
    cfg_mode = mode or str(get_config()["data_sources"]["mode"])
    if mode is None and prefer_live is False:
        cfg_mode = "demo"
    bundle = collect_demo(as_of) if cfg_mode == "demo" else collect_strict(as_of)
    LOG.info("\n%s", bundle.summary())
    return bundle
