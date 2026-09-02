"""数据采集总调度。
strict（默认）：只采集真实数据，逐字段过质量门并生成 lineage；
    每个字段按多数据源优先级链依次尝试，尝试全过程（含失败原因）写入谱系；
    任何源不可达/过期/样本不足都如实记录，缺口保留 NaN，绝不合成补齐。
demo（显式 --demo）：全部使用 synthetic，mode=demo，报告强制演示水印。
"""
from __future__ import annotations
from datetime import datetime
from typing import List, Optional
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
    "demand_outlook": "工业产出同比/需求代理(月频)", "geopolitical_risk": "地缘风险(GPRD/事件代理)",
    "supply_disruption": "供给中断/OPEC+政策(事件)",
    "events": "地缘与宏观事件", "institutional_view": "机构观点",
}
DAILY_MACRO = {"usd_index": "dxy", "us_treasury_10y": "us10y",
               "fed_policy_expectation": "fed_expectation", "geopolitical_risk": "gpr_index"}
MONTHLY_MACRO = {"cpi_surprise": "cpi_yoy", "jobs_surprise": "nonfarm_surprise",
                 "demand_outlook": "demand_proxy"}


def _empty_events():
    return pd.DataFrame(columns=["date", "title", "source", "feed", "theme", "sentiment",
                                 "intensity", "est_price_impact", "url"])


def _series_trail(meta: dict) -> str:
    atts = meta.get("attempts")
    return " → ".join(a.line() for a in atts) if atts else meta.get("tried_sources", "")


def _feed_trail(attempts: List[dict]) -> str:
    parts = []
    for a in attempts or []:
        if a.get("ok"):
            n = a.get("n_kept", a.get("n", 0))
            parts.append(f"{a['source']}✓保留{n}条")
        else:
            parts.append(f"{a['source']}✗({a.get('reason', '不可达')})")
    return " → ".join(parts)


def collect_strict(as_of: pd.Timestamp) -> DataBundle:
    cfg = get_config()
    history_days = int(cfg["model"]["history_days"])
    qg = cfg["quality_gate"]
    lineage = {}
    idx = pd.bdate_range(as_of - pd.Timedelta(days=history_days), as_of)

    # 1) 真实油价（多源优先级链：FRED→EIA→Yahoo→yfinance）
    got = run_with_timeout(price_mod.fetch_prices, (as_of, history_days), timeout_sec=150)
    prices = got[0] if got else pd.DataFrame(index=idx)
    pmeta = got[1] if got else {}
    prices = prices.reindex(idx)
    for f in ("wti", "brent"):
        m = pmeta.get(f, {"source_name": "UNAVAILABLE", "url": "", "frequency": "daily_business"})
        lineage[f] = assess_daily(f, DISPLAY[f], prices.get(f), as_of,
                                  m["source_name"], m.get("url", ""),
                                  dict(qg["price_daily"]),
                                  note=m.get("note", ""), tried_sources=_series_trail(m)).to_dict()

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

    # 2) 真实事件（多 RSS 源链；先于宏观，供 GPR 事件代理使用）
    eres = run_with_timeout(ev_mod.fetch_events,
                            (as_of, int(qg["events"]["lookback_days"])), timeout_sec=120)
    events, event_attempts = eres if eres else (None, [])
    if events is None or len(events) == 0:
        events = _empty_events()
        lineage["events"] = FieldLineage(
            "events", DISPLAY["events"], source_name="UNAVAILABLE", status="unavailable",
            tried_sources=_feed_trail(event_attempts),
            note="各新闻RSS源均不可达或无有效条目，保持空缺、不生成模拟事件").to_dict()
    else:
        lineage["events"] = FieldLineage(
            "events", DISPLAY["events"],
            source_name="多源RSS(" + ",".join(sorted(events["feed"].unique())) + ")+规则打分",
            frequency="event", n_obs=len(events),
            first_observed=pd.to_datetime(events["date"]).min().strftime("%Y-%m-%d"),
            last_observed=pd.to_datetime(events["date"]).max().strftime("%Y-%m-%d"),
            tried_sources=_feed_trail(event_attempts), status="ok").to_dict()

    # 3) 真实机构观点（多 RSS 源链）
    vres = run_with_timeout(inst_mod.fetch_institutional_views, (as_of,), timeout_sec=120)
    views, view_attempts = vres if vres else (None, [])
    if views is None or len(views) == 0:
        views = pd.DataFrame(columns=["date", "institution", "target_wti", "stance", "note"])
        # 专门的机构评级未抽到，但若事件流中有机构预测/目标价主题，以真实事件代理该因素
        n_inst = int((events["theme"] == "institutional_view").sum()) if len(events) else 0
        if n_inst > 0:
            lineage["institutional_view"] = FieldLineage(
                "institutional_view", DISPLAY["institutional_view"],
                source_name="多源真实事件代理（未抽到结构化机构评级）",
                frequency="event", status="ok", n_obs=n_inst,
                last_observed=pd.to_datetime(events["date"]).max().strftime("%Y-%m-%d"),
                tried_sources=_feed_trail(view_attempts),
                note="结构化机构评级不可得，改以新闻中机构预测/目标价主题事件代理").to_dict()
        else:
            lineage["institutional_view"] = FieldLineage(
                "institutional_view", DISPLAY["institutional_view"],
                source_name="UNAVAILABLE", status="unavailable",
                tried_sources=_feed_trail(view_attempts),
                note="各源未抽取到可核验机构观点，保持空缺").to_dict()
    else:
        lineage["institutional_view"] = FieldLineage(
            "institutional_view", DISPLAY["institutional_view"],
            source_name="多源RSS(" + ",".join(sorted(views["source"].astype(str).unique())) + ")结构化抽取",
            frequency="event", n_obs=len(views),
            first_observed=pd.to_datetime(views["date"]).min().strftime("%Y-%m-%d"),
            last_observed=pd.to_datetime(views["date"]).max().strftime("%Y-%m-%d"),
            tried_sources=_feed_trail(view_attempts), status="ok").to_dict()

    # 4) 真实宏观（FRED/财政部/Yahoo 多源链 + GPRD）
    mgot = run_with_timeout(macro_mod.fetch_macro, (as_of, history_days), timeout_sec=180)
    if mgot:
        macro, mmeta, vintage = mgot
    else:
        macro, mmeta, vintage = pd.DataFrame(index=idx), {}, {}
    macro = macro.reindex(idx)

    # GPRD 长历史序列不可达时保持缺失（稀疏近期事件不足以冒充完整历史指数，
    # 质量门如实标记）；模型特征层另用多源真实事件构造近期代理，口径分离。
    if macro.get("gpr_index") is None or macro["gpr_index"].notna().sum() <= 30:
        n_geo = int(events["theme"].isin(["geopolitical_risk", "supply_disruption"]).sum()) \
            if len(events) else 0
        gm = mmeta.get("geopolitical_risk", {})
        if n_geo:
            gm["note"] = ("GPRD官方长序列不可达；模型层另以近30天%d条多源真实事件构造近期代理特征" % n_geo)
        mmeta["geopolitical_risk"] = gm

    for f, col in DAILY_MACRO.items():
        m = mmeta.get(f, {"source_name": "UNAVAILABLE", "url": "", "frequency": "daily_business"})
        lineage[f] = assess_daily(f, DISPLAY[f], macro.get(col), as_of,
                                  m["source_name"], m.get("url", ""),
                                  dict(qg["macro_daily"]),
                                  note=m.get("note", ""), tried_sources=_series_trail(m)).to_dict()
    for f, col in MONTHLY_MACRO.items():
        m = mmeta.get(f, {"source_name": "UNAVAILABLE", "url": "", "frequency": "monthly"})
        lineage[f] = assess_monthly(f, DISPLAY[f], macro.get(col), as_of,
                                    m["source_name"], m.get("url", ""),
                                    dict(qg["macro_monthly"]),
                                    tried_sources=_series_trail(m)).to_dict()

    # 5) 事件衍生因素谱系（源可达时，"当天无相关事件"是真实的 0，因素可进模型）
    event_trail = lineage.get("events", {}).get("tried_sources", "")
    ev_available = len(events) > 0
    # 5a) 供给中断/OPEC+：纯事件因素
    n_sup = int((events["theme"] == "supply_disruption").sum()) if ev_available else 0
    if ev_available:
        lineage["supply_disruption"] = FieldLineage(
            "supply_disruption", DISPLAY["supply_disruption"],
            source_name=lineage["events"]["source_name"].replace("+规则打分", "")+"·供给主题",
            frequency="event", status="ok", n_obs=n_sup,
            last_observed=pd.to_datetime(events["date"]).max().strftime("%Y-%m-%d"),
            tried_sources=event_trail,
            note="多源真实新闻按供给/制裁/减产主题打分；无事件日记为真实0").to_dict()
    else:
        lineage["supply_disruption"] = FieldLineage(
            "supply_disruption", DISPLAY["supply_disruption"], source_name="UNAVAILABLE",
            status="unavailable", tried_sources=event_trail,
            note="事件源不可达，供给因素保持缺失").to_dict()
    # 5b) 地缘风险：GPRD 长序列优先；不可达时若事件源可用则以真实事件代理（标注口径）
    if lineage.get("geopolitical_risk", {}).get("status") != "ok" and ev_available:
        n_geo = int(events["theme"].isin(["geopolitical_risk", "supply_disruption"]).sum())
        lineage["geopolitical_risk"] = FieldLineage(
            "geopolitical_risk", DISPLAY["geopolitical_risk"],
            source_name="多源真实事件代理（GPRD官方长序列不可达）",
            frequency="event", status="ok", n_obs=n_geo,
            last_observed=pd.to_datetime(events["date"]).max().strftime("%Y-%m-%d"),
            tried_sources=event_trail,
            note="GPRD指数不可达，改以多源真实地缘/供给事件强度代理，口径不同于GPRD").to_dict()

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
