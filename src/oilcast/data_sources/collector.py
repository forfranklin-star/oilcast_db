"""数据采集总调度：真实源优先，失败逐表回退到合成数据，并记录 provenance。"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ..config import get_config
from ..utils import get_logger, now_beijing, run_with_timeout
from . import events as ev_mod
from . import institutional as inst_mod
from . import macro as macro_mod
from . import prices as price_mod
from . import synthetic
from .base import DataBundle
from .diesel import estimate_from_brent, fetch_diesel_live

LOG = get_logger(__name__)


def collect(as_of: Optional[datetime] = None, prefer_live: Optional[bool] = None) -> DataBundle:
    """采集一整套数据。

    Parameters
    ----------
    as_of : 报告基准时点（默认北京时间现在）
    prefer_live : 覆盖配置中的 data_sources.prefer_live（测试时可强制 False）
    """
    cfg = get_config()
    as_of = pd.Timestamp(as_of or now_beijing())
    if as_of.tzinfo is not None:          # 内部索引统一为 tz-naive，避免比较/ reindex 报错
        as_of = as_of.tz_convert("Asia/Shanghai").tz_localize(None)
    history_days = int(cfg["model"]["history_days"])
    live = bool(cfg["data_sources"]["prefer_live"]) if prefer_live is None else prefer_live

    # 1) 合成基底，保证任何网络条件下结构完整
    synth = synthetic.build_synthetic_bundle(as_of, history_days)
    prices, macro = synth["prices"].copy(), synth["macro"].copy()
    events, views = synth["events"].copy(), synth["views"].copy()
    provenance = {c: "simulated" for c in ("wti", "brent", "diesel", "macro", "events", "views")}

    if not live:
        LOG.info("prefer_live=False，直接使用合成数据")
        return DataBundle(prices, macro, events, views, provenance)

    # 2) 国际油价（yfinance / EIA）—— 每个真实源都有总耗时预算，超时即降级
    real_prices = run_with_timeout(price_mod.fetch_prices,
                                   (as_of, history_days), timeout_sec=90)
    if real_prices is not None:
        for col in ("wti", "brent"):
            if col in real_prices.columns and real_prices[col].notna().sum() >= 30:
                prices[col] = real_prices[col].reindex(prices.index).ffill().combine_first(prices[col])
                provenance[col] = "live(yfinance)"
        # 用真实 Brent 重估柴油
        diesel_live = fetch_diesel_live(as_of, history_days)
        if diesel_live is not None:
            prices["diesel"] = diesel_live.reindex(prices.index).ffill().combine_first(prices["diesel"])
            provenance["diesel"] = "live"
        else:
            est = estimate_from_brent(prices["brent"].dropna())
            prices["diesel"] = est.reindex(prices.index).combine_first(prices["diesel"])
            provenance["diesel"] = "estimated(brent-cost-pass-through)"

    # 3) 宏观（逐列替换，缺列保留合成）
    real_macro = run_with_timeout(macro_mod.fetch_macro,
                                  (as_of, history_days), timeout_sec=90)
    if real_macro is not None:
        for col in real_macro.columns:
            aligned = real_macro[col].reindex(macro.index).ffill()
            if aligned.notna().sum() >= 30:
                macro[col] = aligned.combine_first(macro[col])
                provenance["macro"] = "live(FRED/yfinance/GPRD)"
        # 若真实 GPR 缺失，用事件强度自建（特征层也会再兜底一次）

    # 4) 事件与机构观点
    real_events = run_with_timeout(ev_mod.fetch_events, (as_of,), timeout_sec=60)
    if real_events is not None and not real_events.empty:
        events = real_events
        provenance["events"] = "live(GoogleNews-RSS+rule-scoring)"
    real_views = run_with_timeout(inst_mod.fetch_institutional_views,
                                  (as_of,), timeout_sec=60)
    if real_views is not None and not real_views.empty:
        views = real_views
        provenance["views"] = "live(GoogleNews-RSS+NLP-parse)"

    # 只保留到 as_of 为止的数据，防止未来泄漏
    prices = prices.loc[prices.index <= as_of]
    macro = macro.loc[macro.index <= as_of]
    return DataBundle(prices, macro, events, views, provenance)
