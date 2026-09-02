"""宏观与金融市场【真实】数据（FRED + GPRD），全部可核验、带观测日期。

日频（工作日）：DGS10 10Y、DGS2 2Y、DTWEXBGS 广义美元、GPRD 地缘风险
月频（发布日对齐到工作日，携带 vintage 原始发布日期）：
    CPIAUCSL→CPI同比；PAYEMS→非农新增及意外z；FEDFUNDS；INDPRO→工业产出同比
铁律：源不可达则该列保持 NaN 并由质量门标记，绝不合成。
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import get_config
from ..utils import PoliteSession, get_logger
from .fred_client import fetch_fred, fred_url
from .quality import align_monthly_with_vintage, limited_ffill_daily

LOG = get_logger(__name__)


def fetch_gpr(start: pd.Timestamp, end: pd.Timestamp,
              sess: PoliteSession) -> Optional[pd.Series]:
    """Caldara-Iacoviello 日频地缘风险指数；不可达返回 None（保持缺失）。"""
    url = str(get_config()["data_sources"]["gprd_url"])
    resp = sess.get(url)
    if resp is None:
        return None
    try:
        df = pd.read_csv(io.StringIO(resp.text))
        dcol = next(c for c in df.columns if c.lower() in ("date", "day"))
        vcol = next(c for c in df.columns if "gpr" in c.lower())
        s = pd.Series(df[vcol].values, index=pd.to_datetime(df[dcol]),
                      name="gpr_index").sort_index()
        return s.loc[(s.index >= start) & (s.index <= end)]
    except Exception as exc:
        LOG.warning("GPRD 解析失败（该因素将标记缺失）：%s", exc)
        return None


def fetch_macro(as_of: datetime, history_days: int
                ) -> Tuple[pd.DataFrame, Dict[str, dict], Dict[str, pd.Series]]:
    """返回 (日频宏观表, 字段来源元信息, vintage 月频发布日期)。"""
    cfg = get_config()
    start, end = pd.Timestamp(as_of) - timedelta(days=history_days), pd.Timestamp(as_of)
    bdays = pd.bdate_range(start, end)
    smap = dict(cfg["data_sources"]["fred_series_map"])
    ffill_lim = int(dict(cfg["quality_gate"]["macro_daily"])["ffill_limit_bdays"])
    sess = PoliteSession()
    meta: Dict[str, dict] = {}
    vintage: Dict[str, pd.Series] = {}
    frame = pd.DataFrame(index=bdays, dtype=float)

    def put_daily(field, sid, col_name):
        s = fetch_fred(sid, start, end, sess) if sid else None
        if s is None:
            frame[col_name] = np.nan
            meta[field] = {"source_name": "UNAVAILABLE", "url": "", "frequency": "daily_business"}
        else:
            frame[col_name] = limited_ffill_daily(s, ffill_lim).reindex(bdays)
            meta[field] = {"source_name": f"FRED:{sid}", "url": fred_url(sid),
                           "frequency": "daily_business"}

    def put_monthly(field, sid, col_name, transform):
        s = fetch_fred(sid, start - timedelta(days=400), end, sess) if sid else None
        if s is None:
            frame[col_name] = np.nan
            meta[field] = {"source_name": "UNAVAILABLE", "url": "", "frequency": "monthly"}
            return
        s = transform(s).dropna()
        aligned, vint = align_monthly_with_vintage(s, bdays)
        frame[col_name] = aligned
        vintage[col_name] = vint
        meta[field] = {"source_name": f"FRED:{sid}（月频，值为最近发布值并附发布日期）",
                       "url": fred_url(sid), "frequency": "monthly"}

    # ---- 日频 ----
    put_daily("us_treasury_10y", smap.get("us10y"), "us10y")
    put_daily("us2y", smap.get("us2y"), "us2y")
    put_daily("usd_index", smap.get("dxy"), "dxy")

    # ---- 月频 ----
    put_monthly("cpi_surprise", smap.get("cpi_level"), "cpi_yoy",
                lambda s: s.pct_change(12) * 100)

    def _nfp(s):   # 非农新增相对其 12 个月均值的标准化意外（真实统计代理）
        chg = s.diff()
        return (chg - chg.rolling(12, min_periods=3).mean()) / \
               chg.rolling(12, min_periods=3).std() * 60
    put_monthly("jobs_surprise", smap.get("payems"), "nonfarm_surprise", _nfp)
    put_monthly("fedfunds", smap.get("fedfunds"), "fedfunds", lambda s: s)
    put_monthly("demand_outlook", smap.get("demand_proxy"), "demand_proxy",
                lambda s: s.pct_change(12) * 100)

    # ---- 降息/加息预期：由真实短端利率走势推导（DGS2 优先，FEDFUNDS 备选）----
    if frame["us2y"].notna().sum() > 60:
        frame["fed_expectation"] = -((frame["us2y"].diff(20)) /
                                     frame["us2y"].rolling(60, min_periods=20).std()
                                     ).clip(-2, 2) / 2
        meta["fed_policy_expectation"] = {"source_name": "由 FRED:DGS2 短端利率走势推导",
                                          "url": fred_url(smap.get("us2y")),
                                          "frequency": "daily_business"}
    elif frame["fedfunds"].notna().sum() > 60:
        frame["fed_expectation"] = -((frame["fedfunds"].diff(20)) /
                                     frame["fedfunds"].rolling(60, min_periods=20).std()
                                     ).clip(-2, 2) / 2
        meta["fed_policy_expectation"] = {"source_name": "由 FRED:FEDFUNDS 推导",
                                          "url": fred_url(smap.get("fedfunds")),
                                          "frequency": "monthly"}
    else:
        frame["fed_expectation"] = np.nan
        meta["fed_policy_expectation"] = {"source_name": "UNAVAILABLE", "url": "",
                                          "frequency": "na"}

    # ---- GPR 地缘风险（日频独立源，不可达即缺失）----
    gpr = fetch_gpr(start, end, sess)
    if gpr is not None and len(gpr) > 30:
        frame["gpr_index"] = limited_ffill_daily(gpr, ffill_lim).reindex(bdays)
        meta["geopolitical_risk"] = {"source_name": "GPRD(Caldara&Iacoviello)",
                                     "url": str(cfg["data_sources"]["gprd_url"]),
                                     "frequency": "daily_business"}
    else:
        frame["gpr_index"] = np.nan
        meta["geopolitical_risk"] = {"source_name": "UNAVAILABLE", "url": "",
                                     "frequency": "daily_business"}

    LOG.info("真实宏观表非空观测：%s",
             {c: int(frame[c].notna().sum()) for c in frame.columns})
    return frame, meta, vintage
