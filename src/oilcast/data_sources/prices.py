"""国际原油【真实】价格采集。

主源：FRED 转发的 EIA 现货序列（无需 key，可核验）
    DCOILWTICO   WTI Cushing 现货 美元/桶（工作日日频）
    DCOILBRENTEU Brent 现货 美元/桶
备份源：yfinance 期货近月（CL=F / BZ=F），仅在主源不可达时启用，
    并在 lineage 中注明源切换；两源数值不做混合拼接。
任何源不可达 -> 对应列保持缺失，由质量门判定 unavailable，严禁估算补齐。
另提供 EIA v2 API 可选接入（环境变量 EIA_API_KEY）。
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd

from ..config import get_config
from ..utils import get_logger, run_with_timeout
from .fred_client import fetch_fred, fred_url

LOG = get_logger(__name__)


def _download_one(ticker: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    import yfinance as yf
    df = yf.download(ticker, start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                     end=(pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=True, threads=False)
    if df is None or df.empty:
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze().dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.rename(ticker)


def fetch_yf_series(ticker: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    try:
        import yfinance  # noqa: F401
    except ImportError:
        return None
    try:
        return run_with_timeout(_download_one, (ticker, start, end), timeout_sec=20)
    except Exception as exc:
        LOG.warning("yfinance %s 失败：%s", ticker, exc)
        return None


def fetch_eia_series(series_id: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    """EIA v2 API（可选，需环境变量 EIA_API_KEY）。"""
    key = os.getenv(str(get_config()["data_sources"]["eia_api_key_env"]))
    if not key:
        return None
    import requests
    url = (f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={key}"
           f"&start={pd.Timestamp(start):%Y-%m-%d}&end={pd.Timestamp(end):%Y-%m-%d}")
    try:
        j = requests.get(url, timeout=12).json()["response"]["data"]
        return pd.Series({pd.Timestamp(r["period"]): float(r["value"]) for r in j}).sort_index()
    except Exception as exc:
        LOG.warning("EIA %s 失败：%s", series_id, exc)
        return None


def fetch_prices(as_of: datetime, history_days: int
                 ) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """返回 (真实价格表[wti,brent], 每字段来源元信息)。缺口保留 NaN，不做填充。"""
    cfg = get_config()
    start, end = pd.Timestamp(as_of) - timedelta(days=history_days), pd.Timestamp(as_of)
    idx = pd.bdate_range(start, end)
    smap = dict(cfg["data_sources"]["fred_series_map"])
    meta: Dict[str, dict] = {}
    out = {}

    for field in ("wti", "brent"):
        sid = smap.get(field)
        fred_s = fetch_fred(sid, start, end) if sid else None
        if fred_s is not None and len(fred_s) > 0:
            out[field] = fred_s.reindex(idx)
            meta[field] = {"source_name": f"FRED:{sid}（源头 EIA 现货）",
                           "url": fred_url(sid), "frequency": "daily_business"}
            continue
        if bool(cfg["data_sources"].get("yfinance_backup", True)):
            tk = str(cfg["instruments"][field]["yf_ticker"])
            yf_s = fetch_yf_series(tk, start, end)
            if yf_s is not None and len(yf_s) > 0:
                out[field] = yf_s.reindex(idx)
                meta[field] = {"source_name": f"yfinance:{tk}（期货近月，备份源）",
                               "url": f"https://finance.yahoo.com/quote/{tk}",
                               "frequency": "daily_business"}
                continue
        out[field] = pd.Series(index=idx, dtype=float)   # 保持缺失，绝不造数
        meta[field] = {"source_name": "UNAVAILABLE", "url": "", "frequency": "daily_business"}

    return pd.DataFrame(out)[["wti", "brent"]], meta


def fetch_us_diesel_retail(as_of: datetime, history_days: int
                           ) -> Tuple[Optional[pd.Series], dict]:
    """【美国】柴油零售价（GASDESM，美元/加仑，月频）参考序列。
    明确不是中国柴油，不参与国内柴油价格展示与预测。"""
    sid = dict(get_config()["data_sources"]["fred_series_map"]).get("us_diesel_retail")
    s = fetch_fred(sid, pd.Timestamp(as_of) - timedelta(days=history_days), pd.Timestamp(as_of))
    meta = {"source_name": f"FRED:{sid}（美国零售，非中国口径）",
            "url": fred_url(sid), "frequency": "monthly"}
    return s, meta
