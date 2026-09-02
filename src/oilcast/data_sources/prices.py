"""国际原油价格：优先 yfinance（无需 key），EIA 官方 API 作为可选增强。

yfinance 标的：
    CL=F  NYMEX WTI 近月连续
    BZ=F  ICE Brent 近月连续
EIA：https://www.eia.gov/opendata/ 注册免费 key 后设置环境变量 EIA_API_KEY。
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from ..config import get_config
from ..utils import get_logger, run_with_timeout

LOG = get_logger(__name__)


def _download_one(ticker: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    import yfinance as yf
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or df.empty:
        LOG.warning("%s 返回空数据", ticker)
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):   # 多 ticker 时的多级列兼容
        close = close.iloc[:, 0]
    close = close.squeeze().dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.rename(ticker)


def fetch_yf_series(ticker: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    """单标的收盘价序列；任何异常/超时都返回 None，由上层兜底。"""
    try:
        import yfinance as yf  # noqa: F401
    except ImportError:
        LOG.warning("未安装 yfinance，跳过真实行情抓取")
        return None
    try:
        return run_with_timeout(_download_one, (ticker, start, end), timeout_sec=20)
    except Exception as exc:  # 网络/解析问题一律视为抓取失败
        LOG.warning("yfinance 抓取 %s 失败：%s", ticker, exc)
        return None


def fetch_eia_series(series_id: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    """EIA v2 API（可选）。需要环境变量 EIA_API_KEY。"""
    cfg = get_config()
    key = os.getenv(str(cfg["data_sources"]["eia_api_key_env"]))
    if not key:
        return None
    import requests
    url = (f"https://api.eia.gov/v2/seriesid/{series_id}"
           f"?api_key={key}&start={start:%Y-%m-%d}&end={end:%Y-%m-%d}")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()["response"]["data"]
        s = pd.Series({pd.Timestamp(r["period"]): float(r["value"]) for r in data}).sort_index()
        return s.rename(series_id)
    except Exception as exc:
        LOG.warning("EIA 抓取 %s 失败：%s", series_id, exc)
        return None


def fetch_prices(as_of: datetime, history_days: int) -> Optional[pd.DataFrame]:
    """返回列 [wti, brent] 的工作日价格表（不含国内柴油，柴油见 diesel.py）。"""
    start = pd.Timestamp(as_of) - timedelta(days=history_days)
    end = pd.Timestamp(as_of)
    inst = get_config()["instruments"]
    wti = fetch_yf_series(str(inst["wti"]["yf_ticker"]), start, end)
    brent = fetch_yf_series(str(inst["brent"]["yf_ticker"]), start, end)
    if wti is None and brent is None:
        return None
    df = pd.concat([wti, brent], axis=1)
    df.columns = ["wti", "brent"]
    df = df.dropna(how="all").ffill()
    if df.dropna().shape[0] < 30:          # 数据太少不足以建模
        LOG.warning("真实油价有效样本不足 30 条，回退合成数据")
        return None
    return df
