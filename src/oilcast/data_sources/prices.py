"""国际原油【真实】价格采集 —— 多数据源优先级链（failover）。

每个标的按 config.data_sources.source_chains 中的顺序依次尝试：
    FRED（EIA 现货转发，无 key）→ EIA v2（可选 key）→ Yahoo 原生 API → yfinance，
第一个返回足量真实观测的源被采用，尝试全过程写入谱系；源之间绝不混合拼接。
任何源都不可达 -> 该列保持缺失，由质量门判定 unavailable，严禁估算补齐。
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from ..config import get_config
from ..utils import get_logger, run_with_timeout
from .cnbc_client import fetch_cnbc_bars
from .fred_client import fetch_fred, fred_url
from .sources import run_chain
from .treasury_client import fetch_treasury_curve  # noqa: F401  (供宏观复用)
from .yahoo_client import fetch_yahoo_chart

LOG = get_logger(__name__)


# ----------------------------------------------------------- yfinance 备份
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


def _provider_url(kind: str, ref: str) -> str:
    if kind == "fred":
        return fred_url(ref)
    if kind in ("yahoo", "yfinance"):
        return f"https://finance.yahoo.com/quote/{ref}"
    if kind == "eia":
        return f"https://www.eia.gov/opendata/browser/{ref.split('.')[0].lower()}"
    return ""


def _with_meta(s, caliber: str):
    """把单一 series 包成 (series, meta)，携带口径与末次观测，供新鲜度选源。"""
    if s is None or len(s) == 0:
        return None
    ss = s.dropna()
    return (s, {"caliber": caliber,
                "last_observed": ss.index.max().strftime("%Y-%m-%d")})


def _cnbc_tuple(got, caliber: str):
    if got is None:
        return None
    s, meta = got
    if caliber:
        meta["caliber"] = caliber
    return s, meta


def _build_providers(chain: List[dict], start, end) -> List[Tuple[str, callable]]:
    """把 config 中的有序源描述翻译成 run_chain 需要的 (name, fn) 列表。"""
    providers = []
    for item in chain:
        kind, ref, name = item["kind"], str(item["ref"]), item["name"]
        caliber = item.get("caliber", "")
        if kind == "cnbc":   # CNBC 已自带 (series, meta)
            providers.append((name, lambda r=ref, c=caliber:
                              _cnbc_tuple(fetch_cnbc_bars(r, start, end), c)))
        elif kind == "fred":
            providers.append((name, lambda r=ref, c=caliber: _with_meta(fetch_fred(r, start, end), c)))
        elif kind == "eia":
            providers.append((name, lambda r=ref, c=caliber: _with_meta(fetch_eia_series(r, start, end), c)))
        elif kind == "yahoo":
            providers.append((name, lambda r=ref, c=caliber: _with_meta(fetch_yahoo_chart(r, start, end), c)))
        elif kind == "yfinance":
            if bool(get_config()["data_sources"].get("yfinance_backup", True)):
                providers.append((name, lambda r=ref, c=caliber: _with_meta(fetch_yf_series(r, start, end), c)))
    return providers


def fetch_prices(as_of: datetime, history_days: int
                 ) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """返回 (真实价格表[wti,brent], 每字段来源元信息)。缺口保留 NaN，不做填充。"""
    cfg = get_config()
    start, end = pd.Timestamp(as_of) - timedelta(days=history_days), pd.Timestamp(as_of)
    idx = pd.bdate_range(start, end)
    chains = dict(cfg["data_sources"]["source_chains"])
    gate_min = int(cfg["quality_gate"]["price_daily"]["min_obs"])
    meta: Dict[str, dict] = {}
    out = {}
    for field in ("wti", "brent"):
        result = run_chain(_build_providers(chains.get(field, []), start, end),
                           field_name=field, min_obs=20, chain_timeout_sec=120,
                           select="freshest")
        if result.ok:
            out[field] = result.payload.reindex(idx)
            kind = next((i["kind"] for i in chains[field] if i["name"] == result.used), "")
            ref = next((str(i["ref"]) for i in chains[field] if i["name"] == result.used), "")
            caliber = result.meta.get("caliber", "")
            meta[field] = {
                "source_name": result.used,
                "url": _provider_url(kind, ref),
                "frequency": "daily_business",
                "attempts": result.attempts,
                "caliber": caliber,
                "note": f"口径：{caliber}；新鲜度优先选源链：{result.trail_text()}",
            }
        else:
            out[field] = pd.Series(index=idx, dtype=float)   # 保持缺失，绝不造数
            meta[field] = {"source_name": "UNAVAILABLE", "url": "",
                           "frequency": "daily_business",
                           "attempts": result.attempts,
                           "note": f"全部价格源均失败：{result.trail_text()}"}
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
