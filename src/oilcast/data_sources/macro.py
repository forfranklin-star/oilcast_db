"""宏观与金融市场数据。

无 key 公开源：
  FRED（圣路易斯联储）CSV 端点 https://fred.stlouisfed.org/graph/fredgraph.csv?id=XXX
    DGS10     美国10年期国债收益率（日）
    DGS2      美国2年期国债收益率（日，用于推导政策预期）
    DTWEXBGS  美元广义名义指数（日，DXY 的官方代理）
    CPIAUCSL  CPI 水平（月，自算同比）
    PAYEMS    非农就业总人数（月，自算新增与意外度）
    FEDFUNDS  联邦基金有效利率（月）
  GPRD（Caldara & Iacoviello 日频地缘政治风险指数，公开 CSV）
DXY 本体优先用 yfinance 的 DX-Y.NYB。
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils import PoliteSession, get_logger
from .prices import fetch_yf_series

LOG = get_logger(__name__)

FRED_SERIES = {
    "us10y": "DGS10",
    "us2y": "DGS2",
    "dxy_fred": "DTWEXBGS",
    "cpi_level": "CPIAUCSL",
    "payems": "PAYEMS",
    "fedfunds": "FEDFUNDS",
}
# 日频 GPR 指数（若地址变动，不影响主流程，会回退到事件聚合自建 GPR）
GPRD_URL = "https://www.matteoiacoviello.com/research_files/gprd_web_current.csv"


def fetch_fred(start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.Series]:
    """逐个抓取 FRED 序列；单个失败不影响其他。"""
    from ..config import get_config
    base = str(get_config()["data_sources"]["fred_base_csv"])
    sess = PoliteSession()
    out: Dict[str, pd.Series] = {}
    for col, sid in FRED_SERIES.items():
        resp = sess.get(base, params={"id": sid,
                                      "cosd": start.strftime("%Y-%m-%d"),
                                      "coed": end.strftime("%Y-%m-%d")})
        if resp is None:
            continue
        try:
            df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
            s = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
            s.name = col
            out[col] = s
        except Exception as exc:
            LOG.warning("FRED %s 解析失败：%s", sid, exc)
    return out


def fetch_gpr(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    sess = PoliteSession()
    resp = sess.get(GPRD_URL)
    if resp is None:
        return None
    try:
        df = pd.read_csv(io.StringIO(resp.text))
        date_col = next(c for c in df.columns if c.lower() in ("date", "day"))
        val_col = next(c for c in df.columns if "gpr" in c.lower())
        s = pd.Series(df[val_col].values, index=pd.to_datetime(df[date_col]), name="gpr_index")
        return s.loc[(s.index >= start) & (s.index <= end)].sort_index()
    except Exception as exc:
        LOG.warning("GPRD 解析失败：%s", exc)
        return None


def fetch_macro(as_of: datetime, history_days: int) -> Optional[pd.DataFrame]:
    """组装统一宏观表；核心序列（10Y 或 DXY）全部缺失时返回 None。"""
    start = pd.Timestamp(as_of) - timedelta(days=history_days)
    end = pd.Timestamp(as_of)
    bdays = pd.bdate_range(start, end)

    fred = fetch_fred(start, end)
    dxy_yf = fetch_yf_series("DX-Y.NYB", start, end)   # 真实 DXY
    gpr = fetch_gpr(start, end)

    if "us10y" not in fred and dxy_yf is None:
        LOG.warning("FRED 与 yfinance 均不可用，宏观数据回退合成")
        return None

    frame = pd.DataFrame(index=bdays)
    if "us10y" in fred:
        frame["us10y"] = fred["us10y"].reindex(bdays).ffill()
    # DXY：优先 yfinance 本体，否则用 FRED 广义指数（量纲不同，标准化在特征层处理）
    if dxy_yf is not None:
        frame["dxy"] = dxy_yf.reindex(bdays).ffill()
    elif "dxy_fred" in fred:
        frame["dxy"] = fred["dxy_fred"].reindex(bdays).ffill()
    us2y = fred["us2y"].reindex(bdays).ffill() if "us2y" in fred else None

    # CPI 同比
    if "cpi_level" in fred:
        cpi = fred["cpi_level"].sort_index()
        cpi_yoy = cpi.pct_change(12) * 100
        frame["cpi_yoy"] = cpi_yoy.reindex(bdays, method="ffill")
    # 非农意外：新增就业相对过去 12 个月均值的标准差倍数（无一致预期数据时的代理）
    if "payems" in fred:
        chg = fred["payems"].diff()
        z = (chg - chg.rolling(12, min_periods=3).mean()) / chg.rolling(12, min_periods=3).std()
        frame["nonfarm_surprise"] = z.reindex(bdays, method="ffill") * 60  # 还原到"千人"量级
    # 降息预期代理：短端利率 20 日下行越快 → 降息预期越强（+1）
    if us2y is not None:
        fed_exp = -((us2y.diff(20)) / us2y.rolling(60).std()).clip(-2, 2) / 2
        frame["fed_expectation"] = fed_exp.reindex(bdays).ffill()
    elif "fedfunds" in fred:
        ff = fred["fedfunds"].reindex(bdays).ffill()
        frame["fed_expectation"] = -(ff.diff(20) / ff.rolling(60).std()).clip(-2, 2) / 2
    if gpr is not None:
        frame["gpr_index"] = gpr.reindex(bdays).ffill()

    frame = frame.dropna(how="all")
    if frame.shape[0] < 30:
        return None
    LOG.info("真实宏观数据就绪：%d 行，列=%s", len(frame), list(frame.columns))
    return frame
