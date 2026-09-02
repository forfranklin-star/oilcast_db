"""机构观点提取：高盛/摩根大通/瑞银/IEA/EIA/OPEC 等的目标价与方向。

实现路径：复用 Google News RSS 检索机构关键词，用正则从标题中抽取
"raises/lowers ... $NN ... brent/wti forecast" 类结构化信息。
官网月报（IEA OMR / EIA STEO / OPEC MOMR）提供 PDF/RSS，可在此扩展。
抓取不到时返回 None，由合成数据兜底，保证报告结构完整。
"""
from __future__ import annotations
import re
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote_plus

import pandas as pd

from ..utils import PoliteSession, get_logger

LOG = get_logger(__name__)

INSTITUTIONS = {
    "goldman": "高盛", "goldman sachs": "高盛",
    "jpmorgan": "摩根大通", "jp morgan": "摩根大通",
    "morgan stanley": "摩根士丹利", "ubs": "瑞银", "citi": "花旗",
    "barclays": "巴克莱", "iea": "IEA", "eia": "EIA", "opec": "OPEC",
}
RAISE_WORDS = re.compile(r"(raises?|raises? estimate|hikes?|boosts?|upgrades?|上调)", re.I)
LOWER_WORDS = re.compile(r"(lowers?|cuts? .{0,20}forecast|downgrades?|slashes?|下调)", re.I)
PRICE = re.compile(r"\$?\s?(\d{2,3}(?:\.\d)?)\s?(?:dollars|usd)?", re.I)


def parse_view(title: str) -> Optional[dict]:
    lower = title.lower()
    inst_cn = next((cn for en, cn in INSTITUTIONS.items() if en in lower), None)
    if inst_cn is None:
        return None
    is_raise = bool(RAISE_WORDS.search(title))
    is_lower = bool(LOWER_WORDS.search(title))
    m = PRICE.search(title)
    target = float(m.group(1)) if m else None
    stance = "看涨" if is_raise and not is_lower else ("看跌" if is_lower and not is_raise else "中性")
    if target is None and stance == "中性":
        return None
    return {"institution": inst_cn, "target_wti": target, "stance": stance, "note": title}


def fetch_institutional_views(as_of: datetime, lookback_days: int = 90) -> Optional[pd.DataFrame]:
    try:
        import feedparser
    except ImportError:
        return None
    from ..config import get_config
    sess = PoliteSession()
    since = (pd.Timestamp(as_of) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    queries = ["Goldman Sachs oil price forecast", "JPMorgan Brent forecast",
               "UBS oil price target", "IEA oil demand outlook", "OPEC oil demand"]
    rows = []
    for q in queries:
        url = (f"{get_config()['data_sources']['google_news_rss']}"
               f"?q={quote_plus(q + ' after:' + since)}&hl=en-US&gl=US&ceid=US:en")
        resp = sess.get(url)
        if resp is None:
            continue
        for ent in feedparser.parse(resp.text).entries:
            parsed = parse_view(ent.get("title", ""))
            if parsed is None:
                continue
            parsed["date"] = pd.Timestamp(datetime(*ent.published_parsed[:6])) \
                if ent.get("published_parsed") else pd.Timestamp(as_of)
            parsed["source"] = ent.get("source", {}).get("title", "Google News")
            rows.append(parsed)
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset=["note"])
    df = df.sort_values("date", ascending=False).head(20).reset_index(drop=True)
    LOG.info("真实机构观点提取：%d 条", len(df))
    return df
