"""地缘政治与宏观突发事件挖掘。

数据源：Google News RSS（免费、无需 key、返回标题+时间+链接）。
处理：标题关键词 → 主题归类；多空词典 → 情感与强度；
再按主题的经验价格弹性估算"若该事件单独主导，当日约影响多少美元/桶"。

这是一个可解释的规则基线（transparent baseline）；后续可替换为
FinBERT 等金融情感模型，只需保持输出表结构不变。
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import pandas as pd

from ..config import get_config
from ..utils import PoliteSession, get_logger

LOG = get_logger(__name__)

# 主题关键词（小写匹配）；顺序即优先级
THEME_KEYWORDS: Dict[str, List[str]] = {
    "supply_disruption": ["opec", "production cut", "output cut", "supply disruption",
                          "sanction", "embargo", "halt export", "voluntary cut", "supply cut"],
    "geopolitical_risk": ["red sea", "houthi", "attack", "strike", "war", "military",
                          "iran", "israel", "gaza", "ukraine", "russia", "tanker",
                          "political unrest", "coup", "conflict", "escalat"],
    "demand_outlook": ["demand", "inventory", "stockpile", "iea", "slowdown",
                       "refinery", "crude stock", "trade"],
    "cpi_surprise": ["cpi", "inflation", "consumer price"],
    "jobs_surprise": ["nonfarm", "payroll", "jobs report", "unemployment"],
    "fed_policy_expectation": ["fed", "rate cut", "rate hike", "fomc", "powell",
                               "interest rate", "central bank"],
    "institutional_view": ["goldman", "jpmorgan", "morgan stanley", "ubs", "citi",
                           "forecast", "price target", "raises estimate", "cuts estimate"],
}

# 利多 / 利空词强度（对油价方向）
BULLISH = {"surge": 1.0, "soar": 1.0, "rally": 0.8, "jump": 0.7, "spike": 0.9,
           "disruption": 1.0, "attack": 0.9, "sanction": 0.9, "halt": 0.7,
           "cut supply": 1.0, "production cut": 1.0, "shortage": 0.9,
           "escalat": 0.8, "raise": 0.5, "upgrade": 0.6, "rate cut": 0.8,
           "tighter": 0.6, "drop in export": 0.8}
BEARISH = {"plunge": 1.0, "slump": 0.9, "tumble": 0.9, "fall": 0.5, "drop": 0.4,
           "surplus": 0.8, "recession": 0.9, "weak demand": 1.0, "rate hike": 0.8,
           "downgrade": 0.7, "lower forecast": 0.9, "output raise": 0.9,
           "pump more": 0.8, "ease supply": 0.6, "risk-off": 0.5}

# 主题经验弹性：强度=1 时的单日价格影响量级（美元/桶）
THEME_ELASTICITY = {
    "supply_disruption": 3.5, "geopolitical_risk": 3.0, "demand_outlook": 1.8,
    "cpi_surprise": 1.4, "jobs_surprise": 1.2, "fed_policy_expectation": 1.6,
    "institutional_view": 2.0,
}


def classify_theme(title_lower: str) -> str:
    for theme, kws in THEME_KEYWORDS.items():
        if any(kw in title_lower for kw in kws):
            return theme
    return "demand_outlook"


def score_title(title_lower: str) -> tuple[float, float]:
    """返回（方向净值, 强度0~1）。"""
    bull = sum(w for kw, w in BULLISH.items() if kw in title_lower)
    bear = sum(w for kw, w in BEARISH.items() if kw in title_lower)
    net = bull - bear
    intensity = min(1.0, abs(net) / 2.5)
    return net, intensity


def fetch_events(as_of: datetime, lookback_days: int = 30) -> Optional[pd.DataFrame]:
    """从 Google News RSS 拉取并打分；网络不可用返回 None。"""
    try:
        import feedparser
    except ImportError:
        LOG.warning("未安装 feedparser，跳过事件抓取")
        return None

    cfg = get_config()
    sess = PoliteSession()
    keywords = list(cfg["data_sources"]["news_keywords"])
    since = (pd.Timestamp(as_of) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    rows, seen = [], set()

    for kw in keywords:
        q = f"{kw} after:{since}"
        url = f"{cfg['data_sources']['google_news_rss']}?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
        resp = sess.get(url)
        if resp is None:
            continue
        feed = feedparser.parse(resp.text)
        for ent in feed.entries:
            title = ent.get("title", "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            lower = title.lower()
            theme = classify_theme(lower)
            net, intensity = score_title(lower)
            if intensity < 0.05:        # 过滤无明显多空信息的标题
                continue
            elasticity = THEME_ELASTICITY.get(theme, 1.5)
            impact = (1 if net >= 0 else -1) * intensity * elasticity
            rows.append({
                "date": pd.Timestamp(ent.get("published_parsed") and
                                     datetime(*ent.published_parsed[:6]) or as_of),
                "title": title,
                "source": ent.get("source", {}).get("title", "Google News"),
                "theme": theme,
                "sentiment": "positive" if net > 0 else ("negative" if net < 0 else "neutral"),
                "intensity": round(intensity, 3),
                "est_price_impact": round(impact, 2),
                "url": ent.get("link", ""),
            })

    if not rows:
        return None
    ev = pd.DataFrame(rows)
    ev = ev[ev["date"] >= pd.Timestamp(as_of) - timedelta(days=lookback_days)]
    ev = ev.sort_values("date", ascending=False).head(40).reset_index(drop=True)
    LOG.info("真实新闻事件挖掘完成：%d 条", len(ev))
    return ev
