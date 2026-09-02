"""国内 0# 柴油价格。

真实渠道（金投网 / 生意社 / 发改委价格监测中心）页面结构变动频繁且多有反爬，
本模块保留**可替换的抓取钩子** ``fetch_diesel_live``：接入时只需在该函数内
实现站点解析并返回 ``pd.Series``（元/吨，日频）。

默认路径：按"布伦特原油 → 到岸完税成本 → 批发价"的成本传导公式估算，
provenance 标注为 ``estimated``，与真实抓取 / 合成兜底明确区分。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ..config import get_config
from ..utils import get_logger

LOG = get_logger(__name__)


def fetch_diesel_live(as_of: datetime, history_days: int) -> Optional[pd.Series]:
    """真实国内柴油批发价抓取钩子（预留）。

    接入示例（生意社 0#柴油价格曲线）::

        sess = PoliteSession()
        resp = sess.get("https://www.100ppi.com/sf/day-XXXX.html")  # 具体地址按站点更新
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.select_one("table.???")          # 用浏览器检查后填写选择器
        df = pd.read_html(str(table))[0]
        return pd.Series(...)  # index=日期, 值=元/吨

    注意：抓取前先检查目标站点 /robots.txt 与服务条款，控制频率。
    """
    # 显式返回 None → 上层自动使用成本估算 / 合成兜底
    return None


def estimate_from_brent(brent: pd.Series) -> pd.Series:
    """成本传导估算：元/吨 = 布伦特($/桶) × 汇率 × 桶吨换算 + 税费物流。

    发改委成品油调价以 10 个工作日为周期，用 10 日滚动均值近似调价的阶梯平滑。
    """
    inst = get_config()["instruments"]["diesel"]
    fx, bpt, tax = (float(inst["fx_usdcny"]),
                    float(inst["barrel_per_tonne_diesel"]),
                    float(inst["tax_and_logistics"]))
    raw = (brent * fx * bpt + tax).rolling(10, min_periods=1).mean()
    return raw.rename("diesel").round(1)


def fetch_diesel(as_of: datetime, history_days: int,
                 brent: Optional[pd.Series] = None) -> Optional[pd.Series]:
    live = fetch_diesel_live(as_of, history_days)
    if live is not None and len(live) >= 20:
        LOG.info("使用真实国内柴油价格，%d 条", len(live))
        return live
    if brent is not None and len(brent.dropna()) >= 20:
        LOG.info("使用布伦特成本传导估算国内柴油价格（estimated）")
        return estimate_from_brent(brent.dropna())
    return None
