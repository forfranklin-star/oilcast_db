"""国内 0# 柴油价格【真实源钩子】。

数据原则：国内柴油批发/零售价必须来自可核验的真实发布（发改委价格监测中心、
生意社、金投网等）。在接入合规真实源之前，本模块返回 None，对应标的在 strict
模式下标记 unavailable —— 不按国际油价"推算"一个数字冒充观测。

接入方法：在 ``fetch_diesel_live`` 内实现目标站点解析，返回
``pd.Series(index=发布日期, values=人民币/吨)``，并在返回的 meta 中写明
来源名称、URL、频率；collector 会自动完成质量门评估与入库。
抓取前请检查 /robots.txt 与服务条款，使用 PoliteSession 控制频率。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from ..utils import get_logger

LOG = get_logger(__name__)


def fetch_diesel_live(as_of: datetime, history_days: int
                      ) -> Tuple[Optional[pd.Series], dict]:
    """真实国内柴油价格钩子（预留）。

    实现示例（生意社 0# 柴油价格曲线，选择器以站点实际结构为准）::

        from ..utils import PoliteSession
        sess = PoliteSession()
        resp = sess.get("https://www.100ppi.com/...")  # 先核验 robots 与条款
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.select_one("table.选择器")
        df = pd.read_html(str(table))[0]
        s = pd.Series(价格.values, index=pd.to_datetime(日期))
        meta = {"source_name": "生意社 0#柴油批发价",
                "url": "...", "frequency": "daily_business"}
        return s, meta

    在未接入前返回 (None, meta)，由上层标记 unavailable（不估算、不合成）。
    """
    meta = {"source_name": "UNAVAILABLE（国内柴油真实源待接入 diesel.fetch_diesel_live）",
            "url": "", "frequency": "daily_business"}
    return None, meta
