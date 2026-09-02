"""数据源统一契约。

每个真实 fetcher 返回 DataFrame；抓取失败时由 pipeline 调用 synthetic 中的
同结构生成器兜底。``provenance`` 记录每个字段来自 live 还是 simulated，
报告页面会如实展示，避免把模拟数据误当真实行情。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass
class DataBundle:
    """一次数据采集的完整结果。

    prices  列: wti, brent, diesel（日频，工作日）
    macro   列: dxy, us10y, cpi_yoy, nonfarm_surprise, fed_expectation, demand_proxy
    events  列: date, title, source, theme, sentiment, intensity, est_price_impact
    views   列: date, institution, target_wti, stance, note
    """

    prices: pd.DataFrame
    macro: pd.DataFrame
    events: pd.DataFrame
    views: pd.DataFrame
    provenance: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"价格表 {self.prices.index.min().date()}~{self.prices.index.max().date()} 共{len(self.prices)}行"]
        lines.append(f"宏观表 {len(self.macro)}行；事件 {len(self.events)} 条；机构观点 {len(self.views)} 条")
        lines.append("数据来源：" + "，".join(f"{k}={v}" for k, v in self.provenance.items()))
        return "\n".join(lines)
