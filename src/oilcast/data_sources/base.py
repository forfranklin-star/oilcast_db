"""数据统一契约。

strict 模式：prices/macro 只包含真实观测（缺口保留 NaN，绝不填合成值）；
lineage 逐字段记录来源、观测日期、抓取时刻与质量门状态。
demo 模式：全部为合成数据，mode="demo"，报告必须全程显示演示水印。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

# strict 模式允许进入模型的质量状态
USABLE_STATES = {"ok"}


@dataclass
class DataBundle:
    """一次数据采集的完整结果。

    prices  列: wti, brent（真实现货；柴油仅在接入可核验真实源时出现）
    macro   列: dxy, us10y, cpi_yoy, nonfarm_surprise, fed_expectation,
               demand_proxy, gpr_index（缺口保留 NaN）
    vintage 月频对齐值的原始发布日期（逐点可追溯）
    events/views: 真实抓取结果；不可达时为空表，绝不伪造
    lineage 字段 -> FieldLineage.to_dict()
    mode    "strict" | "demo"
    """

    prices: pd.DataFrame
    macro: pd.DataFrame
    events: pd.DataFrame
    views: pd.DataFrame
    lineage: Dict[str, dict] = field(default_factory=dict)
    vintage: Dict[str, pd.Series] = field(default_factory=dict)
    mode: str = "strict"

    def field_status(self, name: str) -> str:
        return self.lineage.get(name, {}).get("status", "unavailable")

    def is_usable(self, name: str) -> bool:
        """字段是否通过质量门、允许进入模型（demo 模式除外，由调用方显式承担）。"""
        return self.mode == "demo" or self.field_status(name) in USABLE_STATES

    def summary(self) -> str:
        lines = [f"[{self.mode}] 价格 {len(self.prices)} 行、宏观 {len(self.macro)} 行、"
                 f"事件 {len(self.events)} 条、机构观点 {len(self.views)} 条"]
        for k, L in self.lineage.items():
            lines.append(f"  - {k:16s} {L['status']:12s} n={L['n_obs']:>4} "
                         f"last={L['last_observed']} src={L['source_name']}")
        return "\n".join(lines)
