"""多数据源优先级链（failover chain）统一框架。

设计原则（与全局数据原则一致）：
- 每个字段配置一张【有序 provider 列表】，按优先级依次尝试；
- 第一个返回足量【真实】观测的 provider 被采用，不同源之间绝不混合拼接；
- 每次尝试（源名、是否成功、观测条数、耗时、失败原因）都记录在案，写入数据
  谱系，做到"为什么用了这个源、前面的源为什么失败"完全可审计；
- 所有源都失败才返回 unavailable，由上层如实标记，严禁造数补齐。

provider 约定：一个无参可调用（用闭包/functools.partial 绑定参数），
成功返回 (payload, meta)，payload 为 pd.Series / pd.DataFrame；
无数据或不可达时返回 None，或直接抛异常（本框架会捕获并记录原因）。
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple
import pandas as pd
from ..utils import get_logger

LOG = get_logger(__name__)

# 部分政府/门户站点会丢弃自报机器人身份或缺少常规浏览器头的请求，
# 对这类源统一使用浏览器兼容头（仍是低速、可识别的善意研究访问）。
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


@dataclass
class Attempt:
    source: str
    ok: bool = False
    n_obs: int = 0
    elapsed_ms: int = 0
    reason: str = ""

    def line(self) -> str:
        if self.ok:
            return f"{self.source}✓{self.n_obs}条/{self.elapsed_ms}ms"
        return f"{self.source}✗({self.reason})"


@dataclass
class ChainResult:
    payload: Any = None
    used: Optional[str] = None
    meta: dict = field(default_factory=dict)
    attempts: List[Attempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.payload is not None

    def trail_text(self) -> str:
        """人类可读的尝试链，写入谱系 note，例如 A✗(超时) → B✓740条。"""
        return " → ".join(a.line() for a in self.attempts)


def _count(payload) -> int:
    if payload is None:
        return 0
    if isinstance(payload, pd.Series):
        return int(payload.dropna().shape[0])
    if isinstance(payload, pd.DataFrame):
        return int(len(payload))
    return 1


def run_chain(providers: List[Tuple[str, Callable[[], Optional[Tuple[Any, dict]]]]],
              field_name: str = "",
              min_obs: int = 1,
              chain_timeout_sec: float = 90.0) -> ChainResult:
    """按优先级依次尝试 providers。

    providers: [(源显示名, 零参 callable -> (payload, meta) | None), ...]
    min_obs:    payload 至少要有多少真实观测才算成功（防止拿到空壳/残缺源）。
    返回 ChainResult；单个 provider 异常不影响后续源尝试。
    """
    result = ChainResult()
    chain_start = time.time()
    for name, fn in providers:
        if time.time() - chain_start > chain_timeout_sec:
            result.attempts.append(Attempt(name, False, reason="整条链超时，跳过后续源"))
            LOG.warning("[%s] 源链总耗时超 %.0fs，停止尝试", field_name, chain_timeout_sec)
            break
        att = Attempt(source=name)
        t0 = time.time()
        try:
            got = fn()
            payload, meta = got if isinstance(got, tuple) else (got, {})
            att.elapsed_ms = int((time.time() - t0) * 1000)
            n = _count(payload)
            if payload is None or n < min_obs:
                att.ok, att.reason = False, f"观测不足({n}<{min_obs})" if payload is not None else "无返回"
                LOG.info("[%s] 源 %s 不可用：%s", field_name, name, att.reason)
            else:
                att.ok, att.n_obs = True, n
                result.payload, result.used, result.meta = payload, name, (meta or {})
                result.attempts.append(att)
                LOG.info("[%s] 命中源 %s（%d 条, %dms）", field_name, name, n, att.elapsed_ms)
                break
        except Exception as exc:  # 单源失败不得拖垮整条链
            att.elapsed_ms = int((time.time() - t0) * 1000)
            att.reason = f"{type(exc).__name__}: {str(exc)[:80]}"
            LOG.warning("[%s] 源 %s 异常：%s", field_name, name, att.reason)
        result.attempts.append(att)
    if not result.ok:
        LOG.warning("[%s] 全部 %d 个源均失败，标记 unavailable：%s",
                    field_name, len(providers), result.trail_text())
    return result
