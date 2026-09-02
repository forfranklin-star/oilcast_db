"""通用工具：日志、带退避的 HTTP 会话、交易日工具、确定性随机。"""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests

from .config import get_config


# ---------------------------------------------------------------- logging
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ------------------------------------------------------------- randomness
def seeded_rng() -> np.random.Generator:
    seed = int(get_config()["system"]["random_seed"])
    return np.random.default_rng(seed)


# ----------------------------------------------------------- http session
class PoliteSession:
    """带请求头、重试、固定延时的 requests 会话（爬虫礼仪）。

    - 统一 User-Agent，便于站点管理员联系；
    - 每次请求之间 sleep，降低目标站点压力；
    - 仅对 5xx / 网络异常做有限次指数退避重试。
    """

    def __init__(self) -> None:
        cfg = get_config()["system"]["request"]
        self.timeout = int(cfg["timeout_sec"])
        self.retry = int(cfg["retry"])
        self.sleep_sec = float(cfg["sleep_between_requests_sec"])
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": str(cfg["user_agent"]),
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        })

    def get(self, url: str, params: Optional[dict] = None,
            encoding: Optional[str] = None) -> Optional[requests.Response]:
        for attempt in range(self.retry + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if encoding:
                    resp.encoding = encoding
                resp.raise_for_status()
                time.sleep(self.sleep_sec)
                return resp
            except requests.RequestException as exc:
                if attempt == self.retry:
                    get_logger(__name__).warning("GET 失败 %s: %s", url, exc)
                    return None
                backoff = self.sleep_sec * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(backoff)
        return None


# ------------------------------------------------------------ timeout guard
def run_with_timeout(func, args=(), kwargs=None, timeout_sec: float = 25):
    """对不受本项目 session 控制的第三方库调用（如 yfinance）加总耗时保护，
    超时即返回 None，让 pipeline 快速降级到兜底数据，而不是长时间挂起。"""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
    # 注意：不能用 with —— __exit__ 会 shutdown(wait=True) 把超时线程等完，保护失效。
    # 遗留的 worker 为 daemon 线程，解释器退出时自动回收。
    ex = ThreadPoolExecutor(max_workers=1)
    future = ex.submit(func, *args, **(kwargs or {}))
    try:
        return future.result(timeout=timeout_sec)
    except FutureTimeout:
        get_logger(__name__).warning("调用 %s 超过 %.0fs，已降级", getattr(func, "__name__", func), timeout_sec)
        return None
    finally:
        ex.shutdown(wait=False)


# ------------------------------------------------------------- date utils
def now_beijing() -> datetime:
    """当前北京时间（UTC+8）。"""
    return datetime.now(timezone(timedelta(hours=8)))


def business_day_index(end: datetime, n_periods: int) -> "pd.DatetimeIndex":  # noqa: F821
    """以 end 结尾、长度 n_periods 的工作日索引（周一至周五）。"""
    import pandas as pd
    return pd.bdate_range(end=end.normalize(), periods=n_periods)


def trading_offsets(n_days: int) -> int:
    """自然日近似换算为交易日（*5/7，至少 1）。"""
    return max(1, int(round(n_days * 5 / 7)))
