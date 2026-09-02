"""配置加载与项目路径管理。

用法::

    from oilcast.config import get_config, PROJECT_ROOT
    cfg = get_config()
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# src/oilcast/config.py -> 上溯三级即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _deep_freeze(obj: Any) -> Any:
    """把 dict/list 递归转成不可变结构，防止运行期误改全局配置。"""
    if isinstance(obj, dict):
        return DictProxy({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_deep_freeze(v) for v in obj)
    return obj


class DictProxy(dict):
    """支持属性访问的只读字典（cfg.model.rolling_window）。"""

    def __getattr__(self, item):  # noqa: D401
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        raise TypeError("config is read-only")


@lru_cache(maxsize=4)
def get_config(path: str | os.PathLike | None = None) -> DictProxy:
    """加载并缓存全局配置。"""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    cfg = _deep_freeze(raw)

    # 把配置中的相对路径统一解析为项目根下的绝对路径
    storage = cfg["storage"]
    for key in ("sqlite_path", "raw_dir", "processed_dir", "archive_dir", "latest_dir"):
        storage[key] = str(PROJECT_ROOT / storage[key]) if not Path(storage[key]).is_absolute() else storage[key]
    return cfg


def ensure_dirs() -> None:
    """确保数据/报告目录存在。"""
    cfg = get_config()
    for key in ("raw_dir", "processed_dir", "archive_dir", "latest_dir"):
        Path(cfg["storage"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["storage"]["sqlite_path"]).parent.mkdir(parents=True, exist_ok=True)
