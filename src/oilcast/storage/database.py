"""SQLite 持久化层：原始数据、特征、预测、权重、报告索引。

同时把每次采集的原始/处理后数据另存 CSV（data/raw、data/processed），
便于 git 历史回溯与离线审计。所有写入都是 upsert（INSERT OR REPLACE），
重复执行同一天的 pipeline 不会产生重复行。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..config import get_config, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_prices (
    date TEXT PRIMARY KEY, wti REAL, brent REAL, diesel REAL);
CREATE TABLE IF NOT EXISTS raw_macro (
    date TEXT PRIMARY KEY, dxy REAL, us10y REAL, cpi_yoy REAL,
    nonfarm_surprise REAL, fed_expectation REAL, demand_proxy REAL, gpr_index REAL);
CREATE TABLE IF NOT EXISTS events (
    date TEXT, title TEXT, source TEXT, theme TEXT, sentiment TEXT,
    intensity REAL, est_price_impact REAL, url TEXT,
    UNIQUE(date, title));
CREATE TABLE IF NOT EXISTS institutional_views (
    date TEXT, institution TEXT, target_wti REAL, stance TEXT, note TEXT,
    UNIQUE(date, note));
CREATE TABLE IF NOT EXISTS forecasts (
    report_date TEXT, horizon TEXT, instrument TEXT, target_date TEXT,
    mean REAL, q05 REAL, q25 REAL, q50 REAL, q75 REAL, q95 REAL,
    prob_up REAL, prob_down REAL);
CREATE TABLE IF NOT EXISTS factor_weights (
    report_date TEXT, factor TEXT, weight REAL, model_importance REAL,
    prior REAL, PRIMARY KEY(report_date, factor));
CREATE TABLE IF NOT EXISTS reports (
    report_date TEXT PRIMARY KEY, json_path TEXT, html_path TEXT, created_at TEXT);
"""


class OilCastDB:
    def __init__(self, path: Optional[str] = None) -> None:
        ensure_dirs()
        cfg = get_config()
        self.path = path or cfg["storage"]["sqlite_path"]
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ---------------------------------------------------------- 原始数据
    def _upsert_df(self, df: pd.DataFrame, table: str) -> None:
        if df is None or df.empty:
            return
        out = df.copy()
        out.index.name = "date"
        out = out.reset_index()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        cols = list(out.columns)
        placeholders = ",".join("?" * len(cols))
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
        with self._conn() as con:
            con.executemany(sql, out.where(pd.notna(out), None).values.tolist())

    def save_prices(self, prices: pd.DataFrame) -> None:
        self._upsert_df(prices[["wti", "brent", "diesel"]], "raw_prices")

    def save_macro(self, macro: pd.DataFrame) -> None:
        cols = [c for c in ["dxy", "us10y", "cpi_yoy", "nonfarm_surprise",
                            "fed_expectation", "demand_proxy", "gpr_index"] if c in macro.columns]
        self._upsert_df(macro[cols], "raw_macro")

    def save_events(self, events: pd.DataFrame) -> None:
        if events is None or events.empty:
            return
        with self._conn() as con:
            for _, r in events.iterrows():
                con.execute(
                    "INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?)",
                    (pd.Timestamp(r["date"]).strftime("%Y-%m-%d"), str(r.get("title", "")),
                     str(r.get("source", "")), str(r.get("theme", "")),
                     str(r.get("sentiment", "")), float(r.get("intensity", 0) or 0),
                     float(r.get("est_price_impact", 0) or 0), str(r.get("url", ""))))

    def save_views(self, views: pd.DataFrame) -> None:
        if views is None or views.empty:
            return
        with self._conn() as con:
            for _, r in views.iterrows():
                tgt = r.get("target_wti")
                con.execute(
                    "INSERT OR IGNORE INTO institutional_views VALUES (?,?,?,?,?)",
                    (pd.Timestamp(r["date"]).strftime("%Y-%m-%d"), str(r.get("institution", "")),
                     None if pd.isna(tgt) else float(tgt), str(r.get("stance", "")),
                     str(r.get("note", ""))))

    # ---------------------------------------------------------- 预测/权重
    def save_forecasts(self, report_date: str, records: list[dict]) -> None:
        with self._conn() as con:
            con.execute("DELETE FROM forecasts WHERE report_date=?", (report_date,))
            con.executemany(
                "INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [(report_date, r["horizon"], r["instrument"], r["target_date"],
                  r["mean"], r["q05"], r["q25"], r["q50"], r["q75"], r["q95"],
                  r.get("prob_up"), r.get("prob_down")) for r in records])

    def save_weights(self, report_date: str, weights: pd.DataFrame) -> None:
        with self._conn() as con:
            con.execute("DELETE FROM factor_weights WHERE report_date=?", (report_date,))
            con.executemany(
                "INSERT INTO factor_weights VALUES (?,?,?,?,?)",
                [(report_date, r["factor"], float(r["weight"]),
                  float(r.get("model_importance", 0) or 0), float(r.get("prior", 0) or 0))
                 for _, r in weights.iterrows()])

    def register_report(self, report_date: str, json_path: str, html_path: str) -> None:
        with self._conn() as con:
            con.execute("INSERT OR REPLACE INTO reports VALUES (?,?,?,?)",
                        (report_date, json_path, html_path,
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # ------------------------------------------------------------- 读取
    def read_table(self, table: str) -> pd.DataFrame:
        with self._conn() as con:
            df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY date", con,
                                   parse_dates=["date"])
        return df.set_index("date") if "date" in df.columns else df

    def list_report_dates(self) -> list[str]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT report_date FROM reports ORDER BY report_date DESC").fetchall()
        return [r[0] for r in rows]

    def weight_history(self, factor: str) -> pd.DataFrame:
        with self._conn() as con:
            return pd.read_sql_query(
                "SELECT report_date, weight FROM factor_weights WHERE factor=? ORDER BY report_date",
                con, params=(factor,))


def save_csv_snapshot(prices: pd.DataFrame, macro: pd.DataFrame, tag: str) -> Dict[str, str]:
    """把原始数据快照写到 data/raw，便于 git 追踪。"""
    cfg = get_config()
    raw = Path(cfg["storage"]["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    paths = {}
    p1, p2 = raw / f"prices_{tag}.csv", raw / f"macro_{tag}.csv"
    prices.to_csv(p1, encoding="utf-8-sig")
    macro.to_csv(p2, encoding="utf-8-sig")
    paths["prices"], paths["macro"] = str(p1), str(p2)
    return paths
