"""SQLite 持久化层：原始数据、谱系、事件、预测、权重、报告索引。

可追溯设计：
- raw_prices/raw_macro 每行带 mode（strict/demo），严格区分真实与演示；
- data_lineage 记录每个字段每次报告的来源、抓取时刻、首末观测、质量门状态；
- 所有写入 upsert，重复执行同一天不产生重复行。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..config import get_config, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_prices (
    date TEXT, mode TEXT, wti REAL, brent REAL, diesel REAL,
    PRIMARY KEY(date, mode));
CREATE TABLE IF NOT EXISTS raw_macro (
    date TEXT, mode TEXT, dxy REAL, us10y REAL, us2y REAL, cpi_yoy REAL,
    nonfarm_surprise REAL, fedfunds REAL, fed_expectation REAL,
    demand_proxy REAL, gpr_index REAL, PRIMARY KEY(date, mode));
CREATE TABLE IF NOT EXISTS data_lineage (
    report_date TEXT, field TEXT, display TEXT, status TEXT, source_name TEXT,
    url TEXT, frequency TEXT, retrieved_at TEXT, n_obs INTEGER,
    first_observed TEXT, last_observed TEXT, stale INTEGER, mode TEXT, note TEXT,
    tried_sources TEXT, PRIMARY KEY(report_date, field));
CREATE TABLE IF NOT EXISTS events (
    date TEXT, title TEXT, source TEXT, theme TEXT, sentiment TEXT,
    intensity REAL, est_price_impact REAL, url TEXT, UNIQUE(date, title));
CREATE TABLE IF NOT EXISTS institutional_views (
    date TEXT, institution TEXT, target_wti REAL, stance TEXT, note TEXT,
    UNIQUE(date, note));
CREATE TABLE IF NOT EXISTS forecasts (
    report_date TEXT, horizon TEXT, instrument TEXT, target_date TEXT,
    mean REAL, q05 REAL, q25 REAL, q50 REAL, q75 REAL, q95 REAL,
    prob_up REAL, prob_down REAL);
CREATE TABLE IF NOT EXISTS factor_weights (
    report_date TEXT, factor TEXT, weight REAL, model_importance REAL,
    prior REAL, available INTEGER, PRIMARY KEY(report_date, factor));
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

    def _upsert_df(self, df: pd.DataFrame, table: str, mode: str, keep_cols) -> None:
        if df is None or df.empty:
            return
        cols = [c for c in keep_cols if c in df.columns]
        out = df[cols].copy()
        out.insert(0, "mode", mode)
        out.index.name = "date"
        out = out.reset_index()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        placeholders = ",".join("?" * len(out.columns))
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(out.columns)}) VALUES ({placeholders})"
        with self._conn() as con:
            con.executemany(sql, out.where(pd.notna(out), None).values.tolist())

    def save_prices(self, prices: pd.DataFrame, mode: str = "strict",
                    lineage: Optional[dict] = None) -> None:
        self._upsert_df(prices, "raw_prices", mode, ["wti", "brent", "diesel"])

    def save_macro(self, macro: pd.DataFrame, mode: str = "strict",
                   lineage: Optional[dict] = None) -> None:
        self._upsert_df(macro, "raw_macro", mode,
                        ["dxy", "us10y", "us2y", "cpi_yoy", "nonfarm_surprise",
                         "fedfunds", "fed_expectation", "demand_proxy", "gpr_index"])

    def save_lineage(self, report_date: str, lineage: Dict[str, dict], mode: str) -> None:
        with self._conn() as con:
            con.execute("DELETE FROM data_lineage WHERE report_date=?", (report_date,))
            con.executemany(
                "INSERT INTO data_lineage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(report_date, f, L.get("display", ""), L.get("status", ""),
                  L.get("source_name", ""), L.get("url", ""), L.get("frequency", ""),
                  L.get("retrieved_at", ""), int(L.get("n_obs", 0) or 0),
                  L.get("first_observed"), L.get("last_observed"),
                  L.get("stale"), mode, L.get("note", ""), L.get("tried_sources", ""))
                 for f, L in lineage.items()])

    def save_events(self, events: pd.DataFrame) -> None:
        if events is None or events.empty:
            return
        with self._conn() as con:
            for _, r in events.iterrows():
                con.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?)",
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
                con.execute("INSERT OR IGNORE INTO institutional_views VALUES (?,?,?,?,?)",
                            (pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                             str(r.get("institution", "")),
                             None if pd.isna(tgt) else float(tgt),
                             str(r.get("stance", "")), str(r.get("note", ""))))

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
            for _, r in weights.iterrows():
                w = r.get("weight")
                mi = r.get("model_importance")
                pr = r.get("prior")
                con.execute("INSERT INTO factor_weights VALUES (?,?,?,?,?,?)",
                            (report_date, r["factor"],
                             None if pd.isna(w) else float(w),
                             None if pd.isna(mi) else float(mi),
                             None if pd.isna(pr) else float(pr),
                             1 if r.get("available", False) else 0))

    def register_report(self, report_date: str, json_path: str, html_path: str) -> None:
        with self._conn() as con:
            con.execute("INSERT OR REPLACE INTO reports VALUES (?,?,?,?)",
                        (report_date, json_path, html_path,
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def read_table(self, table: str) -> pd.DataFrame:
        with self._conn() as con:
            df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY date", con)
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
    cfg = get_config()
    raw = Path(cfg["storage"]["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    paths = {}
    p1, p2 = raw / f"prices_{tag}.csv", raw / f"macro_{tag}.csv"
    prices.to_csv(p1, encoding="utf-8-sig")
    macro.to_csv(p2, encoding="utf-8-sig")
    paths["prices"], paths["macro"] = str(p1), str(p2)
    return paths
