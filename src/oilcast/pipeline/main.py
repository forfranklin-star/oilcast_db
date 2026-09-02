"""每日主流程：采集 → 存储 → 特征 → 权重/模型 → 预测 → 回测 → 报告存档。

命令行::

    python -m oilcast.pipeline.main                # 正常运行（优先真实数据）
    python -m oilcast.pipeline.main --offline      # 强制使用合成数据（演示/CI）
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import get_config, ensure_dirs
from ..data_sources.collector import collect
from ..features.engineering import build_features, make_supervised
from ..models.evaluation import backtest_short
from ..models.long_term import LongTermMonteCarlo, scenario_probabilities
from ..models.mid_term import MidTermVAR
from ..models.short_term import ShortTermForecaster
from ..models.weights import learn_factor_weights
from ..reporting import narratives as narr
from ..reporting.static_html import render_static_html
from ..storage.database import OilCastDB, save_csv_snapshot
from ..utils import get_logger, now_beijing

LOG = get_logger(__name__)
TARGETS = {"wti": ("WTI原油", "美元/桶"),
           "brent": ("布伦特原油", "美元/桶"),
           "diesel": ("国内0#柴油批发价", "元/吨")}


# ----------------------------------------------------------- 序列化工具
def _path_records(path: pd.DataFrame) -> list:
    out = path.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    return out.round(2).to_dict("records")


def _hist_records(prices: pd.DataFrame, n: int = 180) -> list:
    p = prices.tail(n).copy()
    p.insert(0, "date", p.index.strftime("%Y-%m-%d"))
    return p.round(2).to_dict("records")


def _events_records(ev: pd.DataFrame, n: int = 30) -> list:
    if ev is None or ev.empty:
        return []
    e = ev.head(n).copy()
    e["date"] = pd.to_datetime(e["date"]).dt.strftime("%Y-%m-%d")
    return e.to_dict("records")


def _views_records(v: pd.DataFrame, n: int = 12) -> list:
    if v is None or v.empty:
        return []
    v = v.head(n).copy()
    v["date"] = pd.to_datetime(v["date"]).dt.strftime("%Y-%m-%d")
    return v.replace({np.nan: None}).to_dict("records")


def future_index(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=periods)


# --------------------------------------------------------------- 主流程
def run(as_of: Optional[datetime] = None, offline: bool = False) -> dict:
    cfg = get_config()
    ensure_dirs()
    as_of = pd.Timestamp(as_of or now_beijing())
    if as_of.tzinfo is not None:      # 全流程统一 tz-naive（内部价格索引均为 naive）
        as_of = as_of.tz_convert("Asia/Shanghai").tz_localize(None)
    report_date = as_of.strftime("%Y-%m-%d")
    LOG.info("===== OilCast 每日报告开始：%s =====", report_date)

    # 1) 数据采集与落库
    bundle = collect(as_of=as_of, prefer_live=False if offline else None)
    LOG.info(bundle.summary())
    db = OilCastDB()
    db.save_prices(bundle.prices)
    db.save_macro(bundle.macro)
    db.save_events(bundle.events)
    db.save_views(bundle.views)
    save_csv_snapshot(bundle.prices, bundle.macro, report_date)

    prices, macro, events, views = bundle.prices, bundle.macro, bundle.events, bundle.views
    mcfg = cfg["model"]
    h_short, h_mid, h_long = (int(mcfg["short_horizon_td"]),
                              int(mcfg["mid_horizon_td"]),
                              int(mcfg["long_horizon_td"]))
    last_date = prices.index[-1]

    # 2) 特征矩阵（每个预测目标一套技术面；因素权重以 WTI 为主定价品种学习）
    feats = {t: build_features(prices, macro, events, views, target=t) for t in TARGETS}
    feats["wti"].to_csv(Path(cfg["storage"]["processed_dir"]) / f"features_{report_date}.csv",
                        encoding="utf-8-sig")
    X_w, y_w = make_supervised(feats["wti"], prices["wti"], h_short)

    # 3) 因素权重（与上一轮 EMA 自适应）
    prev = None
    hist_dates = db.list_report_dates()
    if hist_dates:
        prev_tbl = pd.read_sql_query(
            "SELECT factor, weight FROM factor_weights WHERE report_date=?",
            __import__("sqlite3").connect(cfg["storage"]["sqlite_path"]),
            params=(hist_dates[0],))
        prev = prev_tbl.set_index("factor")["weight"] if not prev_tbl.empty else None
    weights = learn_factor_weights(X_w, y_w, prev_weights=prev)
    db.save_weights(report_date, weights)

    # 4) 短期：三标的 Direct 多步
    short_out: Dict[str, dict] = {}
    for t, (cn, unit) in TARGETS.items():
        fc = ShortTermForecaster(horizon=h_short, window=int(mcfg["rolling_window"]))
        fc.fit(feats[t], prices[t])
        res = fc.predict(feats[t].iloc[[-1]].drop(columns=[]),
                         float(prices[t].iloc[-1]),
                         future_index(last_date, h_short))
        short_out[t] = {"endpoint": res.endpoint, "path": _path_records(res.path),
                        "arima_benchmark": res.benchmark_endpoint}
    LOG.info("短期预测完成")

    # 5) 中期：VAR（WTI），其余按协整/成本关系映射
    var = MidTermVAR().fit(prices, macro)
    latest_f = feats["wti"].iloc[-1]
    sc_probs = scenario_probabilities(latest_f)
    mid_wti = var.predict(future_index(last_date, h_mid), sc_probs, cfg["scenarios"])
    mid_out = _map_related(mid_wti, prices)
    LOG.info("中期VAR预测完成，情景概率=%s", sc_probs)

    # 6) 长期：情景 + 蒙特卡洛
    anchor = None
    if views is not None and not views.empty and "target_wti" in views.columns:
        vals = pd.to_numeric(views["target_wti"], errors="coerce").dropna()
        anchor = round(float(vals.median()), 1) if len(vals) else None
    lt = LongTermMonteCarlo().fit(float(prices["wti"].iloc[-1]))
    long_wti = lt.predict(future_index(last_date, h_long), sc_probs, institution_anchor=anchor)
    long_out = _map_related(long_wti, prices)
    LOG.info("长期蒙特卡洛预测完成")

    # 7) 短期模型滚动回测
    bt = backtest_short(feats["wti"], prices["wti"], horizon=h_short)

    # 8) 汇总报告
    report = {
        "report_date": report_date,
        "generated_at": now_beijing().strftime("%Y-%m-%d %H:%M:%S %z"),
        "provenance": bundle.provenance,
        "current": {t: round(float(prices[t].iloc[-1]), 2) for t in TARGETS},
        "units": {t: TARGETS[t][1] for t in TARGETS},
        "names": {t: TARGETS[t][0] for t in TARGETS},
        "history": _hist_records(prices),
        "forecasts": {
            "short": {t: short_out[t] for t in TARGETS},
            "mid": mid_out,
            "long": long_out,
        },
        "weights": weights.to_dict("records"),
        "events": _events_records(events),
        "views": _views_records(views),
        "backtest": bt,
        "narratives": _build_narratives(prices, short_out, mid_out, long_out,
                                        weights, events, bt, bundle.provenance),
    }

    # 9) 预测明细落库 + JSON/HTML 存档
    _persist_forecasts(db, report_date, report)
    archive = Path(cfg["storage"]["archive_dir"])
    latest = Path(cfg["storage"]["latest_dir"])
    json_path = archive / f"{report_date}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    (latest / "latest.json").write_text(json.dumps(report, ensure_ascii=False, default=str),
                                        encoding="utf-8")
    html = render_static_html(report)
    html_path = archive / f"{report_date}.html"
    html_path.write_text(html, encoding="utf-8")
    (latest / "index.html").write_text(html, encoding="utf-8")
    db.register_report(report_date, str(json_path), str(html_path))
    LOG.info("报告已存档：%s / %s", json_path, html_path)
    return report


def _map_related(wti_result: dict, prices: pd.DataFrame) -> dict:
    """把 WTI 的中/长期路径映射到 Brent（协整价差）与柴油（成本传导）。"""
    cfg = get_config()
    path = wti_result["path"]
    ratio_b = float((prices["brent"] / prices["wti"]).tail(20).median())
    inst = cfg["instruments"]["diesel"]
    fx, bpt, tax = float(inst["fx_usdcny"]), float(inst["barrel_per_tonne_diesel"]), \
        float(inst["tax_and_logistics"])
    out = {"wti": {"endpoint": wti_result["endpoint"], "path": _path_records(path)}}
    brent_path = path * ratio_b
    diesel_path = brent_path * fx * bpt + tax
    for name, p in (("brent", brent_path), ("diesel", diesel_path)):
        last_now = float(prices[name].iloc[-1])
        end_mean = float(p["mean"].iloc[-1])
        ep = {
            "target_date": wti_result["endpoint"]["target_date"],
            "mean": round(end_mean, 2),
            "q05": round(float(p["q05"].iloc[-1]), 2),
            "q25": round(float(p["q25"].iloc[-1]), 2),
            "q75": round(float(p["q75"].iloc[-1]), 2),
            "q95": round(float(p["q95"].iloc[-1]), 2),
            "pct_mean": round((end_mean / last_now - 1) * 100, 2),
            "prob_up": wti_result["endpoint"]["prob_up"],
            "prob_down": wti_result["endpoint"]["prob_down"],
        }
        if "checkpoints" in wti_result["endpoint"]:
            ep["checkpoints"] = wti_result["endpoint"]["checkpoints"]
            ep["scenario_probs"] = wti_result["endpoint"]["scenario_probs"]
            ep["institution_anchor"] = wti_result["endpoint"].get("institution_anchor")
        out[name] = {"endpoint": ep, "path": _path_records(p)}
    return out


def _build_narratives(prices, short_out, mid_out, long_out, weights, events, bt, provenance):
    n = {
        "trends": {t: narr.trend_narrative(prices[t], TARGETS[t][0], TARGETS[t][1])
                   for t in TARGETS},
        "short": {t: narr.forecast_narrative(short_out[t]["endpoint"], TARGETS[t][0],
                                             TARGETS[t][1], "两周") for t in TARGETS},
        "mid": {t: narr.forecast_narrative(mid_out[t]["endpoint"], TARGETS[t][0],
                                           TARGETS[t][1], "三个月") for t in TARGETS},
        "long_wti": narr.scenario_narrative(long_out["wti"]["endpoint"]),
        "weights": narr.weights_narrative(weights),
        "events": narr.events_narrative(events),
        "backtest": narr.backtest_narrative(bt),
        "sources": narr.source_narrative(provenance),
    }
    return n


def _persist_forecasts(db: OilCastDB, report_date: str, report: dict) -> None:
    records = []
    for horizon, pack in report["forecasts"].items():
        for inst, item in pack.items():
            ep = item["endpoint"]
            records.append({
                "horizon": horizon, "instrument": inst,
                "target_date": ep["target_date"], "mean": ep["mean"],
                "q05": ep["q05"], "q25": ep["q25"], "q50": ep.get("q50", ep["mean"]),
                "q75": ep["q75"], "q95": ep["q95"],
                "prob_up": ep.get("prob_up"), "prob_down": ep.get("prob_down")})
    db.save_forecasts(report_date, records)


def main() -> None:
    parser = argparse.ArgumentParser(description="OilCast 每日报告流水线")
    parser.add_argument("--offline", action="store_true", help="强制使用合成数据")
    parser.add_argument("--as-of", type=str, default=None, help="指定基准日期 YYYY-MM-DD")
    args = parser.parse_args()
    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    run(as_of=as_of, offline=args.offline)


if __name__ == "__main__":
    main()
