"""每日主流程：采集 → 存储 → 特征 → 权重/模型 → 预测 → 回测 → 报告存档。

命令行::
    python -m oilcast.pipeline.main              # strict：只用真实可追溯数据
    python -m oilcast.pipeline.main --demo       # 显式演示：全部合成数据
    python -m oilcast.pipeline.main --require-prices   # 价格不可用时以非零码退出（CI告警）

任何标的/因素真实数据不足时，对应模块输出 status=unavailable 与原因，
绝不用残缺或合成数据硬算预测。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import get_config, ensure_dirs
from ..data_sources.collector import collect
from ..features.engineering import (FACTOR_GROUPS, build_features,
                                    factor_availability, make_supervised)
from ..models.errors import InsufficientData
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
    return out.round(2).where(pd.notnull(out), None).to_dict("records")


def _hist_records(prices: pd.DataFrame, n: int = 180) -> list:
    p = prices.tail(n).copy()
    p.insert(0, "date", p.index.strftime("%Y-%m-%d"))
    return p.where(pd.notnull(p), None).round(2).to_dict("records")


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


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def _available_feature_cols(factor_avail: Dict[str, bool]) -> set:
    cols = set()
    for f, ok in factor_avail.items():
        if ok:
            cols.update(FACTOR_GROUPS.get(f, []))
    return cols


# --------------------------------------------------------------- 主流程
def run(as_of: Optional[datetime] = None, demo: bool = False,
        require_prices: bool = False) -> dict:
    cfg = get_config()
    ensure_dirs()
    as_of = pd.Timestamp(as_of or now_beijing())
    if as_of.tzinfo is not None:
        as_of = as_of.tz_convert("Asia/Shanghai").tz_localize(None)
    report_date = as_of.strftime("%Y-%m-%d")
    LOG.info("===== OilCast 每日报告开始：%s（mode=%s）=====", report_date,
             "demo" if demo else "strict")

    # 1) 数据采集与落库
    bundle = collect(as_of=as_of, mode="demo" if demo else "strict")
    db = OilCastDB()
    db.save_prices(bundle.prices, bundle.mode, bundle.lineage)
    db.save_macro(bundle.macro, bundle.mode, bundle.lineage)
    db.save_events(bundle.events)
    db.save_views(bundle.views)
    db.save_lineage(report_date, bundle.lineage, bundle.mode)
    save_csv_snapshot(bundle.prices, bundle.macro, report_date)
    prices, macro, events, views = bundle.prices, bundle.macro, bundle.events, bundle.views

    mcfg = cfg["model"]
    h_short, h_mid, h_long = (int(mcfg["short_horizon_td"]),
                              int(mcfg["mid_horizon_td"]),
                              int(mcfg["long_horizon_td"]))

    # 2) 标的可用性（只认真实质量门；demo 模式全部放开）
    target_ok = {t: (bundle.mode == "demo" or bundle.field_status(t) == "ok")
                 for t in TARGETS}
    usable_targets = [t for t, ok in target_ok.items() if ok]
    anchor = "wti" if target_ok["wti"] else ("brent" if target_ok["brent"] else None)
    if anchor is None:
        LOG.error("WTI/Brent 真实价格均不可用，本期不产出任何价格预测")
    last_real_date = {t: (prices[t].dropna().index.max() if prices[t].notna().any() else None)
                      for t in TARGETS}

    # 3) 特征矩阵（事件/观点源不可达时对应因素保持缺失）
    ev_ok = bundle.field_status("events") == "ok"
    vw_ok = bundle.field_status("institutional_view") == "ok"
    feats = {t: build_features(prices, macro, events, views, target=t,
                               events_available=ev_ok, views_available=vw_ok)
             for t in usable_targets}
    if feats:
        Path(cfg["storage"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
        feats[anchor or usable_targets[0]].to_csv(
            Path(cfg["storage"]["processed_dir"]) / f"features_{report_date}.csv",
            encoding="utf-8-sig")

    # 4) 因素权重（以主定价品种学习；因素可用性来自真实覆盖率）
    factor_avail = factor_availability(feats[anchor]) if anchor else \
        {f: False for f in cfg["prior_weights"]}
    weights = pd.DataFrame(columns=["factor", "weight", "model_importance",
                                    "prior", "available"])
    if anchor:
        X_w, y_w = make_supervised(feats[anchor], prices[anchor], h_short)
        prev = None
        hist_dates = db.list_report_dates()
        if hist_dates:
            conn = sqlite3.connect(cfg["storage"]["sqlite_path"])
            prev_tbl = pd.read_sql_query(
                "SELECT factor, weight FROM factor_weights WHERE report_date=?",
                conn, params=(hist_dates[0],))
            conn.close()
            prev = prev_tbl.set_index("factor")["weight"] if not prev_tbl.empty else None
        weights = learn_factor_weights(X_w, y_w, available=factor_avail, prev_weights=prev)
        db.save_weights(report_date, weights)

    avail_cols = _available_feature_cols(factor_avail)

    # 5) 短期预测：逐可用标的直接建模（不做跨标的推算）
    short_out: Dict[str, dict] = {}
    for t in TARGETS:
        if not target_ok[t]:
            lin = bundle.lineage.get(t, {})
            short_out[t] = _unavailable(
                f"{TARGETS[t][0]}真实数据不可用（{lin.get('status','unavailable')}）："
                f"{lin.get('note') or lin.get('source_name','源不可达')}")
            continue
        try:
            fc = ShortTermForecaster(horizon=h_short, window=int(mcfg["rolling_window"]),
                                     compute_residuals=True)
            fc.fit(feats[t], prices[t])
            last_dt = last_real_date[t]
            res = fc.predict(feats[t].loc[[last_dt]], float(prices[t].loc[last_dt]),
                             future_index(last_dt, h_short))
            short_out[t] = {"status": "ok", "endpoint": res.endpoint,
                            "path": _path_records(res.path),
                            "arima_benchmark": res.benchmark_endpoint,
                            "observed_date": last_dt.strftime("%Y-%m-%d")}
        except InsufficientData as exc:
            short_out[t] = _unavailable(str(exc))
    LOG.info("短期预测完成：%s", {t: short_out[t]["status"] for t in short_out})

    # 6) 中期 VAR：逐可用标的
    mid_out: Dict[str, dict] = {}
    for t in TARGETS:
        if not target_ok[t]:
            mid_out[t] = short_out[t]
            continue
        try:
            var = MidTermVAR().fit(prices, macro, target=t)
            probs = scenario_probabilities(feats[t].loc[last_real_date[t]],
                                           available=avail_cols)
            r = var.predict(future_index(last_real_date[t], h_mid), probs, cfg["scenarios"])
            mid_out[t] = {"status": "ok", "endpoint": r["endpoint"],
                          "path": _path_records(r["path"]), "scenario_probs": probs,
                          "observed_date": last_real_date[t].strftime("%Y-%m-%d")}
        except InsufficientData as exc:
            mid_out[t] = _unavailable(str(exc))

    # 7) 长期情景 + 蒙特卡洛：逐可用标的
    long_out: Dict[str, dict] = {}
    anchor_inst = None
    if vw_ok and "target_wti" in views.columns:
        vals = pd.to_numeric(views["target_wti"], errors="coerce").dropna()
        anchor_inst = round(float(vals.median()), 1) if len(vals) else None
    for t in TARGETS:
        if not target_ok[t]:
            long_out[t] = short_out[t]
            continue
        try:
            probs = scenario_probabilities(feats[t].loc[last_real_date[t]],
                                           available=avail_cols)
            lt = LongTermMonteCarlo().fit(float(prices[t].loc[last_real_date[t]]))
            r = lt.predict(future_index(last_real_date[t], h_long), probs,
                           institution_anchor=anchor_inst if t == "wti" else None)
            long_out[t] = {"status": "ok", "endpoint": r["endpoint"],
                           "path": _path_records(r["path"]),
                           "observed_date": last_real_date[t].strftime("%Y-%m-%d")}
        except InsufficientData as exc:
            long_out[t] = _unavailable(str(exc))
    LOG.info("中/长期预测完成")

    # 8) 回测（仅主锚）
    bt = backtest_short(feats[anchor], prices[anchor], horizon=h_short) if anchor else \
        {"status": "unavailable", "reason": "无可用真实价格序列，未执行回测"}

    # 9) 当前值（携带真实观测日期，缺失为 None）
    current, current_meta = {}, {}
    for t in TARGETS:
        lin = bundle.lineage.get(t, {})
        if target_ok[t] and last_real_date[t] is not None:
            current[t] = round(float(prices[t].loc[last_real_date[t]]), 2)
            current_meta[t] = {"value": current[t],
                               "observed_date": last_real_date[t].strftime("%Y-%m-%d"),
                               "status": "ok", "source": lin.get("source_name", "")}
        else:
            current[t] = None
            current_meta[t] = {"value": None, "observed_date": lin.get("last_observed"),
                               "status": lin.get("status", "unavailable"),
                               "source": lin.get("source_name", ""),
                               "reason": lin.get("note", "无真实可核验数据")}

    report = {
        "report_date": report_date,
        "generated_at": now_beijing().strftime("%Y-%m-%d %H:%M:%S %z"),
        "mode": bundle.mode,
        "lineage": bundle.lineage,
        "factor_availability": factor_avail,
        "current": current,
        "current_meta": current_meta,
        "units": {t: TARGETS[t][1] for t in TARGETS},
        "names": {t: TARGETS[t][0] for t in TARGETS},
        "history": _hist_records(prices),
        "forecasts": {"short": short_out, "mid": mid_out, "long": long_out},
        "weights": weights.to_dict("records") if len(weights) else [],
        "events": _events_records(events),
        "views": _views_records(views),
        "backtest": bt,
        "narratives": _build_narratives(bundle, prices, current_meta, short_out,
                                        mid_out, long_out, weights, events, bt),
    }

    # 10) 落库 + JSON/HTML 存档
    _persist_forecasts(db, report_date, report)
    archive = Path(cfg["storage"]["archive_dir"])
    latest = Path(cfg["storage"]["latest_dir"])
    archive.mkdir(parents=True, exist_ok=True)
    latest.mkdir(parents=True, exist_ok=True)
    (archive / f"{report_date}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (latest / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, default=str), encoding="utf-8")
    html = render_static_html(report)
    (archive / f"{report_date}.html").write_text(html, encoding="utf-8")
    (latest / "index.html").write_text(html, encoding="utf-8")
    # Pages 根入口：自动跳到 latest（upload 整个 reports 目录时生效）
    (latest.parent / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=latest/index.html">'
        '<a href="latest/index.html">最新报告</a>', encoding="utf-8")
    db.register_report(report_date, str(archive / f"{report_date}.json"),
                       str(archive / f"{report_date}.html"))
    LOG.info("报告已存档：%s", archive / f"{report_date}.html")

    if require_prices and anchor is None:
        raise SystemExit(2)
    return report


def _is_ok(item: dict) -> bool:
    return isinstance(item, dict) and item.get("status", "ok") == "ok"


def _build_narratives(bundle, prices, current_meta, short_out, mid_out, long_out,
                     weights, events, bt):
    def one_narr(item, cn, unit, horizon):
        if not _is_ok(item):
            return (f"{cn}{horizon}预测暂不提供：{item.get('reason','真实数据不可用')}。"
                    f"按数据原则，不以估算或合成数据替代。")
        return narr.forecast_narrative(item["endpoint"], cn, unit, horizon)

    def trend_narr(t):
        meta = current_meta[t]
        if meta["status"] != "ok":
            return (f"{TARGETS[t][0]}本期无真实可核验价格（状态：{meta['status']}；"
                    f"{meta.get('reason','')}），不做走势判断。")
        return narr.trend_narrative(prices[t].dropna(), TARGETS[t][0], TARGETS[t][1])

    return {
        "trends": {t: trend_narr(t) for t in TARGETS},
        "short": {t: one_narr(short_out[t], TARGETS[t][0], TARGETS[t][1], "两周")
                  for t in TARGETS},
        "mid": {t: one_narr(mid_out[t], TARGETS[t][0], TARGETS[t][1], "三个月")
                for t in TARGETS},
        "long_wti": narr.scenario_narrative(long_out["wti"]["endpoint"])
        if _is_ok(long_out["wti"]) else
        f"WTI长期预测暂不提供：{long_out['wti'].get('reason','真实数据不可用')}",
        "weights": narr.weights_narrative(weights) if len(weights) else
        "可用真实因素不足，本期不计算因素权重。",
        "events": narr.events_narrative(events),
        "backtest": narr.backtest_narrative(bt) if isinstance(bt, dict) and
                    bt.get("status") != "unavailable" else
                    "无可用真实价格序列，本期未执行滚动回测。",
        "sources": narr.lineage_narrative(bundle),
    }


def _persist_forecasts(db: OilCastDB, report_date: str, report: dict) -> None:
    records = []
    for horizon, pack in report["forecasts"].items():
        for inst, item in pack.items():
            if item.get("status") == "unavailable":
                continue
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
    parser.add_argument("--demo", action="store_true",
                        help="演示模式：全部使用合成数据（报告强制水印）")
    parser.add_argument("--offline", action="store_true",
                        help="--demo 的旧别名")
    parser.add_argument("--require-prices", action="store_true",
                        help="WTI/Brent 真实价格均不可用时以退出码 2 失败（CI 告警）")
    parser.add_argument("--as-of", type=str, default=None, help="基准日期 YYYY-MM-DD")
    args = parser.parse_args()
    run(as_of=pd.Timestamp(args.as_of) if args.as_of else None,
        demo=args.demo or args.offline, require_prices=args.require_prices)


if __name__ == "__main__":
    main()
