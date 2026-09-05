"""Streamlit 交互式报告页面。

本地运行::
    streamlit run src/oilcast/app.py

部署到 Streamlit Community Cloud 后，每日 GitHub Actions 更新仓库内
reports/latest/latest.json，页面随之刷新；侧边栏可回溯任意历史报告。
数据原则：strict 模式只展示真实可追溯数据，缺失即明示，不做任何补齐。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1])     # .../src
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 项目根
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pandas as pd
import streamlit as st

from oilcast.config import get_config
from oilcast.reporting.narratives import FACTOR_CN, STATUS_CN
from oilcast.reporting.static_html import (HORIZON_CN, price_figure,
                                           scenario_figure, weights_figure)

st.set_page_config(page_title="多因素油价智能分析与预测系统", layout="wide", page_icon="🛢️")
CFG = get_config()


# --------------------------------------------------------------- 数据加载
@st.cache_data(ttl=300, show_spinner=False)
def list_archive_dates() -> list[str]:
    arch = Path(CFG["storage"]["archive_dir"])
    return sorted([p.stem for p in arch.glob("*.json")], reverse=True) if arch.exists() else []


@st.cache_data(ttl=300, show_spinner=False)
def load_report(date: str | None) -> dict | None:
    if date and date != "latest":
        path = Path(CFG["storage"]["archive_dir"]) / f"{date}.json"
    else:
        path = Path(CFG["storage"]["latest_dir"]) / "latest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_pipeline(demo: bool) -> None:
    cmd = [sys.executable, "-m", "oilcast.pipeline.main"] + (["--demo"] if demo else [])
    with st.spinner("正在采集真实数据、训练模型并生成报告，约需 1~3 分钟…"):
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT / "src"),
                              capture_output=True, text=True)
    if proc.returncode not in (0,):
        st.error(f"流水线失败（退出码 {proc.returncode}）：{proc.stderr[-1500:]}")
    else:
        st.success("报告已更新")
        st.cache_data.clear()


def _is_ok(item) -> bool:
    return isinstance(item, dict) and item.get("status", "ok") == "ok" and "endpoint" in item


# --------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.header("⚙️ 报告控制台")
    dates = list_archive_dates()
    chosen = st.selectbox("历史报告存档", ["latest"] + dates,
                          format_func=lambda x: "最新报告" if x == "latest" else x)
    target = st.radio("价格标的", ["wti", "brent", "diesel"],
                      format_func=lambda x: {"wti": "WTI原油", "brent": "布伦特原油",
                                             "diesel": "国内0#柴油"}[x])
    st.divider()
    demo = st.checkbox("演示模式（全部合成数据，仅用于测试）", value=False,
                       help="strict 模式只用真实数据；演示模式生成的数据严禁用于真实判断")
    if st.button("🔄 立即重新生成报告", width="stretch"):
        run_pipeline(demo)
        st.rerun()
    st.caption("每日 UTC 02:00（北京 10:00）由 GitHub Actions 自动运行")

report = load_report(chosen)
st.title("🛢️ 多因素油价智能分析与预测系统")
if report is None:
    st.warning("尚未找到报告。请在左侧点击「立即重新生成报告」，"
               "或在终端执行 `python -m oilcast.pipeline.main`（strict 真实模式）。")
    st.stop()

if report.get("mode") == "demo":
    st.error("⚠ 当前为演示模式：全部数据为人工合成、非真实观测，严禁用于任何真实判断 ⚠")

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
c1.markdown(f"**报告日期：{report['report_date']}** ｜ 生成于 {report['generated_at']} ｜ "
            f"模式：{'演示合成' if report.get('mode') == 'demo' else '严格真实'}")
for col, t in ((c2, "wti"), (c3, "brent"), (c4, "diesel")):
    meta = report["current_meta"][t]
    if meta["status"] == "ok":
        col.metric(report["names"][t],
                   f"{meta['value']:.2f} {report['units'][t]}",
                   f"观测 {meta['observed_date']}")
    else:
        col.metric(report["names"][t], "不可用")

with st.expander("📋 数据谱系与质量门（字段状态 / 真实来源 / 末次观测日期 / 样本数）"):
    lin = pd.DataFrame(report.get("lineage", {}).values())
    if not lin.empty:
        lin["状态"] = lin["status"].map(STATUS_CN).fillna(lin["status"])
        if "tried_sources" not in lin.columns:
            lin["tried_sources"] = ""
        st.dataframe(lin[["field", "display", "状态", "source_name",
                          "last_observed", "n_obs", "tried_sources"]].rename(
            columns={"field": "字段", "display": "含义", "source_name": "命中来源",
                     "last_observed": "末次观测", "n_obs": "样本数",
                     "tried_sources": "数据源优先级尝试链"}),
            width="stretch", hide_index=True)

# --------------------------------------------------------------- 预测卡片
st.subheader("多周期预测卡片（WTI 原油）")
cols = st.columns(3)
for col, hz in zip(cols, ("short", "mid", "long")):
    item = report["forecasts"][hz]["wti"]
    with col:
        st.markdown(f"#### {HORIZON_CN[hz]}")
        if not _is_ok(item):
            st.warning(f"暂不预测：{item.get('reason', '真实数据不可用')}")
            continue
        ep = item["endpoint"]
        st.caption(f"截至 {ep['target_date']}（观测日 {item.get('observed_date','—')}）")
        st.metric("预测均值（美元/桶）", f"{ep['mean']:.2f}", delta=f"{ep['pct_mean']:+.2f}%")
        st.markdown(f"95%区间 **{ep['q05']} ~ {ep['q95']}**　"
                    f"看涨 {ep['prob_up']*100:.0f}% / 看跌 {ep['prob_down']*100:.0f}%")
        st.progress(float(ep["prob_up"]), text=f"看涨概率 {ep['prob_up']*100:.0f}%")

# --------------------------------------------------------------- 走势图
st.subheader("价格走势、预测曲线与概率区间（缺口=当时无真实观测）")
name_map = {"wti": ("wti", "WTI原油", "美元/桶"),
            "brent": ("brent", "布伦特原油", "美元/桶"),
            "diesel": ("diesel", "国内0#柴油批发价", "元/吨")}
key, cn, unit = name_map[target]
st.plotly_chart(price_figure(report, key, cn, unit), width="stretch")

tab_w, tab_s = st.tabs(["因素权重解释", "长期情景与机构锚"])
with tab_w:
    fig_w = weights_figure(report)
    if fig_w is None:
        st.warning("可用真实因素不足，本期不计算因素权重。")
    else:
        st.plotly_chart(fig_w, width="stretch")
    st.info(report["narratives"]["weights"])
with tab_s:
    item = report["forecasts"]["long"]["wti"]
    if _is_ok(item):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(scenario_figure(report), width="stretch")
        with cc2:
            ep = item["endpoint"]
            st.markdown("**长期关键节点（WTI，美元/桶）**")
            if "checkpoints" in ep:
                df_cp = pd.DataFrame(ep["checkpoints"]).T
                df_cp.columns = ["均值", "5%分位", "95%分位"]
                st.dataframe(df_cp, width="stretch")
            if ep.get("institution_anchor"):
                st.metric("机构目标价中位数（真实抽取）", f"{ep['institution_anchor']:.1f} 美元/桶")
        st.info(report["narratives"]["long_wti"])
    else:
        st.warning(f"长期预测暂不可用：{item.get('reason','真实数据不可用')}")

# --------------------------------------------------------------- 事件列表
st.subheader("关键事件与量化影响（真实新闻）")
ev = pd.DataFrame(report["events"])
if not ev.empty:
    filt = st.radio("时间范围", ["一周", "一月"], horizontal=True)
    ev["date"] = pd.to_datetime(ev["date"])
    cutoff = pd.Timestamp(report["report_date"]) - pd.Timedelta(days=7 if filt == "一周" else 30)
    show = ev[ev["date"] >= cutoff].copy()
    show["因素主题"] = show["theme"].map(FACTOR_CN).fillna(show["theme"])
    show["强度"] = (show["intensity"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(show[["date", "title", "source", "因素主题", "强度", "est_price_impact"]].rename(
        columns={"date": "日期", "title": "事件", "source": "来源",
                 "est_price_impact": "估算影响(美元/桶)"}),
        width="stretch", hide_index=True)
    for t in report["narratives"]["events"][:5]:
        st.markdown(f"- {t}")
else:
    st.caption("事件源不可达或无有效真实条目，按数据原则不展示任何模拟事件。")

# --------------------------------------------------------------- 机构观点
st.subheader("机构观点与目标价（真实抽取）")
vw = pd.DataFrame(report["views"])
if not vw.empty:
    st.dataframe(vw.rename(columns={"date": "日期", "institution": "机构",
                                    "target_wti": "WTI目标价", "stance": "方向",
                                    "note": "摘要"}), width="stretch", hide_index=True)
else:
    st.caption("机构观点源当前不可达，保持空缺。")

# ----------------------------------------------- 模型学习 / 回测 / 预测复盘
st.subheader("模型学习、回测与预测复盘（每日用最新真实数据重训并复测）")
ml = report.get("model_learning") or {}
if ml:
    if report["narratives"].get("learning"):
        st.info(report["narratives"]["learning"])
    st.caption(f"本期重训时刻：{ml.get('retrained_at','—')}｜滚动窗口：最近 "
               f"{ml.get('rolling_window','—')} 个交易日｜累计第 {ml.get('n_runs','—')} 期")
    tm = ml.get("train_meta", {})
    if tm:
        rows = [{"标的": report["names"].get(k, k), "训练区间": f"{v['train_start']} ~ {v['train_end']}",
                 "有效样本行": v["n_valid"], "逐日模型数": v["n_steps"]} for k, v in tm.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**滚动样本外回测（短期）**")
        bt = ml.get("backtest", {})
        if bt.get("available"):
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("MAE", f"{bt['mae_pct']}%")
            b2.metric("RMSE", f"{bt['rmse_pct']}%")
            b3.metric("方向命中率", f"{bt['direction_accuracy']*100:.0f}%")
            b4.metric("随机游走MAE", f"{bt['benchmark_mae_pct']}%")
            st.caption(("已跑赢随机游走基准" if bt["mae_pct"] < bt["benchmark_mae_pct"]
                        else "暂未跑赢随机游走，继续积累样本"))
        else:
            st.caption(bt.get("reason", "真实样本积累中，回测暂不可用"))
    with c2:
        st.markdown("**历史预测 vs 已实现真实价（复测）**")
        rv = ml.get("review", {})
        if rv.get("available"):
            sm = rv["summary"]
            r1, r2, r3 = st.columns(3)
            r1.metric("已到期预测", f"{sm['n']} 条")
            r2.metric("平均误差", f"{sm['mae_pct']}%")
            r3.metric("方向命中", f"{sm['dir_acc']}%" if sm["dir_acc"] is not None else "—")
            st.caption(f"95%区间覆盖真实价比例：{sm['coverage95']}%"
                       if sm["coverage95"] is not None else "区间覆盖统计积累中")
        else:
            st.caption(rv.get("reason", "历史预测目标日尚未到期，下一期起复测"))
    wd = pd.DataFrame(ml.get("weight_delta", []))
    if not wd.empty:
        st.markdown(f"**权重自适应更新（对比 {ml.get('prev_weight_date') or '上一期'}）**")
        st.dataframe(wd.rename(columns={"factor": "因素", "prev": "上期权重%",
                                        "now": "本期权重%", "delta_pp": "变化(pp)"}),
                     width="stretch", hide_index=True)
    det = pd.DataFrame((ml.get("review") or {}).get("detail", []))
    if not det.empty:
        st.markdown("**最近预测复盘明细（预测 vs 后续真实收盘）**")
        det = det.copy()
        det["instrument"] = det["instrument"].map(report["names"]).fillna(det["instrument"])
        st.dataframe(det.rename(columns={"report_date": "发布日", "target_date": "目标日",
                                         "horizon": "周期", "instrument": "标的", "base": "发布时价",
                                         "pred": "预测", "actual": "实际", "pred_ret_pct": "预测涨跌%",
                                         "actual_ret_pct": "实际涨跌%", "abs_err_pct": "误差%",
                                         "dir_hit": "方向命中", "covered": "落95%区间"}),
                     width="stretch", hide_index=True)
else:
    st.caption("旧版本报告缺少学习记录，重新运行后生成。")

st.subheader("文字解读")
for t, cn2 in [("wti", "WTI原油"), ("brent", "布伦特原油"), ("diesel", "国内柴油")]:
    with st.expander(f"{cn2}：历史与两周/三个月预测解读", expanded=(t == "wti")):
        st.write("**历史走势**：" + report["narratives"]["trends"][t])
        st.write("**两周预测**：" + report["narratives"]["short"][t])
        st.write("**三个月预测**：" + report["narratives"]["mid"][t])

st.caption(report["narratives"]["sources"] +
           " ｜ 缺失、过期或无法验证的数据一律不补齐；预测不构成投资建议。")
