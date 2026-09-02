"""Streamlit 交互式报告页面。

本地运行::

    streamlit run src/oilcast/app.py
部署到 Streamlit Community Cloud 后，每日由 GitHub Actions 更新仓库内
reports/latest/latest.json，页面随之展示最新报告；也可在侧边栏回溯任意历史报告。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# 兼容 `streamlit run src/oilcast/app.py` 直接运行（未 pip install -e . 时也能找到包）
_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oilcast.config import get_config
from oilcast.reporting.narratives import FACTOR_CN
from oilcast.reporting.static_html import (H_COLOR, HORIZON_CN, price_figure,
                                           scenario_figure, weights_figure)

st.set_page_config(page_title="多因素油价智能分析与预测系统", layout="wide", page_icon="🛢️")
CFG = get_config()


# --------------------------------------------------------------- 数据加载
@st.cache_data(ttl=300, show_spinner=False)
def list_archive_dates() -> list[str]:
    arch = Path(CFG["storage"]["archive_dir"])
    if not arch.exists():
        return []
    return sorted([p.stem for p in arch.glob("*.json")], reverse=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_report(date: str | None) -> dict | None:
    if date and date != "latest":
        path = Path(CFG["storage"]["archive_dir"]) / f"{date}.json"
    else:
        path = Path(CFG["storage"]["latest_dir"]) / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_pipeline(offline: bool) -> None:
    """手动触发一次完整流水线（Streamlit Cloud 小机型也可运行）。"""
    cmd = [sys.executable, "-m", "oilcast.pipeline.main"] + (["--offline"] if offline else [])
    with st.spinner("正在采集数据、训练模型并生成报告，约需 1~3 分钟…"):
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT / "src"), capture_output=True, text=True)
    if proc.returncode != 0:
        st.error(f"流水线失败：{proc.stderr[-1500:]}")
    else:
        st.success("报告已更新")
        st.cache_data.clear()


# --------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.header("⚙️ 报告控制台")
    dates = list_archive_dates()
    options = ["latest"] + dates
    chosen = st.selectbox("历史报告存档", options,
                          format_func=lambda x: "最新报告" if x == "latest" else x)
    target = st.radio("价格标的", ["wti", "brent", "diesel"],
                      format_func=lambda x: {"wti": "WTI原油", "brent": "布伦特原油",
                                             "diesel": "国内0#柴油"}[x])
    st.divider()
    offline = st.checkbox("离线模式（合成数据）", value=False,
                          help="真实源不可达时用合成数据演示完整功能")
    if st.button("🔄 立即重新生成报告", width="stretch"):
        run_pipeline(offline)
        st.rerun()
    st.caption("每日 UTC 02:00（北京 10:00）由 GitHub Actions 自动运行")

report = load_report(chosen)

st.title("🛢️ 多因素油价智能分析与预测系统")
if report is None:
    st.warning("尚未找到报告。请在左侧点击「立即重新生成报告」，"
               "或在终端执行 `python -m oilcast.pipeline.main --offline`。")
    st.stop()

# 顶部元信息
c1, c2, c3 = st.columns([2, 1, 1])
c1.markdown(f"**报告日期：{report['report_date']}** ｜ 生成于 {report['generated_at']}")
with c2:
    st.metric("WTI 当前", f"{report['current']['wti']:.2f} 美元/桶")
with c3:
    st.metric("布伦特 当前", f"{report['current']['brent']:.2f} 美元/桶")
with st.expander("数据来源标注（live=真实抓取 / estimated=成本估算 / simulated=合成兜底）"):
    st.json(report["provenance"], expanded=False)

# --------------------------------------------------------------- 预测卡片
st.subheader("多周期预测卡片（WTI 原油）")
cols = st.columns(3)
for col, hz in zip(cols, ("short", "mid", "long")):
    ep = report["forecasts"][hz]["wti"]["endpoint"]
    with col:
        st.markdown(f"#### {HORIZON_CN[hz]} ｜ 截至 {ep['target_date']}")
        delta = f"{ep['pct_mean']:+.2f}%"
        st.metric("预测均值（美元/桶）", f"{ep['mean']:.2f}", delta=delta)
        st.markdown(
            f"95%区间 **{ep['q05']} ~ {ep['q95']}**　"
            f"看涨 {ep['prob_up']*100:.0f}% / 看跌 {ep['prob_down']*100:.0f}%")
        st.progress(float(ep["prob_up"]), text=f"看涨概率 {ep['prob_up']*100:.0f}%")

# --------------------------------------------------------------- 走势图
st.subheader("价格走势、预测曲线与概率区间")
name_map = {"wti": ("wti", "WTI原油", "美元/桶"),
            "brent": ("brent", "布伦特原油", "美元/桶"),
            "diesel": ("diesel", "国内0#柴油批发价", "元/吨")}
key, cn, unit = name_map[target]
st.plotly_chart(price_figure(report, key, cn, unit), width="stretch")

tab_w, tab_s = st.tabs(["因素权重解释", "长期情景与机构锚"])
with tab_w:
    st.plotly_chart(weights_figure(report), width="stretch")
    st.info(report["narratives"]["weights"])
with tab_s:
    cc1, cc2 = st.columns(2)
    with cc1:
        st.plotly_chart(scenario_figure(report), width="stretch")
    with cc2:
        ep = report["forecasts"]["long"]["wti"]["endpoint"]
        st.markdown("**长期关键节点（WTI，美元/桶）**")
        if "checkpoints" in ep:
            df_cp = pd.DataFrame(ep["checkpoints"]).T
            df_cp.columns = ["均值", "5%分位", "95%分位"]
            st.dataframe(df_cp, width="stretch")
        if ep.get("institution_anchor"):
            st.metric("机构目标价中位数", f"{ep['institution_anchor']:.1f} 美元/桶")
    st.info(report["narratives"]["long_wti"])

# --------------------------------------------------------------- 事件列表
st.subheader("关键事件与量化影响")
ev = pd.DataFrame(report["events"])
if not ev.empty:
    filt = st.radio("时间范围", ["一周", "一月"], horizontal=True)
    ev["date"] = pd.to_datetime(ev["date"])
    cutoff = pd.Timestamp(report["report_date"]) - pd.Timedelta(days=7 if filt == "一周" else 30)
    show = ev[ev["date"] >= cutoff].copy()
    show["因素主题"] = show["theme"].map(FACTOR_CN).fillna(show["theme"])
    show["强度"] = (show["intensity"] * 100).round(0).astype(int).astype(str) + "%"
    show["估算影响(美元/桶)"] = show["est_price_impact"]
    st.dataframe(show[["date", "title", "source", "因素主题", "强度",
                       "估算影响(美元/桶)"]].rename(
        columns={"date": "日期", "title": "事件", "source": "来源"}),
        width="stretch", hide_index=True)
    for t in report["narratives"]["events"][:5]:
        st.markdown(f"- {t}")
else:
    st.caption("近期未捕捉到达到阈值的事件")

# --------------------------------------------------------------- 机构观点
st.subheader("机构观点与目标价")
vw = pd.DataFrame(report["views"])
if not vw.empty:
    st.dataframe(vw.rename(columns={"date": "日期", "institution": "机构",
                                    "target_wti": "WTI目标价", "stance": "方向",
                                    "note": "摘要"}),
                 width="stretch", hide_index=True)

# --------------------------------------------------------------- 回测 + 解读
st.subheader("模型回测表现")
bt = report.get("backtest") or {}
if bt.get("available"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{bt['mae_pct']}%")
    m2.metric("RMSE", f"{bt['rmse_pct']}%")
    m3.metric("方向命中率", f"{bt['direction_accuracy']*100:.0f}%")
    m4.metric("随机游走基准MAE", f"{bt['benchmark_mae_pct']}%")
    if report["narratives"].get("backtest"):
        st.caption(report["narratives"]["backtest"])
else:
    st.caption("样本积累中，回测暂不可用")

st.subheader("文字解读")
for t, cn2 in [("wti", "WTI原油"), ("brent", "布伦特原油"), ("diesel", "国内柴油")]:
    with st.expander(f"{cn2}：历史与两周/三个月预测解读", expanded=(t == "wti")):
        st.write("**历史走势**：" + report["narratives"]["trends"][t])
        st.write("**两周预测**：" + report["narratives"]["short"][t])
        st.write("**三个月预测**：" + report["narratives"]["mid"][t])

st.caption(report["narratives"]["sources"] +
           " ｜ 预测区间反映历史波动与模型不确定性，不构成投资建议。")
