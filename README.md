# OilCast · 多因素油价智能分析与预测系统

每天 **北京时间 10:00（UTC 02:00）** 自动完成「真实数据采集 → 质量门 → 特征工程 →
权重学习/模型训练 → 短中长期预测 → 交互式报告发布」，覆盖 **WTI、布伦特原油与国内 0# 柴油**，
输出带概率区间、事件量化影响、因素权重与文字解读的分析报告：

- **静态 HTML 报告**（图表引擎本地内联共享、不依赖外网 CDN，自动发布到 GitHub Pages，即"公开网页"）；
- **Streamlit 交互页面**（可部署 Streamlit Community Cloud，支持历史回溯）；
- **SQLite + CSV 双存储**（原始数据、数据谱系、特征、预测、权重全部可审计回溯）。

---

## 0. 最高原则：只用真实、可追溯、带观测日期的数据

> **模型只建立在真实观测之上；任何缺失、过期或无法验证的数据都保持缺失，
> 绝不用插值、外推或合成值"补齐"成假数据。**

落地机制：

1. **双模式严格隔离**
   - `strict`（默认）：只抓取真实源；每个字段经**质量门**判定 `ok / stale（过期）/
     insufficient（样本不足）/ unavailable（不可达）`，状态写入报告顶部「数据谱系表」与
     SQLite `data_lineage` 表（来源、URL、抓取时刻、首末观测日、样本数、滞后工作日）。
   - `demo`（必须显式 `--demo`）：全部为合成数据，报告全程红色「合成数据」水印，
     且严禁与真实数据混合，仅用于功能演示。
2. **缺口不连线、不填零**：历史价格缺口在图上断开（`connectgaps=False`）；缺失因素不填 0
   冒充"中性"，而是从权重归一化中剔除并标注；样本不足的模型直接输出
   `unavailable + 原因`，不硬算。
3. **月频 vintage**：CPI/非农等月频指标在两次发布之间沿用「最近一次真实发布值」，
   底层逐点保留其原始发布日期（vintage），报告显示该值的观测日而非当天。
4. **当前值标注观测日**：每个现价都写明"截至 YYYY-MM-DD 真实观测"，不把滞后发布说成当日价。

---

## 1. 系统架构

```
                          ┌──────────────────────────────────────────┐
                          │        data_sources 数据层（strict）       │
 FRED(EIA转发) ─► WTI/Brent│ DCOILWTICO / DCOILBRENTEU 现货（主源）   │
 FRED ─────────► 宏观/汇率 │ DGS10/DGS2/DTWEXBGS/CPIAUCSL/PAYEMS/…   │
 GPRD ─────────► 地缘风险  │ Caldara-Iacoviello 日频指数              │
 GoogleNews RSS ► 事件/机构│ 不可达即留空，绝不伪造                    │
 yfinance/EIA ──► 备份/可选│ PoliteSession：UA、延时、重试、守 robots │
                          └───────────────┬──────────────────────────┘
                                          ▼
        quality 质量门 + lineage 谱系（状态/来源/观测日期/样本数）
                                          ▼
   storage（SQLite: 原始/谱系/事件/观点/预测/权重/报告 + CSV 快照）
                                          ▼
   features：九大类因素 → "利多为正"、无未来泄漏、缺失保持 NaN 不填零
                                          ▼
   models ┬─ weights     RF + LASSO 融合（缺失因素剔除）→ 先验收缩 → 跨日 EMA
          ├─ short_term  Direct 多步梯度提升（10 交易日）+ 样本外残差区间 + ARIMA 基准
          ├─ mid_term    动态内生变量 VAR（66 交易日）+ 残差块 bootstrap
          ├─ long_term   高/中/低情景 + 2000 条蒙特卡洛路径（252 交易日）
          └─ evaluation  滚动原点回测（MAE/RMSE/方向命中率 vs 随机游走）
           （任一模型真实样本不足 → 显式 unavailable，不输出假预测）
                                          ▼
   reporting ┬─ static_html  自包含 HTML（谱系表/缺口断开/不可用明示）→ Pages
             ├─ narratives   每个数字可追溯的中文解读；不可用即说明原因
             └─ app.py       Streamlit 交互报告（历史存档/手动重跑）
                                          ▲
              GitHub Actions cron 02:00 UTC 每日触发并 commit 回仓库
```

## 2. 目录结构

```
oilcast/
├── config/config.yaml            # 全部可调参数：数据原则/质量门/窗口/先验/情景
├── src/oilcast/
│   ├── config.py / utils.py
│   ├── data_sources/
│   │   ├── fred_client.py        # FRED 公开 CSV 客户端（油价/宏观主源）
│   │   ├── prices.py            # WTI/Brent 真实现货（FRED 主、yfinance 备、EIA 可选）
│   │   ├── macro.py             # 真实宏观 + 月频 vintage 对齐
│   │   ├── quality.py           # 数据谱系 FieldLineage + 质量门
│   │   ├── diesel.py            # 国内柴油真实源钩子（未接入前 unavailable，不估算）
│   │   ├── events.py / institutional.py
│   │   ├── synthetic.py         # 仅 --demo 使用
│   │   └── collector.py         # strict/demo 双模式调度
│   ├── storage/database.py      # SQLite（含 data_lineage 谱系表）+ CSV 快照
│   ├── features/engineering.py  # 缺失不填零、因素可用性判定
│   ├── models/                  # weights/short/mid/long/evaluation + errors
│   ├── reporting/               # 静态 HTML、文字解读、Streamlit app
│   └── pipeline/main.py
├── .github/workflows/daily_report.yml
├── scripts/（backfill.py、run_once.sh）  ├── tests/test_smoke.py
├── data/、reports/、requirements.txt、pyproject.toml、README.md
```

## 3. 本地快速开始

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt && pip install -e .

# 1) strict 模式（默认）：只用真实数据，缺失/过期会在报告中明示
python -m oilcast.pipeline.main
#   --demo              显式演示：全部合成（报告强制水印，禁止用于真实判断）
#   --require-prices    WTI/Brent 真实价格均不可用时以退出码 2 失败（供 CI 告警）
#   --as-of 2026-09-02  指定基准日期

# 2) 交互式报告
streamlit run src/oilcast/app.py
```

产物：`reports/latest/index.html`（浏览器直接打开）、`latest.json`、
`reports/archive/YYYY-MM-DD.{json,html}`、`data/oilcast.db`。
一键脚本：`bash scripts/run_once.sh`。

## 4. 真实数据源（均可在线核验口径与观测日期）

| 模块 | 真实源（FRED 序列页即原始出处说明） | 需要 Key | 不可达时 |
|---|---|---|---|
| WTI 现货 | FRED `DCOILWTICO`（源头 EIA）；备份 yfinance `CL=F`；可选 EIA v2 | 否（EIA 需 `EIA_API_KEY`） | 该标的 unavailable，不造数 |
| Brent 现货 | FRED `DCOILBRENTEU`；备份 `BZ=F` | 否 | 同上 |
| 美债/美元 | FRED `DGS10` `DGS2` `DTWEXBGS`（广义美元指数） | 否 | 该因素剔除、不参与权重 |
| CPI/非农/联邦基金/需求 | FRED `CPIAUCSL` `PAYEMS` `FEDFUNDS` `INDPRO`（月频 vintage） | 否 | 同上 |
| 地缘风险 | GPRD（Caldara & Iacoviello）日频 | 否 | 用真实事件代理；事件也无则剔除 |
| 事件 / 机构观点 | Google News RSS + 规则打分 / 正则抽取 | 否 | 空表并标注，不生成模拟条目 |
| 国内 0# 柴油 | **预留钩子** `diesel.fetch_diesel_live`（金投网/生意社/发改委） | 视站点 | **unavailable，不按国际油价推算冒充** |

> 注意：FRED `GASDESM` 是**美国**柴油零售价（美元/加仑），口径不同，
> 系统只作参考序列，绝不用它冒充中国柴油。

爬虫礼仪：统一 `PoliteSession`（自定义 UA、请求间隔、有限重试、分级超时预算）；
新增数据源前先检查 `/robots.txt` 与服务条款。接入国内柴油真实源时，在
`fetch_diesel_live` 中解析并返回 `pd.Series(index=发布日期, values=元/吨)` 与来源 meta，
质量门、入库、建模、报告链路自动生效。

## 5. 模型方法论

- **因素体系（9 类）**：供给/OPEC+、地缘风险、美元、美债10Y、CPI 意外、非农、
  美联储政策预期、需求前景、机构观点；统一"数值越大越利多油价"。
- **权重学习**：RandomForest 重要性 ×0.6 + |LASSO 系数| ×0.4 聚合到因素，
  **只在真实数据可用的因素间归一化**；向人工先验收缩、与上一轮权重 EMA 平滑。
  真实样本不足时退回先验并标注"模型未训练"。
- **短期（10 交易日）**：Direct multi-step 梯度提升（每步长一个模型，避免误差累积），
  区间来自 rolling-origin 样本外残差分位数，ARIMA 自动降阶作基准；点预测向随机游走收缩。
- **中期（66 交易日）**：油价 + 真实覆盖充分的宏观变量动态组建 VAR，AIC 选阶并校验
  特征根在单位圆外（稳定性），残差块 bootstrap 500 条路径叠加情景漂移。
- **长期（252 交易日）**：三情景 softmax 概率（缺失因素不投票、权重重新归一化），
  2000 条几何布朗路径；真实机构目标价中位数作外部锚并列展示。
- **自适应**：180 交易日滚动窗口，每日新真实价格入库后重训，权重历史在
  `factor_weights` 表可追溯。
- **回测**：最近若干滚动原点样本外检验 MAE/RMSE/方向命中率并对比随机游走；
  价格缺口处不计误差。

## 6. 自动化部署（GitHub Actions + Pages / Streamlit Cloud）

1. GitHub 新建 **public** 仓库并推送；
2. **Settings → Pages → Build and deployment → Source 选 "GitHub Actions"**；
3. （可选）Settings → Secrets and variables → Actions 添加 `EIA_API_KEY`；
4. 每天 **UTC 02:00（北京 10:00）** 自动运行（也可 Actions 页手动触发；
   `demo_mode` 输入仅用于演示），结果 commit 回仓库并把整个 `reports/` 部署为公开网页：
   站点根路径自动跳转到 `latest/index.html`，历史报告在 `archive/`，
   图表引擎 `assets/plotly.min.js` 随仓库发布，**不引用任何外网 CDN**；
5. Streamlit Cloud 连同一仓库、入口填 `src/oilcast/app.py`，每日 push 后自动刷新。

> 测试/容器隔离：设置环境变量 `OILCAST_HOME=/path` 可把数据库与报告重定向到该目录
> （pytest 已用临时目录自动隔离，demo 合成数据绝不会写入真实 `data/`）。

## 7. 常用配置（config/config.yaml）

| 参数 | 含义 | 默认 |
|---|---|---|
| `data_sources.mode` | 数据模式：strict / demo | strict |
| `quality_gate.price_daily` | 价格最少观测/最大滞后工作日 | 120 / 7 |
| `quality_gate.macro_monthly.max_stale_days` | 月频发布滞后容忍 | 60 |
| `model.rolling_window` / `min_train_obs` | 滚动窗口 / 建模最少真实样本 | 180 / 250 |
| `model.short/mid/long_horizon_td` | 三周期步长（交易日） | 10 / 66 / 252 |
| `model.weight_ema_alpha` / `prior_shrinkage` | 跨日平滑 / 先验收缩 | 0.6 / 0.7 |

## 8. 测试

```bash
pytest -q   # demo 主链路 + strict 数据原则（质量门/vintage/缺失不填零/样本不足拒绝训练）
```

## 9. 局限与免责

- 免费公开源可能限流或改版：strict 模式下表现为对应字段 `unavailable/stale` 并在报告明示，
  而不是用假数据保持"页面完整"；配置 `--require-prices` 可让 CI 在价格全失时显式失败告警。
- 国内柴油在接入可核验真实源之前保持 unavailable，这是有意的诚实设计。
- 规则词典情感打分为可解释基线，可替换为 FinBERT（保持输出表结构即可）。
- 预测区间反映历史波动与模型不确定性，**不构成投资建议**；第三方数据版权归原方所有。
