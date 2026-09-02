# OilCast · 多因素油价智能分析与预测系统

每天 **北京时间 10:00（UTC 02:00）** 自动完成「数据采集 → 特征工程 → 权重学习/模型训练 →
短中长期预测 → 交互式报告发布」，覆盖 **WTI、布伦特原油与国内 0# 柴油**，输出带概率区间、
事件量化影响、因素权重与文字解读的分析报告，并同时提供：

- **静态 HTML 报告**（自包含，自动发布到 GitHub Pages，即"公开网页"）；
- **Streamlit 交互页面**（可部署到 Streamlit Community Cloud，支持历史报告回溯）；
- **SQLite + CSV 双存储**（原始数据、特征、预测、权重全部可回溯，便于持续训练）。

---

## 1. 系统架构

```
                          ┌──────────────────────────────────────────┐
                          │            data_sources 数据层            │
 yfinance/EIA ─► WTI/Brent│ 真实源优先，逐表失败自动回退合成/估算数据 │
 FRED/GPRD ─────► 宏观/GPR│ 每次抓取带 UA、延时、重试，遵守 robots   │
 GoogleNews RSS ► 事件/机构│ provenance 字段如实标注 live/simulated  │
                          └───────────────┬──────────────────────────┘
                                          ▼
   storage（SQLite: 原始/事件/观点/预测/权重/报告索引 + CSV 快照）
                                          ▼
   features：九大类因素 → 统一"利多为正"方向、无未来泄漏的特征矩阵
                                          ▼
   models ┬─ weights     随机森林 + LASSO 融合 → 先验收缩 → 跨日 EMA 自适应
          ├─ short_term  Direct 多步梯度提升（10 交易日）+ walk-forward 残差区间 + ARIMA 基准
          ├─ mid_term    VAR 向量自回归（66 交易日）+ 残差块 bootstrap 区间
          ├─ long_term   高/中/低情景概率 + 2000 条蒙特卡洛路径（252 交易日）
          └─ evaluation  滚动原点回测（MAE/RMSE/方向命中率 vs 随机游走）
                                          ▼
   reporting ┬─ static_html.py  自包含 HTML（Plotly）→ GitHub Pages
             ├─ narratives.py   每个数字可追溯的中文文字解读
             └─ app.py          Streamlit 交互报告（历史存档/手动重跑）
                                          ▲
              GitHub Actions cron 02:00 UTC 每日触发并 commit 回仓库
```

## 2. 目录结构

```
oilcast/
├── config/config.yaml            # 全部可调参数：标的、窗口、horizon、先验权重、情景
├── src/oilcast/
│   ├── config.py / utils.py      # 配置加载、日志、礼貌型 HTTP 会话
│   ├── data_sources/             # prices/macro/diesel/events/institutional + synthetic 兜底 + collector 调度
│   ├── storage/database.py       # SQLite 建表/upsert/读取 + CSV 快照
│   ├── features/engineering.py   # 特征工程（细特征 ↔ 九大因素映射）
│   ├── models/                   # 权重学习、短/中/长期模型、回测
│   ├── reporting/                # 静态 HTML、文字解读、Streamlit app
│   ├── pipeline/main.py          # 每日主流程（命令行入口）
│   └── app.py                    # Streamlit 入口
├── .github/workflows/daily_report.yml
├── scripts/                      # backfill.py 历史回填、run_once.sh 一键运行
├── tests/test_smoke.py           # 端到端冒烟测试
├── data/                         # raw/ 原始CSV、processed/ 特征、oilcast.db
├── reports/                      # archive/ 历史 JSON+HTML，latest/ 最新报告
├── requirements.txt / pyproject.toml
└── README.md
```

## 3. 本地快速开始

```bash
# Python 3.10+
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 1) 跑一次完整流水线（优先真实数据，网络不通自动用合成数据兜底）
python -m oilcast.pipeline.main
#   --offline   强制使用合成数据（演示/CI）
#   --as-of 2026-09-02   指定基准日期

# 2) 打开交互式报告
streamlit run src/oilcast/app.py

# 产物：
#   reports/latest/index.html   静态报告（浏览器直接打开）
#   reports/latest/latest.json  结构化结果
#   reports/archive/YYYY-MM-DD.{json,html}  历史存档
#   data/oilcast.db             SQLite 数据库
```

一键脚本：`bash scripts/run_once.sh [--offline]`。

## 4. 数据源与接入方式

| 模块 | 默认真实源 | 是否需要 Key | 兜底策略 |
|---|---|---|---|
| WTI/Brent | yfinance（`CL=F`/`BZ=F`）；可选 EIA v2 API | 否（EIA 需 `EIA_API_KEY`） | 经济学自洽的合成行情 |
| 美债10Y/CPI/非农/联邦基金 | FRED 公开 CSV（DGS10/CPIAUCSL/PAYEMS…） | 否 | 合成宏观序列 |
| 美元指数 DXY | yfinance `DX-Y.NYB`，FRED 广义指数备选 | 否 | 合成序列 |
| 地缘风险 GPR | Caldara-Iacoviello GPRD 日频指数 | 否 | 事件强度聚合自建 |
| 地缘/宏观事件 | Google News RSS + 多空词典规则打分 | 否 | 合成事件库 |
| 机构观点 | Google News RSS + 正则抽取目标价 | 否 | 合成观点库 |
| 国内柴油 | **预留钩子** `diesel.fetch_diesel_live` | 视站点而定 | Brent 成本传导估算 → 合成 |

爬虫礼仪：统一 `PoliteSession`（自定义 UA、1.5s 间隔、有限重试）；新增数据源前请先检查
目标站点 `/robots.txt` 与服务条款。接入金投网/生意社等国内柴油源时，只需在
`fetch_diesel_live` 中实现解析并返回 `pd.Series(index=日期, values=元/吨)`，其余链路自动生效。

## 5. 模型方法论

- **因素体系（9 类）**：供给/OPEC+、地缘风险、美元指数、美债10Y、CPI 意外、非农、
  美联储政策预期、需求前景、机构观点。所有特征统一为"数值越大越利多油价"，系数符号可直接解释。
- **权重学习**：RandomForest 重要性（非线性）× 0.6 + |LASSO 系数|（线性稀疏）× 0.4，
  细特征聚合到 9 因素；再按 `prior_shrinkage` 向人工先验收缩，并与上一轮权重做 EMA，
  实现"每日新增真实数据后平滑自适应"。
- **短期（10 交易日）**：对每个步长 h 各训练一个梯度提升模型（Direct multi-step，避免误差累积），
  外生变量即多因素特征；概率区间来自 **rolling-origin 样本外残差分位数**；并拟合 ARIMA(2,0,2)
  作为无外生变量基准。
- **中期（66 交易日）**：油价/美元/美债/GPR/政策预期五维 VAR，AIC 选阶；保留同期相关性的
  残差块 bootstrap 生成 500 条路径，叠加情景漂移得到区间与概率。
- **长期（252 交易日）**：高/中/低三情景（漂移、波动、先验概率在 `config.yaml` 配置），
  用当前多因素读数经 softmax 调整情景概率，混合抽样 2000 条几何布朗运动路径；
  机构目标价中位数作为外部锚并列展示。
- **自适应滚动**：默认 180 交易日滚动窗口；每日运行把最新实际价格写入 SQLite 后重训，
  权重历史可在 `factor_weights` 表追溯。
- **回测**：最近 8 个滚动原点做样本外检验，报告 MAE/RMSE/方向命中率并与随机游走对比。

> 可选增强：`requirements.txt` 中注释了 `prophet` 与 `xgboost`，安装后可在对应模块扩展，
> 不安装不影响任何现有功能。

## 6. 自动化与部署（GitHub Actions + Pages / Streamlit）

1. 在 GitHub 新建 **public** 仓库并推送本项目；
2. 仓库 **Settings → Pages → Source 选择 "GitHub Actions"**；
3. （可选）Settings → Secrets and variables → Actions 添加 `EIA_API_KEY`；
4. 推送后 Actions 会在每天 **UTC 02:00（北京 10:00）** 运行，也可在 Actions 页手动触发；
   任务会把数据/报告 commit 回仓库，并把 `reports/latest` 部署为公开网页（Pages URL 在
   workflow 的 deploy 作业输出）。
5. Streamlit Cloud：连接同一仓库、启动文件填 `src/oilcast/app.py`，每日 Actions push 后
   平台会自动重启加载最新报告。

若改用 Render/Vercel 等托管：把 `reports/latest` 作为静态站点输出目录即可，流水线与托管解耦。

## 7. 常用配置（config/config.yaml）

| 参数 | 含义 | 默认 |
|---|---|---|
| `model.rolling_window` | 滚动训练窗口（交易日） | 180 |
| `model.short/mid/long_horizon_td` | 三周期预测步长 | 10 / 66 / 252 |
| `model.weight_ema_alpha` | 权重跨日平滑强度（越大越稳） | 0.6 |
| `model.prior_shrinkage` | 模型权重相对人工先验的信任度 | 0.7 |
| `prior_weights.*` | 九大因素人工先验（自动归一化） | 见文件 |
| `scenarios.*` | 长期三情景漂移/波动/先验概率 | 见文件 |
| `data_sources.prefer_live` | 是否优先抓取真实源 | true |

## 8. 测试

```bash
pytest -q          # 数据/特征/权重/短期模型/全链路+HTML 冒烟
```

## 9. 局限与免责

- 免费公开源存在字段变动/限流可能，系统设计为"逐表降级不中断"，并在报告显著位置标注
  `live / estimated / simulated`，请勿在数据为 simulated 时将结果当作真实行情；
- 规则词典情感打分是可解释基线，可替换为 FinBERT（输出表结构保持不变即可）；
- 预测区间反映历史波动与模型不确定性，**不构成投资建议**；第三方数据版权归原方所有。
