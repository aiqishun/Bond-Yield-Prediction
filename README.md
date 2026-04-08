# Debt-Forecasting-Model

## 项目结构

- `data/`: 原始 Excel 数据（`.xlsx`）
- `src/`: 代码
- `outputs/`: 运行产物（默认输出目录）

## 数据文件字典

| 文件名（data/） | 关键字段（示例） | 频率 | 用途/备注 |
| --- | --- | --- | --- |
| `资金与流动性.xlsx` | `日期`，`中债国债到期收益率:10年` | 日频 | 目标序列（y）与滞后特征来源 |
| `高频宏观指标_day.xlsx` | `截止日期`，发行/偿还/净融资等指标 | 日频 | 宏观/供给侧因子（按日期对齐后做前向填充） |
| `高频宏观指标.xlsx` | 同上（字段基本一致） | 日频 | 作为 `高频宏观指标_day.xlsx` 缺失时的回退输入 |
| `高频宏观指标_week.xlsx` | `时间`，粗钢产量/开工率/运价等指标 | 周频 | 宏观因子（按日期对齐后做前向填充） |
| `现券成交分机构统计20230701 - 20230930.xlsx` | `R_日期`，`DR001/DR007/R001/R007`，`货币日期`，`中国:公开市场操作:*` | 日频 | 资金面综合表（清洗后的综合表，主用途固定：资金面与货币操作类指标） |
| `现券成交分机构统计20230401 - 20230630.xlsx` | `机构类型`，`机构分类` | 静态 | 机构分类映射表（清洗后的映射表，主用途固定：机构类型 -> 类别） |
| `现券成交分机构统计20230101 - 20230331.xlsx` | `交易日期`，`机构分类`，`净买入交易量（亿元）` | 日频 | 需求面分类汇总表（清洗后的综合表，覆盖长区间，主用途固定：需求面 fallback/补齐） |
| `现券成交分机构统计2024*~2025*.xlsx` | sheet `数据`：`交易日期`，`机构类型`，`期限`，`债券类型`，`净买入/买入/卖出交易量（亿元）` | 日频（明细） | 现券明细表（主用途：需求面按机构汇总 + `trade__*` 资金流特征；不承担资金面/映射/分类汇总角色） |
| `债券发行与到期20250818国债及政金债.xlsx` | `order_date_dt`，`year_1_lpr`，`year_5_lpr` | 月频 | LPR 因子（低频，转日频后 forward fill；主用途固定：直接因子） |
| `债市需求_现券净买入_分机构统计.xlsx` | `时间`，`year_week`，多项周度指标 | 周频/日对齐 | 看起来是已处理后的指标集合，当前 pipeline 未使用 |
| `10年期国债收益率.xlsx` | `时间/order_date_dt`，若干序列（含 `year_week`） | 日频 | 目标变量候选源（当前文件内未识别到“10年收益率”列时会回退到 `资金与流动性.xlsx`） |
| `lpr.xlsx` | - | - | 目前不是标准 xlsx（读取报 “File is not a zip file”），无法直接使用 |

## 运行

生成输入清单（不需要安装 `pandas`）：

`python src/run_pipeline.py`

### 口径冻结（主线定义）

本项目已冻结关键口径（见 `pipeline_spec.json` 与 `SPEC.md`），默认不允许随意改动：

- 主线 `horizon_days = 20`（Stage1/Stage2 统一以 `t+20` 为主线）
- Feature Registry 的 `role`/`stage` 分类规则
- legacy/baseline 命名规则
- Stage1 / Stage2 主线定义

`src/run_pipeline.py` 默认会校验 spec：如果你传入的 `--horizon-days` 或 `--dataset-stem` 偏离 spec，会直接报错退出。
如需“有意识地漂移验证”，显式加 `--allow-spec-drift`，并把验证产物放在 `outputs/experiments/validation/`（不要留在 `outputs/datasets/`）。

如需把每个 Excel 的首个 sheet 导出为 CSV（需要依赖）：

`python -m pip install pandas openpyxl`

`python src/run_pipeline.py --export-first-sheet`

## 处理顺序（分阶段产出中间表）

### 第1步：目标变量表（y）

从源文件抽取成两列：`date | yield_10y`，并做：

- 日期列统一命名为 `date`，转成 datetime
- 收益率列统一命名为 `yield_10y`，转成数值
- 按日期排序、去重、去空值

运行：

`python src/run_pipeline.py --export-target-table`

输出：

- `outputs/targets/yield_10y.csv`

说明：

- 优先尝试从 `data/10年期国债收益率.xlsx` 识别“10年收益率”列；若未找到会回退到 `data/资金与流动性.xlsx` 的 `中债国债到期收益率:10年`

### 第2步：资金面表

把资金面拆成两张表：

- `liquidity_features.csv`：用于资金面分量（DR/R/利差等）
- `direct_policy_factors.csv`：货币操作类直接因子（公开市场净投放等）+ LPR

运行：

`python src/run_pipeline.py --export-liquidity-tables`

输出：

- `outputs/liquidity_features.csv`：`date | dr007 | r007 | spread_* | ...`
- `outputs/direct_policy_factors.csv`：`date | omo_net | omo_inject | omo_withdraw | lpr_1y | lpr_5y | ...`
- `outputs/funding_split_report.json`：本次拆分使用的源文件与缺失字段报告

### 第3步：需求面表（现券净买入）

处理流程：

1. 拼接所有季度“现券成交分机构统计”文件（抽取 `交易日期/机构类型/净买入交易量（亿元）`）
2. 统一字段得到：`date | institution_name | net_buy`
3. 用机构分类表 merge，得到：`date | institution_name | category | net_buy`
4. 按 `date + category` 聚合求和，得到宽表：`date | 银行 | 基金 | 保险 | 券商 | 理财子 | 其他`

运行：

`python src/run_pipeline.py --export-demand-table`

输出：

- `outputs/demand_long.csv`：`date | institution_name | category | net_buy`
- `outputs/demand_wide.csv`：`date | 银行 | 基金 | 保险 | 券商 | 理财子 | 其他`
- `outputs/institution_category_mapping.csv`：机构类型到类别映射（当前使用 `data/现券成交分机构统计20230401 - 20230630.xlsx`）
- `outputs/demand_build_report.json`：本次拼接使用的源文件与 fallback 文件记录

### 第4步：直接因子表（direct factors）

主要包括：

- LPR（低频，转日频后 forward fill）
- 国债净融资（如果数据源里存在对应列；否则暂时留空并在报告里提示）
- 货币操作类指标（公开市场净投放/投放/回笼等）

运行：

`python src/run_pipeline.py --export-direct-factors`

输出：

- `outputs/direct_factors.csv`：`date | omo_net | lpr_1y | gov_net_financing | net_financing_total | ...`
- `outputs/direct_factors_report.json`：本次使用的源文件与缺失字段列表

### 第5步：宏观表（统一到日频）

你有两类宏观指标：

- 日频宏观：直接保留为 `date | macro_day__*`
- 周频宏观：先转日期并 merge 到日频日期框架，再 forward fill 到每天，形成 `macro_week__*`

运行：

`python src/run_pipeline.py --export-macro-table`

输出：

- `outputs/macro_features.csv`：`date | macro_day__* | macro_week__*`
- `outputs/macro_features_report.json`：宏观特征列清单与计数

### 第6步：合成 master dataset

以目标变量表为基准（`date,yield_10y`），按 `date` 依次左连接：

- 资金面表（`liquidity_features.csv`）
- 需求面表（`demand_wide.csv`）
- 直接因子表（`direct_factors.csv`）
- 宏观表（`macro_features.csv`）

运行：

`python src/run_pipeline.py --export-master-dataset`

输出：

- `outputs/processed/master_dataset.csv`
- `outputs/processed/master_dataset_report.json`

缺失值处理（当前实现）：

- 需求/公开市场操作/融资等“流量型”列缺失填 `0`
- 其他列做 forward fill（更适合低频或水平类变量）

### 第7步：生成特征与建模集 (Train/Val Split)

依赖：

`python -m pip install pandas openpyxl`

说明：

- 第6步产出 `master_dataset.csv` 是“基础宽表”
- 第7步生成滞后特征（Lag）与监督学习标签（Label），并产出 Train/Val 切分文件（当前实现为直接从源表对齐构建；后续也可以改为从第6步宽表读取）

两阶段建模进度说明：

- “分量预测”作为独立模型层，目前仅完成了数据准备（中间表与可监督学习的 component dataset），尚未落地训练/预测接口与预测结果文件的约定
- 如需导出 stage-1 分量预测训练集（预测未来资金面/需求面分量），使用：`python src/run_pipeline.py --export-component-datasets --horizon-days 20`

补充：特征注册表（Feature Registry）

- 导出：`python src/run_pipeline.py --export-feature-registry --horizon-days 20`
- 输出：`outputs/metadata/feature_registry.csv`（包含每列的 `role/stage/frequency/fill_method` 等元信息，便于快速排查“某列是否直接因子/是否只用于分量层/缺失值如何处理”等）

运行：

`python src/run_pipeline.py --build-modeling-dataset`

输出文件（默认）：

- `outputs/datasets/modeling_dataset_full.csv`
- `outputs/datasets/modeling_dataset_train.csv`
- `outputs/datasets/modeling_dataset_val.csv`
- `outputs/datasets/modeling_dataset_metadata.json`

标签（label）与目标序列：

- 目标序列来自 `data/资金与流动性.xlsx` 的 `中债国债到期收益率:10年`（列名会自动匹配包含“10年+收益率”的列）
- 主线默认生成 `label__yield_10y__t+20`（预测 `t+20` 的 10 年期收益率）与 `label__yield_10y_chg__t+20`（预测变动）
- baseline（legacy）仅保留一套 canonical：`outputs/datasets/modeling_dataset_legacy_t+1_*.csv`（用于对比；不作为主线训练目标）
- spec-drift 验证产物（重复版本/其它 horizon）统一放到：`outputs/experiments/validation/datasets/`

特征（features）来源（按 `date` 对齐合并）：

- `data/高频宏观指标_day.xlsx`（或回退到 `data/高频宏观指标.xlsx`）
- `data/高频宏观指标_week.xlsx`
- `data/现券成交分机构统计*.xlsx`：抽取 sheet `数据`，对 7-10Y 的国债/政金债分类型汇总净买入/买入/卖出，形成 `trade__*` 特征
- 内置滞后特征：`feat__yield_10y__lag1`、`feat__yield_10y__lag5`

可用参数：

- `--horizon-days 5`：把标签改为 `t+5`（例如 `label__yield_10y__t+5`）
- 如果目标是直接预测“20 天后”的收益率：使用 `--horizon-days 20`
- `--val-ratio 0.2`：按时间顺序切分验证集比例
- `--no-split`：只输出 `*_full.csv`，不输出 train/val
- `--dataset-stem myset`：输出为 `outputs/datasets/myset_*.csv`
- `--allow-spec-drift`：允许偏离冻结 spec（仅用于验证；产物会归档为 legacy/validation）

已知限制：

- `data/lpr.xlsx` 当前不是标准 xlsx（读取会报 “File is not a zip file”），所以不会参与合并

## 最小 Baseline（Elastic Net / XGBoost）

为快速验证数据与主线口径，提供一个最小 baseline 训练与汇总：

依赖：

`python -m pip install scikit-learn joblib xgboost`

训练顺序（Stage1 liquidity -> Stage1 demand -> Stage2 final_10y）：

`python src/train_baselines.py`

输出：

- 模型与预测：`outputs/baselines/`（含 `stage1_liquidity/`、`stage1_demand/`、`stage2_final_10y/`）
- 每个 label 的预测文件：`outputs/baselines/*/preds/{elasticnet,xgboost}/*.csv`
- 每个 stage 的报告：`outputs/baselines/*/report.json`

统一结果表（从三份报告抽关键指标，含 RMSE/MAE/方向准确率等）：

`python src/summarize_baselines.py`

输出：

- `outputs/baselines/results_summary.csv`
- `outputs/baselines/results_summary_metadata.json`
