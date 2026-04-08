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
| `现券成交分机构统计2024*~2025*.xlsx` | sheet `数据`：`交易日期`，`期限`，`债券类型`，`净买入/买入/卖出交易量（亿元）` | 日频（明细） | 构建 `trade__*` 资金流特征（当前实现聚合 7-10Y，按债券类型拆分） |
| `现券成交分机构统计20230101 - 20230331.xlsx` 等早期文件 | `交易日期`，`机构分类`，`净买入交易量（亿元）` | 日频 | 字段较少，当前 `trade__*` 特征构建会自动忽略（缺少 `债券类型/期限`） |
| `债券发行与到期20250818国债及政金债.xlsx` | `order_date_dt`，`year_1_lpr`，`year_5_lpr` | 月频 | LPR 因子（当前 pipeline 未接入，可扩展按日期合并） |
| `债市需求_现券净买入_分机构统计.xlsx` | `时间`，`year_week`，多项周度指标 | 周频/日对齐 | 看起来是已处理后的指标集合，当前 pipeline 未使用 |
| `10年期国债收益率.xlsx` | `时间`，若干宏观序列（含 `year_week`） | 周频/日对齐 | 文件名与内容不完全一致，当前 pipeline 未使用（备选特征来源） |
| `lpr.xlsx` | - | - | 目前不是标准 xlsx（读取报 “File is not a zip file”），无法直接使用 |

## 运行

生成输入清单（不需要安装 `pandas`）：

`python src/run_pipeline.py`

如需把每个 Excel 的首个 sheet 导出为 CSV（需要依赖）：

`python -m pip install pandas openpyxl`

`python src/run_pipeline.py --export-first-sheet`

生成建模数据集（按日期对齐合并，输出到 `outputs/datasets/`）：

`python src/run_pipeline.py --build-modeling-dataset`

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

### 建模数据集说明

依赖：

`python -m pip install pandas openpyxl`

输出文件（默认）：

- `outputs/datasets/modeling_dataset_full.csv`
- `outputs/datasets/modeling_dataset_train.csv`
- `outputs/datasets/modeling_dataset_val.csv`
- `outputs/datasets/modeling_dataset_metadata.json`

标签（label）与目标序列：

- 目标序列来自 `data/资金与流动性.xlsx` 的 `中债国债到期收益率:10年`（列名会自动匹配包含“10年+收益率”的列）
- 默认生成 `label__yield_10y__t+1`（预测 `t+1` 的 10 年期收益率）与 `label__yield_10y_chg__t+1`（预测变动）

特征（features）来源（按 `date` 对齐合并）：

- `data/高频宏观指标_day.xlsx`（或回退到 `data/高频宏观指标.xlsx`）
- `data/高频宏观指标_week.xlsx`
- `data/现券成交分机构统计*.xlsx`：抽取 sheet `数据`，对 7-10Y 的国债/政金债分类型汇总净买入/买入/卖出，形成 `trade__*` 特征
- 内置滞后特征：`feat__yield_10y__lag1`、`feat__yield_10y__lag5`

可用参数：

- `--horizon-days 5`：把标签改为 `t+5`（例如 `label__yield_10y__t+5`）
- `--val-ratio 0.2`：按时间顺序切分验证集比例
- `--no-split`：只输出 `*_full.csv`，不输出 train/val
- `--dataset-stem myset`：输出为 `outputs/datasets/myset_*.csv`

已知限制：

- `data/lpr.xlsx` 当前不是标准 xlsx（读取会报 “File is not a zip file”），所以不会参与合并
