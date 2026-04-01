# Debt-Forecasting-Model

## 项目结构

- `data/`: 原始 Excel 数据（`.xlsx`）
- `src/`: 代码
- `outputs/`: 运行产物（默认输出目录）

## 运行

生成输入清单（不需要安装 `pandas`）：

`python src/run_pipeline.py`

如需把每个 Excel 的首个 sheet 导出为 CSV（需要依赖）：

`python -m pip install pandas openpyxl`

`python src/run_pipeline.py --export-first-sheet`

生成建模数据集（按日期对齐合并，输出到 `outputs/datasets/`）：

`python src/run_pipeline.py --build-modeling-dataset`

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
