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
