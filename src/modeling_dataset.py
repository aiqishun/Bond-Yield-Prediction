from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetArtifacts:
    full_csv: Path
    train_csv: Path | None
    val_csv: Path | None
    metadata_json: Path
    split_date: str | None


def _require_pandas() -> tuple[object, object]:
    try:
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency. Install with: python3 -m pip install pandas openpyxl"
        ) from exc
    return pd, np


def _to_date(pd: object, series) -> object:
    # Normalize to midnight; keeps timezone-naive dates for simple joins.
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _coerce_numeric(pd: object, df, exclude: set[str]) -> object:
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == "O":
            df[col] = df[col].replace({"--": pd.NA, "—": pd.NA, "-": pd.NA})
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_yield_10y(data_dir: Path):
    pd, _ = _require_pandas()
    path = data_dir / "资金与流动性.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing target file: {path}")

    df = pd.read_excel(path, sheet_name=0)
    if df.empty:
        raise ValueError(f"Empty Excel: {path}")

    date_col = "日期" if "日期" in df.columns else df.columns[0]
    yield_col = None
    for c in df.columns:
        if "10年" in str(c) and "收益率" in str(c):
            yield_col = c
            break
    if yield_col is None:
        # Fall back: second column.
        yield_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    out = df[[date_col, yield_col]].rename(columns={date_col: "date", yield_col: "yield_10y"})
    out["date"] = _to_date(pd, out["date"])
    out["yield_10y"] = pd.to_numeric(out["yield_10y"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
    return out.reset_index(drop=True)


def read_macro_day(data_dir: Path):
    pd, _ = _require_pandas()
    path = data_dir / "高频宏观指标_day.xlsx"
    if not path.exists():
        path = data_dir / "高频宏观指标.xlsx"
    if not path.exists():
        logging.warning("Skip macro_day: file not found")
        return pd.DataFrame(columns=["date"])

    df = pd.read_excel(path, sheet_name=0)
    if df.empty:
        logging.warning("Skip macro_day: empty file %s", path)
        return pd.DataFrame(columns=["date"])

    date_col = "截止日期" if "截止日期" in df.columns else ("时间" if "时间" in df.columns else None)
    if date_col is None:
        date_col = df.columns[0]

    out = df.copy()
    out["date"] = _to_date(pd, out[date_col])
    drop_cols = {date_col, "起始日期", "序号"}
    keep_cols = [c for c in out.columns if c not in drop_cols and c != "date"]
    out = out[["date", *keep_cols]]
    out = _coerce_numeric(pd, out, exclude={"date"})

    feature_cols = [c for c in out.columns if c != "date"]
    out = out.rename(columns={c: f"macro_day__{c}" for c in feature_cols})
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
    return out.reset_index(drop=True)


def read_macro_week(data_dir: Path):
    pd, _ = _require_pandas()
    path = data_dir / "高频宏观指标_week.xlsx"
    if not path.exists():
        logging.warning("Skip macro_week: file not found")
        return pd.DataFrame(columns=["date"])

    try:
        df = pd.read_excel(path, sheet_name="data")
    except Exception:
        df = pd.read_excel(path, sheet_name=0)

    if df.empty:
        logging.warning("Skip macro_week: empty file %s", path)
        return pd.DataFrame(columns=["date"])

    date_col = "时间" if "时间" in df.columns else df.columns[0]
    out = df.copy()
    out["date"] = _to_date(pd, out[date_col])
    if date_col in out.columns:
        out = out.drop(columns=[date_col])
    out = out[["date", *[c for c in out.columns if c != "date"]]]
    out = _coerce_numeric(pd, out, exclude={"date"})

    feature_cols = [c for c in out.columns if c != "date"]
    out = out.rename(columns={c: f"macro_week__{c}" for c in feature_cols})
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
    return out.reset_index(drop=True)


def read_trade_flows_7_10y(data_dir: Path):
    pd, _ = _require_pandas()
    files = sorted(data_dir.glob("现券成交分机构统计*.xlsx"))
    if not files:
        logging.warning("Skip trade_flows: no matching files")
        return pd.DataFrame(columns=["date"])

    frames = []
    for path in files:
        try:
            xl = pd.ExcelFile(path)
            sheet = "数据" if "数据" in xl.sheet_names else 0
            df = pd.read_excel(path, sheet_name=sheet)
        except Exception as exc:
            logging.warning("Skip trade file %s: %s", path, exc)
            continue
        if df is None or df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["date"])

    df = pd.concat(frames, ignore_index=True)
    required = {"交易日期", "债券类型", "净买入交易量（亿元）", "买入交易量（亿元）", "卖出交易量（亿元）"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"trade_flows missing columns: {missing}")

    out = df.copy()
    out["date"] = _to_date(pd, out["交易日期"])

    bond_type_map = {
        "国债-新债": "gov_new",
        "国债-老债": "gov_old",
        "政策性金融债-新债": "policy_new",
        "政策性金融债-老债": "policy_old",
    }
    out["bond_type"] = out["债券类型"].map(bond_type_map)
    out = out.dropna(subset=["date", "bond_type"])

    for src, dst in [
        ("净买入交易量（亿元）", "net"),
        ("买入交易量（亿元）", "buy"),
        ("卖出交易量（亿元）", "sell"),
    ]:
        out[dst] = pd.to_numeric(out[src].replace({"--": pd.NA, "—": pd.NA, "-": pd.NA}), errors="coerce")

    agg = (
        out.groupby(["date", "bond_type"], as_index=False)[["net", "buy", "sell"]]
        .sum(numeric_only=True)
        .sort_values(["date", "bond_type"])
    )

    wide = agg.pivot(index="date", columns="bond_type", values=["net", "buy", "sell"])
    wide.columns = [f"trade__{metric}__{bond_type}" for metric, bond_type in wide.columns.to_list()]
    wide = wide.reset_index()
    wide = wide.sort_values("date").drop_duplicates(subset=["date"])
    return wide.reset_index(drop=True)


def build_modeling_dataset(data_dir: Path, horizon_days: int) -> object:
    pd, _ = _require_pandas()

    target = read_yield_10y(data_dir)
    macro_day = read_macro_day(data_dir)
    macro_week = read_macro_week(data_dir)
    trade = read_trade_flows_7_10y(data_dir)

    df = target.merge(macro_day, on="date", how="left").merge(macro_week, on="date", how="left").merge(
        trade, on="date", how="left"
    )
    df = df.sort_values("date").reset_index(drop=True)

    macro_cols = [c for c in df.columns if c.startswith("macro_day__") or c.startswith("macro_week__")]
    if macro_cols:
        df[macro_cols] = df[macro_cols].ffill()

    trade_cols = [c for c in df.columns if c.startswith("trade__")]
    if trade_cols:
        df[trade_cols] = df[trade_cols].fillna(0.0)

    df["feat__yield_10y__lag1"] = df["yield_10y"].shift(1)
    df["feat__yield_10y__lag5"] = df["yield_10y"].shift(5)

    label_col = f"label__yield_10y__t+{horizon_days}"
    chg_col = f"label__yield_10y_chg__t+{horizon_days}"
    df[label_col] = df["yield_10y"].shift(-horizon_days)
    df[chg_col] = df[label_col] - df["yield_10y"]

    df = df.dropna(subset=[label_col, "feat__yield_10y__lag1"])
    return df.reset_index(drop=True)


def write_dataset_artifacts(
    df,
    out_dir: Path,
    dataset_stem: str,
    val_ratio: float,
    do_split: bool,
    horizon_days: int,
) -> DatasetArtifacts:
    pd, _ = _require_pandas()

    out_dir.mkdir(parents=True, exist_ok=True)
    full_csv = out_dir / f"{dataset_stem}_full.csv"
    df.to_csv(full_csv, index=False, encoding="utf-8-sig")

    split_date = None
    train_csv = None
    val_csv = None
    if do_split:
        dates = df["date"].dropna().sort_values().unique().tolist()
        if len(dates) >= 10:
            cut = max(1, min(len(dates) - 1, int(len(dates) * (1.0 - val_ratio))))
            split_date = str(pd.to_datetime(dates[cut]).date())
            train = df[df["date"] < dates[cut]]
            val = df[df["date"] >= dates[cut]]
            train_csv = out_dir / f"{dataset_stem}_train.csv"
            val_csv = out_dir / f"{dataset_stem}_val.csv"
            train.to_csv(train_csv, index=False, encoding="utf-8-sig")
            val.to_csv(val_csv, index=False, encoding="utf-8-sig")
        else:
            logging.warning("Skip split: too few unique dates (%s)", len(dates))

    metadata = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "horizon_days": int(horizon_days),
        "val_ratio": float(val_ratio),
        "split_date": split_date,
        "columns": list(map(str, df.columns)),
    }
    metadata_json = out_dir / f"{dataset_stem}_metadata.json"
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return DatasetArtifacts(
        full_csv=full_csv,
        train_csv=train_csv,
        val_csv=val_csv,
        metadata_json=metadata_json,
        split_date=split_date,
    )
