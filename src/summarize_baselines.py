#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class DateRange:
    start: str | None
    end: str | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_pandas_numpy():
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependencies. Install with: python -m pip install numpy pandas") from exc
    return pd, np


def _read_date_range(pd, csv_path: Path) -> DateRange:
    if not csv_path.exists():
        return DateRange(None, None)
    df = pd.read_csv(csv_path, usecols=["date"])
    if df.empty:
        return DateRange(None, None)
    d = pd.to_datetime(df["date"], errors="coerce")
    d = d.dropna()
    if d.empty:
        return DateRange(None, None)
    return DateRange(start=str(d.min().date()), end=str(d.max().date()))


def _prediction_kind(target: str) -> str:
    # "change" labels include explicit chg.
    if "yield_10y_chg" in target or re.search(r"__chg__", target):
        return "change"
    return "level"

def _infer_horizon_days(target: str) -> int | None:
    m = re.search(r"__t\+(\d+)$", str(target))
    return int(m.group(1)) if m else None


def _direction_accuracy(pd, np, preds_csv: Path, *, prediction_kind: str) -> float | None:
    if not preds_csv.exists():
        return None

    df = pd.read_csv(preds_csv)
    if df.empty or not {"y_true", "y_pred"}.issubset(df.columns):
        return None

    y_true = pd.to_numeric(df["y_true"], errors="coerce")
    y_pred = pd.to_numeric(df["y_pred"], errors="coerce")
    m = y_true.notna() & y_pred.notna()
    if int(m.sum()) < 3:
        return None
    y_true = y_true[m].reset_index(drop=True)
    y_pred = y_pred[m].reset_index(drop=True)

    if prediction_kind == "change":
        true_sign = np.sign(y_true.to_numpy())
        pred_sign = np.sign(y_pred.to_numpy())
        keep = true_sign != 0
        if int(keep.sum()) == 0:
            return None
        acc = float((true_sign[keep] == pred_sign[keep]).mean())
        return acc

    # level: compare direction of changes in the validation window
    dy_true = y_true.diff().dropna()
    dy_pred = y_pred.diff().dropna()
    if dy_true.empty or dy_pred.empty:
        return None
    # align lengths
    n = min(len(dy_true), len(dy_pred))
    dy_true = dy_true.iloc[:n]
    dy_pred = dy_pred.iloc[:n]
    true_sign = np.sign(dy_true.to_numpy())
    pred_sign = np.sign(dy_pred.to_numpy())
    keep = true_sign != 0
    if int(keep.sum()) == 0:
        return None
    acc = float((true_sign[keep] == pred_sign[keep]).mean())
    return acc


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize baseline reports into a unified table.")
    p.add_argument(
        "--baselines-dir",
        type=Path,
        default=repo_root() / "outputs" / "baselines",
        help="Baselines root dir (default: ./outputs/baselines).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=repo_root() / "outputs" / "baselines" / "results_summary.csv",
        help="Output CSV path (default: ./outputs/baselines/results_summary.csv).",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    pd, np = _require_pandas_numpy()
    args = parse_args(argv)

    baselines_dir: Path = args.baselines_dir
    out_csv: Path = args.out_csv

    rows: list[dict[str, object]] = []

    stage_dirs = [p for p in sorted(baselines_dir.iterdir()) if p.is_dir()]
    for stage_dir in stage_dirs:
        report_path = stage_dir / "report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))

        stage = str(report.get("dataset") or stage_dir.name)
        train_csv = Path(str(report.get("train_csv")))
        val_csv = Path(str(report.get("val_csv")))

        train_range = _read_date_range(pd, train_csv)
        val_range = _read_date_range(pd, val_csv)

        n_train = int(report.get("n_train") or 0)
        n_val = int(report.get("n_val") or 0)

        models = report.get("models") or {}
        if not isinstance(models, dict):
            continue

        for model_name, model_info in models.items():
            if not isinstance(model_info, dict):
                continue
            per_label = model_info.get("per_label") or {}
            if not isinstance(per_label, dict):
                continue
            for target, info in per_label.items():
                if not isinstance(info, dict):
                    continue
                metrics = info.get("metrics") or {}
                if not isinstance(metrics, dict):
                    metrics = {}
                mae = metrics.get("mae")
                rmse = metrics.get("rmse")

                # compute direction accuracy from saved predictions
                preds_csv = stage_dir / "preds" / str(model_name) / f"{target}.csv"
                kind = _prediction_kind(str(target))
                horizon_days = _infer_horizon_days(str(target))
                dir_acc = _direction_accuracy(pd, np, preds_csv, prediction_kind=kind)

                rows.append(
                    {
                        "stage": stage,
                        "target": str(target),
                        "model": str(model_name),
                        "train_start": train_range.start,
                        "train_end": train_range.end,
                        "val_start": val_range.start,
                        "val_end": val_range.end,
                        "n_train": n_train,
                        "n_val": n_val,
                        "rmse": float(rmse) if isinstance(rmse, (int, float)) and not math.isnan(float(rmse)) else None,
                        "mae": float(mae) if isinstance(mae, (int, float)) and not math.isnan(float(mae)) else None,
                        "direction_accuracy": dir_acc,
                        "prediction_kind": kind,
                        "is_change": True if kind == "change" else False,
                        "horizon_days": horizon_days,
                        "preds_csv": str(preds_csv),
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["stage", "target", "model"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(df.shape[0]),
        "csv": str(out_csv),
    }
    (out_csv.parent / "results_summary_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
