#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class SplitPaths:
    train_csv: Path
    val_csv: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_deps():
    try:
        import pandas as pd  # type: ignore
        import joblib  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependencies. Install with: python -m pip install pandas joblib") from exc
    return {"pd": pd, "joblib": joblib}


def _label_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if str(c).startswith("label__")]


def _feature_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c not in {"date"} and not str(c).startswith("label__")]


def _default_stage1_paths(outputs_dir: Path, stage: str) -> SplitPaths:
    if stage == "stage1_liquidity":
        return SplitPaths(
            train_csv=outputs_dir / "stage1" / "liquidity_component_dataset_train.csv",
            val_csv=outputs_dir / "stage1" / "liquidity_component_dataset_val.csv",
        )
    if stage == "stage1_demand":
        return SplitPaths(
            train_csv=outputs_dir / "stage1" / "demand_component_dataset_train.csv",
            val_csv=outputs_dir / "stage1" / "demand_component_dataset_val.csv",
        )
    raise ValueError(f"Unknown stage: {stage}")


def _default_model_dir(baselines_dir: Path, stage: str) -> Path:
    return baselines_dir / stage / "models" / "xgboost"


def _load_splits(*, deps, split: SplitPaths):
    pd = deps["pd"]
    train_df = pd.read_csv(split.train_csv)
    val_df = pd.read_csv(split.val_csv)
    return train_df, val_df


def _build_feature_matrices(*, deps, train_df, val_df):
    pd = deps["pd"]
    cols = list(map(str, train_df.columns))
    label_cols = _label_columns(cols)
    feat_cols = _feature_columns(cols)
    if not label_cols:
        raise ValueError("No label columns found in stage1 dataset")
    if not feat_cols:
        raise ValueError("No feature columns found in stage1 dataset")

    x_train = train_df[feat_cols].copy()
    x_val = val_df[feat_cols].copy()
    for c in feat_cols:
        x_train[c] = pd.to_numeric(x_train[c], errors="coerce")
        x_val[c] = pd.to_numeric(x_val[c], errors="coerce")

    # Keep schema consistent with training: drop features that are all-null on train split.
    non_null_feats = [c for c in feat_cols if x_train[c].notna().any()]
    x_train = x_train[non_null_feats]
    x_val = x_val[non_null_feats]
    return label_cols, non_null_feats, x_train, x_val


def _predict_all_labels(
    *,
    deps,
    stage: str,
    model_dir: Path,
    train_df,
    val_df,
    x_train,
    x_val,
    labels: list[str],
):
    pd = deps["pd"]
    joblib = deps["joblib"]

    if "date" not in train_df.columns or "date" not in val_df.columns:
        raise ValueError("Stage1 datasets must contain a 'date' column")

    # Combine train+val for a unified historical in-sample prediction table.
    base_train = pd.DataFrame({"date": train_df["date"].astype(str), "split": "train"})
    base_val = pd.DataFrame({"date": val_df["date"].astype(str), "split": "val"})
    base = pd.concat([base_train, base_val], ignore_index=True).drop_duplicates(subset=["date"], keep="first")
    base = base.sort_values("date").reset_index(drop=True)

    out = base[["date"]].copy()

    for label in labels:
        model_path = model_dir / f"{label}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"{stage}: model not found for label {label}: {model_path}")
        model = joblib.load(model_path)
        # Predict on both splits (in-sample) and stitch by date.
        pred_train = model.predict(x_train)
        pred_val = model.predict(x_val)
        pred_df = pd.concat(
            [
                pd.DataFrame({"date": train_df["date"].astype(str), "pred": pred_train}),
                pd.DataFrame({"date": val_df["date"].astype(str), "pred": pred_val}),
            ],
            ignore_index=True,
        )
        pred_df = pred_df.drop_duplicates(subset=["date"], keep="first").sort_values("date")
        pred_col = "stage1_pred__" + str(label).removeprefix("label__")
        out = out.merge(pred_df.rename(columns={"pred": pred_col}), on="date", how="left")

    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export in-sample predictions from Stage1 XGBoost models.")
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=repo_root() / "outputs",
        help="Project outputs directory (default: ./outputs).",
    )
    p.add_argument(
        "--baselines-dir",
        type=Path,
        default=repo_root() / "outputs" / "baselines",
        help="Baselines directory (default: ./outputs/baselines).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: outputs/processed/stage1_predictions.csv).",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    deps = _require_deps()

    out_csv = (
        args.out_csv
        if args.out_csv is not None
        else (args.outputs_dir / "processed" / "stage1_predictions.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path = out_csv.with_name(out_csv.stem + "_report.json")

    combined = None
    reports: dict[str, object] = {}
    for stage in ["stage1_liquidity", "stage1_demand"]:
        split = _default_stage1_paths(args.outputs_dir, stage)
        model_dir = _default_model_dir(args.baselines_dir, stage)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Missing {stage} model directory: {model_dir}. Run: python src/train_baselines.py"
            )

        train_df, val_df = _load_splits(deps=deps, split=split)
        labels, feats, x_train, x_val = _build_feature_matrices(deps=deps, train_df=train_df, val_df=val_df)

        pred_df = _predict_all_labels(
            deps=deps,
            stage=stage,
            model_dir=model_dir,
            train_df=train_df,
            val_df=val_df,
            x_train=x_train,
            x_val=x_val,
            labels=labels,
        )

        combined = pred_df if combined is None else combined.merge(pred_df, on="date", how="outer")
        reports[stage] = {
            "stage": stage,
            "train_csv": str(split.train_csv),
            "val_csv": str(split.val_csv),
            "model_dir": str(model_dir),
            "n_train": int(train_df.shape[0]),
            "n_val": int(val_df.shape[0]),
            "n_features_used": int(len(feats)),
            "n_labels": int(len(labels)),
            "prediction_columns": ["stage1_pred__" + str(x).removeprefix("label__") for x in labels],
        }

    assert combined is not None
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(out_csv, index=False, encoding="utf-8-sig")

    report = {
        "out_csv": str(out_csv),
        "rows": int(combined.shape[0]),
        "cols": int(combined.shape[1]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stages": reports,
        "note": "In-sample predictions computed from Stage1 XGBoost models on their train+val splits (stitched by date).",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(os.sys.argv[1:]))

