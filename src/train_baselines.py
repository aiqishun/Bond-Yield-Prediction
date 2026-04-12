#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        from sklearn.impute import SimpleImputer  # type: ignore
        from sklearn.linear_model import ElasticNet  # type: ignore
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore

        import joblib  # type: ignore
        import xgboost as xgb  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependencies. Install with: python -m pip install scikit-learn joblib xgboost"
        ) from exc

    return {
        "np": np,
        "pd": pd,
        "SimpleImputer": SimpleImputer,
        "ElasticNet": ElasticNet,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "joblib": joblib,
        "xgb": xgb,
    }


def default_stage1_liquidity_paths(outputs_dir: Path) -> SplitPaths:
    return SplitPaths(
        train_csv=outputs_dir / "stage1" / "liquidity_component_dataset_train.csv",
        val_csv=outputs_dir / "stage1" / "liquidity_component_dataset_val.csv",
    )


def default_stage1_demand_paths(outputs_dir: Path) -> SplitPaths:
    return SplitPaths(
        train_csv=outputs_dir / "stage1" / "demand_component_dataset_train.csv",
        val_csv=outputs_dir / "stage1" / "demand_component_dataset_val.csv",
    )


def default_stage2_final_10y_paths(outputs_dir: Path) -> SplitPaths:
    # Prefer the Stage2-only dataset (Stage1 preds + direct factors + yield lags) if present.
    stage2_train = outputs_dir / "datasets" / "modeling_dataset_stage2_train.csv"
    stage2_val = outputs_dir / "datasets" / "modeling_dataset_stage2_val.csv"
    if stage2_train.exists() and stage2_val.exists():
        return SplitPaths(train_csv=stage2_train, val_csv=stage2_val)
    return SplitPaths(
        train_csv=outputs_dir / "datasets" / "modeling_dataset_train.csv",
        val_csv=outputs_dir / "datasets" / "modeling_dataset_val.csv",
    )


def _label_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if str(c).startswith("label__")]


def _feature_columns(columns: list[str]) -> list[str]:
    cols = [c for c in columns if c not in {"date"} and not str(c).startswith("label__")]
    return cols


def _fit_predict_elasticnet(deps, x_train, y_train, x_val):
    Pipeline = deps["Pipeline"]
    SimpleImputer = deps["SimpleImputer"]
    StandardScaler = deps["StandardScaler"]
    ElasticNet = deps["ElasticNet"]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10_000)),
        ]
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    return model, pred


def _fit_predict_xgboost(deps, x_train, y_train, x_val):
    xgb = deps["xgb"]
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        objective="reg:squarederror",
    )
    model.fit(x_train, y_train, verbose=False)
    pred = model.predict(x_val)
    return model, pred


def _metrics(deps, y_true, y_pred) -> dict[str, float]:
    mean_absolute_error = deps["mean_absolute_error"]
    mean_squared_error = deps["mean_squared_error"]
    r2_score = deps["r2_score"]

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _train_one_dataset(
    *,
    deps,
    dataset_name: str,
    split: SplitPaths,
    out_dir: Path,
    models: list[str],
) -> dict[str, object]:
    pd = deps["pd"]
    joblib = deps["joblib"]

    train_df = pd.read_csv(split.train_csv)
    val_df = pd.read_csv(split.val_csv)
    cols = list(map(str, train_df.columns))

    label_cols = _label_columns(cols)
    feat_cols = _feature_columns(cols)

    if not label_cols:
        raise ValueError(f"{dataset_name}: no label columns found in {split.train_csv}")
    if not feat_cols:
        raise ValueError(f"{dataset_name}: no feature columns found in {split.train_csv}")

    x_train = train_df[feat_cols].copy()
    x_val = val_df[feat_cols].copy()

    # Ensure numeric matrices (XGBoost / ElasticNet expect numeric).
    for c in feat_cols:
        x_train[c] = pd.to_numeric(x_train[c], errors="coerce")
        x_val[c] = pd.to_numeric(x_val[c], errors="coerce")

    # Drop all-null features based on train split (keep schema stable for val).
    non_null_feats = [c for c in feat_cols if x_train[c].notna().any()]
    x_train = x_train[non_null_feats]
    x_val = x_val[non_null_feats]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "dataset": dataset_name,
        "train_csv": str(split.train_csv),
        "val_csv": str(split.val_csv),
        "n_train": int(train_df.shape[0]),
        "n_val": int(val_df.shape[0]),
        "n_features": int(len(feat_cols)),
        "n_labels": int(len(label_cols)),
        "labels": label_cols,
        "models": {},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    for model_name in models:
        model_out = out_dir / "models" / model_name
        pred_out = out_dir / "preds" / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        pred_out.mkdir(parents=True, exist_ok=True)

        per_label: dict[str, object] = {}
        all_metrics: list[dict[str, float]] = []

        for label in label_cols:
            y_train = pd.to_numeric(train_df[label], errors="coerce")
            y_val = pd.to_numeric(val_df[label], errors="coerce")

            train_mask = y_train.notna()
            val_mask = y_val.notna()
            if int(train_mask.sum()) < 20 or int(val_mask.sum()) < 5:
                raise ValueError(
                    f"{dataset_name}/{label}: too few non-null rows (train={int(train_mask.sum())}, val={int(val_mask.sum())})"
                )

            xtr = x_train.loc[train_mask]
            ytr = y_train.loc[train_mask]
            xv = x_val.loc[val_mask]
            yv = y_val.loc[val_mask]

            if model_name == "elasticnet":
                model, pred = _fit_predict_elasticnet(deps, xtr, ytr, xv)
            elif model_name == "xgboost":
                model, pred = _fit_predict_xgboost(deps, xtr, ytr, xv)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            m = _metrics(deps, yv, pred)
            all_metrics.append(m)

            joblib.dump(model, model_out / f"{label}.joblib")

            pred_df = pd.DataFrame(
                {
                    "date": val_df.loc[val_mask, "date"] if "date" in val_df.columns else pd.NA,
                    "y_true": yv,
                    "y_pred": pred,
                }
            )
            pred_df.to_csv(pred_out / f"{label}.csv", index=False, encoding="utf-8-sig")
            per_label[label] = {"metrics": m}

        avg = {
            "mae": float(sum(x["mae"] for x in all_metrics) / max(1, len(all_metrics))),
            "rmse": float(sum(x["rmse"] for x in all_metrics) / max(1, len(all_metrics))),
            "r2": float(sum(x["r2"] for x in all_metrics) / max(1, len(all_metrics))),
        }
        report["models"][model_name] = {"avg_metrics": avg, "per_label": per_label}

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train minimal baselines (ElasticNet + XGBoost) for stage1/stage2.")
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=repo_root() / "outputs",
        help="Project outputs directory (default: ./outputs).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root() / "outputs" / "baselines",
        help="Baseline artifacts output directory (default: ./outputs/baselines).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["elasticnet", "xgboost"],
        choices=["elasticnet", "xgboost"],
        help="Models to train (default: elasticnet xgboost).",
    )
    p.add_argument("--skip-stage1-liquidity", action="store_true")
    p.add_argument("--skip-stage1-demand", action="store_true")
    p.add_argument("--skip-stage2-final-10y", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    deps = _require_deps()
    args = parse_args(argv)

    outputs_dir: Path = args.outputs_dir
    out_dir: Path = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }

    if not args.skip_stage1_liquidity:
        rep = _train_one_dataset(
            deps=deps,
            dataset_name="stage1_liquidity",
            split=default_stage1_liquidity_paths(outputs_dir),
            out_dir=out_dir / "stage1_liquidity",
            models=list(args.models),
        )
        summary["runs"].append({"stage": "stage1_liquidity", "report": str((out_dir / 'stage1_liquidity' / 'report.json'))})

    if not args.skip_stage1_demand:
        rep = _train_one_dataset(
            deps=deps,
            dataset_name="stage1_demand",
            split=default_stage1_demand_paths(outputs_dir),
            out_dir=out_dir / "stage1_demand",
            models=list(args.models),
        )
        summary["runs"].append({"stage": "stage1_demand", "report": str((out_dir / 'stage1_demand' / 'report.json'))})

    if not args.skip_stage2_final_10y:
        rep = _train_one_dataset(
            deps=deps,
            dataset_name="stage2_final_10y",
            split=default_stage2_final_10y_paths(outputs_dir),
            out_dir=out_dir / "stage2_final_10y",
            models=list(args.models),
        )
        summary["runs"].append({"stage": "stage2_final_10y", "report": str((out_dir / 'stage2_final_10y' / 'report.json'))})

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
