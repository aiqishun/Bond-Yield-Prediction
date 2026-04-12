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


def _set_mpl_config_dir(outputs_dir: Path) -> None:
    # Matplotlib tries to write to ~/.config which may be read-only in some devcontainers.
    os.environ.setdefault("MPLCONFIGDIR", str(outputs_dir / ".mplconfig"))
    mpl_dir = Path(os.environ["MPLCONFIGDIR"])
    mpl_dir.mkdir(parents=True, exist_ok=True)
    # If fonts were installed after the cache was created, Matplotlib may keep using an outdated fontlist.
    for p in mpl_dir.glob("fontlist-v*.json"):
        try:
            p.unlink()
        except Exception:
            pass


def _require_deps():
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import joblib  # type: ignore

        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        # Font config: prefer CJK-capable fonts so Chinese feature names render correctly in SHAP plots.
        from matplotlib import font_manager, rcParams  # type: ignore

        # Ensure Matplotlib sees freshly installed fonts even if a cache existed.
        try:
            if hasattr(font_manager, "_rebuild"):  # matplotlib <= 3.8
                font_manager._rebuild()  # type: ignore[attr-defined]
            elif hasattr(font_manager, "_load_fontmanager"):  # matplotlib >= 3.9
                font_manager.fontManager = font_manager._load_fontmanager(try_read_cache=False)  # type: ignore[attr-defined]
        except Exception:
            pass

        preferred_fonts = [
            "Noto Sans CJK SC",
            "Noto Sans CJK TC",
            "Noto Sans CJK JP",
            "Noto Serif CJK SC",
            "WenQuanYi Zen Hei",
            "SimHei",
            "Microsoft YaHei",
            "PingFang SC",
            "DejaVu Sans",
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = [f for f in preferred_fonts if f in available]
        if chosen:
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = chosen
        rcParams["axes.unicode_minus"] = False
        import matplotlib.pyplot as plt  # type: ignore

        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependencies. Install with: python -m pip install shap matplotlib seaborn"
        ) from exc

    return {"np": np, "pd": pd, "joblib": joblib, "plt": plt, "shap": shap}


def _label_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if str(c).startswith("label__")]


def _feature_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c not in {"date"} and not str(c).startswith("label__")]


def _default_split_paths(stage: str, outputs_dir: Path) -> SplitPaths:
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
    if stage == "stage2_final_10y":
        # Prefer the Stage2-only dataset (Stage1 preds + direct factors + yield lags) if present.
        stage2_train = outputs_dir / "datasets" / "modeling_dataset_stage2_train.csv"
        stage2_val = outputs_dir / "datasets" / "modeling_dataset_stage2_val.csv"
        if stage2_train.exists() and stage2_val.exists():
            return SplitPaths(train_csv=stage2_train, val_csv=stage2_val)
        return SplitPaths(
            train_csv=outputs_dir / "datasets" / "modeling_dataset_train.csv",
            val_csv=outputs_dir / "datasets" / "modeling_dataset_val.csv",
        )
    raise ValueError(f"Unknown stage: {stage}")


def _default_model_dir(stage: str, baselines_dir: Path) -> Path:
    return baselines_dir / stage / "models" / "xgboost"


def _load_feature_matrices(*, deps, split: SplitPaths):
    pd = deps["pd"]
    train_df = pd.read_csv(split.train_csv)
    val_df = pd.read_csv(split.val_csv)

    cols = list(map(str, train_df.columns))
    label_cols = _label_columns(cols)
    feat_cols = _feature_columns(cols)
    if not label_cols:
        raise ValueError(f"No label columns found in {split.train_csv}")
    if not feat_cols:
        raise ValueError(f"No feature columns found in {split.train_csv}")

    x_train = train_df[feat_cols].copy()
    x_val = val_df[feat_cols].copy()
    for c in feat_cols:
        x_train[c] = pd.to_numeric(x_train[c], errors="coerce")
        x_val[c] = pd.to_numeric(x_val[c], errors="coerce")

    non_null_feats = [c for c in feat_cols if x_train[c].notna().any()]
    x_train = x_train[non_null_feats]
    x_val = x_val[non_null_feats]

    return train_df, val_df, label_cols, non_null_feats, x_train, x_val


def _run_one_label(
    *,
    deps,
    label: str,
    model_path: Path,
    train_df,
    val_df,
    x_train,
    x_val,
    out_dir: Path,
    max_samples: int,
    seed: int,
) -> None:
    np = deps["np"]
    pd = deps["pd"]
    joblib = deps["joblib"]
    plt = deps["plt"]
    shap = deps["shap"]

    model = joblib.load(model_path)

    y_train = pd.to_numeric(train_df[label], errors="coerce")
    y_val = pd.to_numeric(val_df[label], errors="coerce")
    train_mask = y_train.notna()
    val_mask = y_val.notna()

    xtr = x_train.loc[train_mask]
    ytr = y_train.loc[train_mask]
    xv = x_val.loc[val_mask]
    yv = y_val.loc[val_mask]

    if xv.shape[0] == 0:
        raise ValueError(f"{label}: no non-null validation rows")

    if max_samples > 0 and xv.shape[0] > max_samples:
        xv = xv.sample(n=max_samples, random_state=seed)
        yv = yv.loc[xv.index]

    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(xv)

    values = explanation.values
    mean_abs = np.abs(values).mean(axis=0)
    imp = (
        pd.DataFrame({"feature": list(xv.columns), "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(out_dir / "shap_importance.csv", index=False, encoding="utf-8-sig")

    # Summary plots (headless).
    plt.figure()
    shap.summary_plot(values, xv, feature_names=list(xv.columns), show=False, plot_type="dot")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_dot.png", dpi=200)
    plt.close()

    plt.figure()
    shap.summary_plot(values, xv, feature_names=list(xv.columns), show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_bar.png", dpi=200)
    plt.close()

    meta = {
        "label": label,
        "model_path": str(model_path),
        "n_train_rows_used": int(train_mask.sum()),
        "n_val_rows_used": int(val_mask.sum()),
        "n_val_rows_shap": int(xv.shape[0]),
        "n_features": int(xv.shape[1]),
        "max_samples": int(max_samples),
        "seed": int(seed),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "shap_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute SHAP explanations for saved XGBoost baselines.")
    p.add_argument(
        "--stage",
        required=True,
        choices=["stage1_liquidity", "stage1_demand", "stage2_final_10y"],
        help="Which baseline stage to explain (matches outputs/baselines/<stage>/...).",
    )
    p.add_argument(
        "--label",
        default=None,
        help="Label column name to explain (default: explain all labels in the dataset).",
    )
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
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for SHAP artifacts (default: ./outputs/interpretability/shap/<stage>/xgboost).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Max validation rows to compute SHAP on per label (default: 1000; set 0 for all).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for subsampling (default: 42).")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    _set_mpl_config_dir(args.outputs_dir)
    deps = _require_deps()

    split = _default_split_paths(args.stage, args.outputs_dir)
    model_dir = _default_model_dir(args.stage, args.baselines_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    out_root = (
        args.out_dir
        if args.out_dir is not None
        else args.outputs_dir / "interpretability" / "shap" / args.stage / "xgboost"
    )

    train_df, val_df, label_cols, _, x_train, x_val = _load_feature_matrices(deps=deps, split=split)

    labels = [args.label] if args.label else label_cols
    for label in labels:
        model_path = model_dir / f"{label}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for label: {label} ({model_path})")
        _run_one_label(
            deps=deps,
            label=label,
            model_path=model_path,
            train_df=train_df,
            val_df=val_df,
            x_train=x_train,
            x_val=x_val,
            out_dir=out_root / label,
            max_samples=args.max_samples,
            seed=args.seed,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(os.sys.argv[1:]))
