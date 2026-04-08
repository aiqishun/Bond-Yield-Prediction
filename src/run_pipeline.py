#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class InputFile:
    path: Path
    size_bytes: int
    mtime_utc: datetime


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def iter_xlsx_files(data_dir: Path) -> Iterable[Path]:
    yield from sorted(p for p in data_dir.glob("*.xlsx") if p.is_file())


def build_manifest(files: Iterable[Path]) -> list[InputFile]:
    manifest: list[InputFile] = []
    for path in files:
        stat = path.stat()
        manifest.append(
            InputFile(
                path=path,
                size_bytes=stat.st_size,
                mtime_utc=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        )
    return manifest


def write_manifest_csv(manifest: list[InputFile], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "size_bytes", "mtime_utc"])
        for item in manifest:
            writer.writerow([str(item.path), item.size_bytes, item.mtime_utc.isoformat()])


def try_export_first_sheet_to_csv(
    xlsx_path: Path,
    out_csv_path: Path,
    max_rows: int | None,
) -> tuple[bool, str | None]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False, "Missing dependency: pandas"

    try:
        df = pd.read_excel(xlsx_path)  # relies on openpyxl for .xlsx
    except Exception as exc:
        return True, f"Failed to read Excel file: {exc}"

    if max_rows is not None:
        df = df.head(max_rows)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    return True, None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the project pipeline (input discovery + optional Excel export)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root() / "data",
        help="Directory containing .xlsx inputs (default: ./data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "outputs",
        help="Output directory (default: ./outputs).",
    )
    parser.add_argument(
        "--export-first-sheet",
        action="store_true",
        help="If pandas+openpyxl are installed, export the first sheet of each .xlsx to CSV.",
    )
    parser.add_argument(
        "--export-target-table",
        action="store_true",
        help="Export target table (date,yield_10y) to outputs/targets (requires pandas+openpyxl).",
    )
    parser.add_argument(
        "--export-liquidity-tables",
        action="store_true",
        help="Export liquidity_features.csv and direct_policy_factors.csv from 资金与流动性.xlsx.",
    )
    parser.add_argument(
        "--export-demand-table",
        action="store_true",
        help="Export demand-side net-buy tables (long + wide) from 现券成交分机构统计*.xlsx.",
    )
    parser.add_argument(
        "--export-direct-factors",
        action="store_true",
        help="Export direct_factors.csv (LPR + OMO + net financing proxy; daily + forward fill).",
    )
    parser.add_argument(
        "--export-macro-table",
        action="store_true",
        help="Export macro_features.csv (day+week aligned to daily; forward fill).",
    )
    parser.add_argument(
        "--export-master-dataset",
        action="store_true",
        help="Export processed/master_dataset.csv by left-joining target + liquidity + demand + direct + macro.",
    )
    parser.add_argument(
        "--build-modeling-dataset",
        action="store_true",
        help="Build a modeling-ready dataset from Excel inputs (requires pandas+openpyxl).",
    )
    parser.add_argument(
        "--dataset-stem",
        default="modeling_dataset",
        help="Output dataset name stem under outputs/datasets (default: modeling_dataset).",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=1,
        help="Label horizon in days for yield forecasting (default: 1).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio for time-based split (default: 0.2).",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Do not write train/val split files; only write the full dataset.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2000,
        help="Max rows to export per file when using --export-first-sheet (default: 2000).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir

    if not data_dir.exists():
        logging.error("data-dir does not exist: %s", data_dir)
        return 2

    xlsx_files = list(iter_xlsx_files(data_dir))
    if not xlsx_files:
        logging.warning("No .xlsx files found in %s", data_dir)

    manifest = build_manifest(xlsx_files)
    manifest_path = output_dir / "inputs_manifest.csv"
    write_manifest_csv(manifest, manifest_path)
    logging.info("Wrote manifest: %s", manifest_path)

    if args.export_first_sheet and xlsx_files:
        for xlsx in xlsx_files:
            out_csv = output_dir / "exported_csv" / f"{xlsx.stem}.csv"
            ok, err = try_export_first_sheet_to_csv(xlsx, out_csv, max_rows=args.max_rows)
            if not ok:
                logging.error(
                    "%s. Install with: python3 -m pip install pandas openpyxl",
                    err,
                )
                return 3
            if err is not None:
                logging.error("%s: %s", xlsx, err)
                return 4
            logging.info("Exported: %s", out_csv)

    if args.build_modeling_dataset:
        try:
            from modeling_dataset import build_modeling_dataset, write_dataset_artifacts
        except Exception as exc:
            logging.error(
                "Missing dependency or import error (%s). Install with: python3 -m pip install pandas openpyxl",
                exc,
            )
            return 5

        try:
            df = build_modeling_dataset(data_dir=data_dir, horizon_days=int(args.horizon_days))
        except Exception as exc:
            logging.error("Failed to build modeling dataset: %s", exc)
            return 6

        artifacts = write_dataset_artifacts(
            df=df,
            out_dir=output_dir / "datasets",
            dataset_stem=str(args.dataset_stem),
            val_ratio=float(args.val_ratio),
            do_split=not bool(args.no_split),
            horizon_days=int(args.horizon_days),
        )
        logging.info("Wrote dataset: %s", artifacts.full_csv)
        if artifacts.train_csv and artifacts.val_csv:
            logging.info("Wrote split: %s (train) / %s (val)", artifacts.train_csv, artifacts.val_csv)

    if args.export_target_table:
        try:
            from modeling_dataset import read_target_yield_10y
        except Exception as exc:
            logging.error(
                "Missing dependency or import error (%s). Install with: python3 -m pip install pandas openpyxl",
                exc,
            )
            return 7

        try:
            target = read_target_yield_10y(data_dir=data_dir)
        except Exception as exc:
            logging.error("Failed to export target table: %s", exc)
            return 8

        out_csv = output_dir / "targets" / "yield_10y.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        target.to_csv(out_csv, index=False, encoding="utf-8-sig")
        logging.info("Wrote target table: %s", out_csv)

    if args.export_liquidity_tables:
        try:
            from modeling_dataset import export_funding_tables
        except Exception as exc:
            logging.error(
                "Missing dependency or import error (%s). Install with: python3 -m pip install pandas openpyxl",
                exc,
            )
            return 9

        try:
            artifacts = export_funding_tables(data_dir=data_dir, out_dir=output_dir)
        except Exception as exc:
            logging.error("Failed to export liquidity tables: %s", exc)
            return 10

        logging.info("Wrote liquidity features: %s", artifacts.liquidity_csv)
        logging.info("Wrote policy factors: %s", artifacts.policy_csv)
        logging.info("Wrote split report: %s", artifacts.report_json)

    if args.export_demand_table:
        try:
            from modeling_dataset import export_demand_tables
        except Exception as exc:
            logging.error(
                "Missing dependency or import error (%s). Install with: python3 -m pip install pandas openpyxl",
                exc,
            )
            return 11

        try:
            artifacts = export_demand_tables(data_dir=data_dir, out_dir=output_dir)
        except Exception as exc:
            logging.error("Failed to export demand tables: %s", exc)
            return 12

        logging.info("Wrote demand wide: %s", artifacts.wide_csv)
        logging.info("Wrote demand long: %s", artifacts.long_csv)
        logging.info("Wrote category mapping: %s", artifacts.mapping_csv)
        logging.info("Wrote demand report: %s", artifacts.report_json)

    if args.export_direct_factors:
        try:
            from modeling_dataset import export_direct_factors
        except Exception as exc:
            logging.error(
                "Missing dependency or import error (%s). Install with: python3 -m pip install pandas openpyxl",
                exc,
            )
            return 13

        try:
            artifacts = export_direct_factors(data_dir=data_dir, out_dir=output_dir)
        except Exception as exc:
            logging.error("Failed to export direct factors: %s", exc)
            return 14

        logging.info("Wrote direct factors: %s", artifacts.csv)
        logging.info("Wrote direct factors report: %s", artifacts.report_json)

    if args.export_macro_table:
        try:
            from modeling_dataset import export_macro_features
        except Exception as exc:
            logging.error(
                "Missing dependency or import error (%s). Install with: python3 -m pip install pandas openpyxl",
                exc,
            )
            return 15

        try:
            artifacts = export_macro_features(data_dir=data_dir, out_dir=output_dir)
        except Exception as exc:
            logging.error("Failed to export macro table: %s", exc)
            return 16

        logging.info("Wrote macro features: %s", artifacts.csv)
        logging.info("Wrote macro report: %s", artifacts.report_json)

    if args.export_master_dataset:
        try:
            from modeling_dataset import export_master_dataset
        except Exception as exc:
            logging.error(
                "Missing dependency or import error (%s). Install with: python3 -m pip install pandas openpyxl",
                exc,
            )
            return 17

        try:
            artifacts = export_master_dataset(data_dir=data_dir, out_dir=output_dir)
        except Exception as exc:
            logging.error("Failed to export master dataset: %s", exc)
            return 18

        logging.info("Wrote master dataset: %s", artifacts.csv)
        logging.info("Wrote master report: %s", artifacts.report_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
