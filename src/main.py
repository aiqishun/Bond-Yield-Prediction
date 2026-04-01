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


def try_read_first_sheet_to_csv(
    xlsx_path: Path,
    out_csv_path: Path,
    max_rows: int | None,
) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False

    try:
        df = pd.read_excel(xlsx_path)  # relies on openpyxl for .xlsx
    except Exception as exc:
        logging.error("Failed to read %s: %s", xlsx_path, exc)
        return True

    if max_rows is not None:
        df = df.head(max_rows)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the project pipeline (currently: input discovery + manifest export)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root(),
        help="Directory containing .xlsx inputs (default: repo root).",
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
        any_attempted = False
        for xlsx in xlsx_files:
            out_csv = output_dir / "exported_csv" / f"{xlsx.stem}.csv"
            ok = try_read_first_sheet_to_csv(xlsx, out_csv, max_rows=args.max_rows)
            any_attempted = True
            if ok:
                logging.info("Exported: %s", out_csv)
            else:
                logging.warning(
                    "Skipping Excel export (missing deps). Install with: python -m pip install pandas openpyxl"
                )
                break

        if any_attempted:
            logging.info("Excel export done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

