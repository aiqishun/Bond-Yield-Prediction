#!/usr/bin/env python3
from __future__ import annotations

"""
Compatibility entrypoint for SHAP explanations.

The main implementation lives in `src/shap_xgboost.py`.
"""

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    # Import lazily so this wrapper stays lightweight.
    from shap_xgboost import main as _main  # type: ignore

    return _main(argv)


if __name__ == "__main__":  # pragma: no cover
    import sys

    # Ensure `src/` is on sys.path when running as `python src/explain_xgb_shap.py ...`.
    if str(_repo_root() / "src") not in sys.path:
        sys.path.insert(0, str(_repo_root() / "src"))
    raise SystemExit(main(sys.argv[1:]))

