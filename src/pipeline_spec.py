from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineSpec:
    spec_version: int
    frozen_on_utc: str
    mainline_horizon_days: int
    stage2_dataset_stem: str
    legacy_dataset_stem_prefix: str
    legacy_allowed_canonical_horizons: frozenset[int]


def load_pipeline_spec(spec_path: Path) -> PipelineSpec:
    obj = json.loads(spec_path.read_text(encoding="utf-8"))
    mainline = obj.get("mainline", {}) if isinstance(obj, dict) else {}
    legacy = obj.get("legacy", {}) if isinstance(obj, dict) else {}

    def req_int(v, name: str) -> int:
        if isinstance(v, int):
            return v
        raise ValueError(f"Invalid pipeline_spec.json field {name}: expected int, got {type(v).__name__}")

    def req_str(v, name: str) -> str:
        if isinstance(v, str) and v:
            return v
        raise ValueError(f"Invalid pipeline_spec.json field {name}: expected non-empty str")

    def opt_int_set(v, name: str) -> frozenset[int]:
        if v is None:
            return frozenset()
        if isinstance(v, list) and all(isinstance(x, int) for x in v):
            return frozenset(v)
        raise ValueError(f"Invalid pipeline_spec.json field {name}: expected list[int]")

    return PipelineSpec(
        spec_version=req_int(obj.get("spec_version"), "spec_version"),
        frozen_on_utc=req_str(obj.get("frozen_on_utc"), "frozen_on_utc"),
        mainline_horizon_days=req_int(mainline.get("horizon_days"), "mainline.horizon_days"),
        stage2_dataset_stem=req_str(mainline.get("stage2_dataset_stem"), "mainline.stage2_dataset_stem"),
        legacy_dataset_stem_prefix=req_str(legacy.get("dataset_stem_prefix"), "legacy.dataset_stem_prefix"),
        legacy_allowed_canonical_horizons=opt_int_set(
            legacy.get("allowed_canonical_horizons"),
            "legacy.allowed_canonical_horizons",
        ),
    )
