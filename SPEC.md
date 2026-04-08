# Pipeline Spec (Frozen)

This repo uses a *frozen* spec to avoid accidentally changing core conventions:

- `horizon`
- `feature role`
- `legacy` naming rules
- `stage1 / stage2` mainline definitions

The spec lives in `pipeline_spec.json`.

## Mainline (Do Not Change Lightly)

- `horizon_days = 20`
- **Stage1 mainline**: component datasets label horizon is `t+20`
- **Stage2 mainline**: final yield dataset label horizon is `t+20`
- Stage2 mainline dataset stem: `modeling_dataset`
- Feature registry mainline stage name: `stage2_final_10y`

## Legacy / Baseline

If Stage2 is rebuilt with a different horizon, the previous dataset is archived as **legacy/baseline**:

- Dataset stem template: `modeling_dataset_legacy_t+{horizon_days}`
- Feature registry stage template: `stage2_final_10y_legacy_t+{horizon_days}`

Legacy datasets are kept for comparison only, not the mainline training target.

Additional rule: legacy baseline is **canonical-only** — at most **one** legacy set per horizon may remain under `outputs/datasets/`. Any duplicate versions produced by spec-drift validation must be moved out (for example to `outputs/experiments/validation/`).

In this repo, the only allowed canonical baseline horizon is `t+1` (see `pipeline_spec.json`).

## Enforcement

By default, `src/run_pipeline.py` enforces `pipeline_spec.json`:

- If you pass `--horizon-days` or `--dataset-stem` that differs from the spec, the command exits with an error.
- To intentionally diverge, rerun with `--allow-spec-drift`.

## Typical commands (Mainline)

- Build Stage2 (t+20): `python src/run_pipeline.py --build-modeling-dataset`
- Export feature registry: `python src/run_pipeline.py --export-feature-registry`
