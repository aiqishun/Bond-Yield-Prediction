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


@dataclass(frozen=True)
class FundingSplitArtifacts:
    liquidity_csv: Path
    policy_csv: Path
    report_json: Path


@dataclass(frozen=True)
class DemandArtifacts:
    long_csv: Path
    wide_csv: Path
    mapping_csv: Path
    report_json: Path


@dataclass(frozen=True)
class DirectFactorsArtifacts:
    csv: Path
    report_json: Path


@dataclass(frozen=True)
class MacroArtifacts:
    csv: Path
    report_json: Path


@dataclass(frozen=True)
class MasterArtifacts:
    csv: Path
    report_json: Path


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


def _extract_target_table(pd: object, df, date_col: str, yield_col: str):
    out = df[[date_col, yield_col]].rename(columns={date_col: "date", yield_col: "yield_10y"}).copy()
    out["date"] = _to_date(pd, out["date"])
    out["yield_10y"] = pd.to_numeric(out["yield_10y"], errors="coerce")
    out = out.dropna(subset=["date", "yield_10y"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"])
    return out.reset_index(drop=True)


def _pick_yield_col(columns: list[str]) -> str | None:
    # Heuristics: prefer columns explicitly mentioning 10y yield.
    for c in columns:
        s = str(c)
        if ("收益率" in s or "yield" in s.lower()) and ("10" in s or "10年" in s or "10Y" in s.upper()):
            return s
    for c in columns:
        s = str(c)
        if ("中债" in s or "国债" in s) and ("收益率" in s or "到期收益率" in s):
            return s
    return None


def read_target_yield_10y(data_dir: Path):
    pd, _ = _require_pandas()

    # 1) Preferred: 10年期国债收益率.xlsx
    preferred = data_dir / "10年期国债收益率.xlsx"
    if preferred.exists():
        df = pd.read_excel(preferred, sheet_name=0)
        if not df.empty:
            date_col = "order_date_dt" if "order_date_dt" in df.columns else ("时间" if "时间" in df.columns else df.columns[0])
            yield_col = _pick_yield_col(list(map(str, df.columns)))
            if yield_col is not None and yield_col in df.columns:
                return _extract_target_table(pd, df, date_col=str(date_col), yield_col=yield_col)
            logging.warning(
                "No yield column found in %s (columns=%s). Falling back to 资金与流动性.xlsx.",
                preferred.name,
                list(map(str, df.columns)),
            )

    # 2) Fallback: 资金与流动性.xlsx
    fallback = data_dir / "资金与流动性.xlsx"
    if not fallback.exists():
        raise FileNotFoundError(f"Missing target source files: {preferred} and {fallback}")

    df = pd.read_excel(fallback, sheet_name=0)
    if df.empty:
        raise ValueError(f"Empty Excel: {fallback}")

    date_col = "日期" if "日期" in df.columns else df.columns[0]
    yield_col = None
    for c in df.columns:
        if "10年" in str(c) and "收益率" in str(c):
            yield_col = c
            break
    if yield_col is None:
        yield_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    return _extract_target_table(pd, df, date_col=str(date_col), yield_col=str(yield_col))


def _sanitize_col_name(name: str) -> str:
    # Keep ASCII-friendly names where possible; leave Chinese as-is unless mapped explicitly.
    import re

    s = str(name).strip()
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"[()（）]", "", s)
    s = s.replace(":", "_").replace("：", "_").replace("/", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def export_funding_tables(data_dir: Path, out_dir: Path) -> FundingSplitArtifacts:
    """
    Split `资金与流动性.xlsx` into:
    - liquidity_features.csv: other liquidity indicators (DR/R/CD/spreads/etc.)
    - direct_policy_factors.csv: policy operations (OMO/逆回购/MLF/到期回笼/etc.)

    If the workbook lacks expected columns, we still export the best-effort split
    and write a report describing what's missing.
    """
    pd, _ = _require_pandas()

    def _find_funding_indicator_file() -> Path | None:
        best: tuple[int, Path] | None = None
        for p in sorted(data_dir.glob("*.xlsx")):
            try:
                xl = pd.ExcelFile(p)
            except Exception:
                continue
            # Look across sheets; pick any that has key columns.
            for s in xl.sheet_names:
                try:
                    hdr = pd.read_excel(p, sheet_name=s, nrows=0)
                except Exception:
                    continue
                cols = {str(c) for c in hdr.columns}
                score = 0
                for need in ["DR001", "DR007", "R001", "R007"]:
                    if need in cols:
                        score += 2
                for need in ["中国:公开市场操作:货币净投放", "中国:公开市场操作:货币投放", "中国:公开市场操作:货币回笼"]:
                    if need in cols:
                        score += 1
                if score > 0 and (best is None or score > best[0]):
                    best = (score, p)
        return best[1] if best else None

    # Preferred funding indicator source: contains DR/R and OMO columns.
    funding_indicator_path = _find_funding_indicator_file()

    # Base dates: take from target yield series if available; this is the index we want for modeling.
    base_dates = read_target_yield_10y(data_dir)[["date"]].copy()

    liquidity = base_dates.copy()
    policy = base_dates.copy()
    sources: list[str] = []

    if funding_indicator_path is not None:
        try:
            df = pd.read_excel(funding_indicator_path, sheet_name=0)
        except Exception as exc:
            logging.warning("Failed reading funding indicator file %s: %s", funding_indicator_path, exc)
            df = None
        if df is not None and not df.empty:
            sources.append(str(funding_indicator_path))
            # Choose a date column; prefer R_日期 then 货币日期 then any *_日期.
            date_candidates = [c for c in df.columns if str(c) in {"R_日期", "货币日期"}] + [
                c for c in df.columns if str(c).endswith("日期")
            ]
            date_col = date_candidates[0] if date_candidates else df.columns[0]
            tmp = df.copy()
            tmp["date"] = _to_date(pd, tmp[date_col])
            tmp = tmp.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
            tmp = tmp.drop(columns=[c for c in tmp.columns if str(c).startswith("Unnamed:")], errors="ignore")

            # Liquidity indicators.
            liq_map = {
                "DR001": "dr001",
                "DR007": "dr007",
                "R001": "r001",
                "R007": "r007",
                "FR007": "fr007",
                "GC001:收盘价": "gc001",
                "GC007:收盘价": "gc007",
            }
            liq_cols = [c for c in liq_map.keys() if c in tmp.columns]
            liq = tmp[["date", *liq_cols]].rename(columns=liq_map).copy()
            for c in [c for c in liq.columns if c != "date"]:
                liq[c] = pd.to_numeric(liq[c], errors="coerce")
            liq = base_dates.merge(liq, on="date", how="left")
            # Common spreads.
            if "r007" in liq.columns and "dr007" in liq.columns:
                liq["spread_r007_dr007"] = liq["r007"] - liq["dr007"]
            if "gc007" in liq.columns and "r007" in liq.columns:
                liq["spread_gc007_r007"] = liq["gc007"] - liq["r007"]
            if "fr007" in liq.columns and "r007" in liq.columns:
                liq["spread_fr007_r007"] = liq["fr007"] - liq["r007"]

            liquidity = liq

            # Policy operation factors (OMO style).
            pol_map = {
                "中国:公开市场操作:货币净投放": "omo_net",
                "中国:公开市场操作:货币投放": "omo_inject",
                "中国:公开市场操作:货币回笼": "omo_withdraw",
            }
            pol_cols = [c for c in pol_map.keys() if c in tmp.columns]
            pol = tmp[["date", *pol_cols]].rename(columns=pol_map).copy()
            for c in [c for c in pol.columns if c != "date"]:
                pol[c] = pd.to_numeric(pol[c], errors="coerce")
            policy = base_dates.merge(pol, on="date", how="left")

    # Add LPR series (monthly) if available.
    lpr_path = data_dir / "债券发行与到期20250818国债及政金债.xlsx"
    if lpr_path.exists():
        try:
            lpr = pd.read_excel(lpr_path, sheet_name=0)
        except Exception as exc:
            logging.warning("Failed reading LPR file %s: %s", lpr_path, exc)
            lpr = None
        if lpr is not None and not lpr.empty and "order_date_dt" in lpr.columns:
            sources.append(str(lpr_path))
            tmp = lpr.copy()
            tmp["date"] = _to_date(pd, tmp["order_date_dt"])
            keep = ["date"]
            if "year_1_lpr" in tmp.columns:
                tmp["lpr_1y"] = pd.to_numeric(tmp["year_1_lpr"], errors="coerce")
                keep.append("lpr_1y")
            if "year_5_lpr" in tmp.columns:
                tmp["lpr_5y"] = pd.to_numeric(tmp["year_5_lpr"], errors="coerce")
                keep.append("lpr_5y")
            tmp = tmp[keep].dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
            # Align to base dates and forward-fill.
            aligned = base_dates.merge(tmp, on="date", how="left").sort_values("date")
            for c in [c for c in aligned.columns if c != "date"]:
                aligned[c] = aligned[c].ffill()
            policy = policy.merge(aligned, on="date", how="left")

    out_dir.mkdir(parents=True, exist_ok=True)
    liquidity_csv = out_dir / "liquidity_features.csv"
    policy_csv = out_dir / "direct_policy_factors.csv"
    report_json = out_dir / "funding_split_report.json"

    # Ensure stable column order.
    liquidity = liquidity.sort_values("date")
    policy = policy.sort_values("date")
    liquidity.to_csv(liquidity_csv, index=False, encoding="utf-8-sig")
    policy.to_csv(policy_csv, index=False, encoding="utf-8-sig")

    expected_liquidity = ["dr001", "dr007", "r001", "r007", "fr007", "gc001", "gc007"]
    expected_policy = ["omo_net", "omo_inject", "omo_withdraw", "mlf", "reverse_repo"]
    missing_liquidity = [c for c in expected_liquidity if c not in liquidity.columns]
    missing_policy = [c for c in expected_policy if c not in policy.columns]

    report = {
        "base_dates_source": "read_target_yield_10y()",
        "source_files_used": sources,
        "rows": int(base_dates.shape[0]),
        "liquidity_columns": [c for c in liquidity.columns],
        "policy_columns": [c for c in policy.columns],
        "missing_expected_liquidity": missing_liquidity,
        "missing_expected_policy": missing_policy,
        "note": "Split uses heuristics + explicit mappings; adjust mappings when you confirm true '货币操作类' fields.",
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return FundingSplitArtifacts(liquidity_csv=liquidity_csv, policy_csv=policy_csv, report_json=report_json)


def export_direct_factors(data_dir: Path, out_dir: Path) -> DirectFactorsArtifacts:
    """
    Step 4: build a daily direct-factor table.

    Includes:
    - LPR (low frequency -> daily forward fill)
    - monetary operations (OMO columns -> daily)
    - 国债净融资 (if present; otherwise left as NaN and reported)
    - a proxy net financing series from 高频宏观指标_day (daily forward fill)
    """
    pd, _ = _require_pandas()

    base_dates = read_target_yield_10y(data_dir)[["date"]].copy().sort_values("date").reset_index(drop=True)
    direct = base_dates.copy()
    sources: list[str] = []
    missing: set[str] = set()

    def find_funding_indicator_file() -> Path | None:
        best: tuple[int, Path] | None = None
        for p in sorted(data_dir.glob("*.xlsx")):
            try:
                xl = pd.ExcelFile(p)
            except Exception:
                continue
            for s in xl.sheet_names:
                try:
                    hdr = pd.read_excel(p, sheet_name=s, nrows=0)
                except Exception:
                    continue
                cols = {str(c) for c in hdr.columns}
                score = 0
                for need in ["DR001", "DR007", "R001", "R007"]:
                    if need in cols:
                        score += 2
                for need in ["中国:公开市场操作:货币净投放", "中国:公开市场操作:货币投放", "中国:公开市场操作:货币回笼"]:
                    if need in cols:
                        score += 1
                if score > 0 and (best is None or score > best[0]):
                    best = (score, p)
        return best[1] if best else None

    # Monetary operations (OMO).
    funding_indicator_path = find_funding_indicator_file()
    if funding_indicator_path is not None:
        try:
            df = pd.read_excel(funding_indicator_path, sheet_name=0)
        except Exception as exc:
            logging.warning("Failed reading OMO source %s: %s", funding_indicator_path, exc)
            df = None
        if df is not None and not df.empty:
            sources.append(str(funding_indicator_path))
            date_candidates = [c for c in df.columns if str(c) in {"货币日期", "R_日期"}] + [
                c for c in df.columns if str(c).endswith("日期")
            ]
            date_col = date_candidates[0] if date_candidates else df.columns[0]
            tmp = df.copy()
            tmp["date"] = _to_date(pd, tmp[date_col])
            tmp = tmp.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
            tmp = tmp.drop(columns=[c for c in tmp.columns if str(c).startswith("Unnamed:")], errors="ignore")

            pol_map = {
                "中国:公开市场操作:货币净投放": "omo_net",
                "中国:公开市场操作:货币投放": "omo_inject",
                "中国:公开市场操作:货币回笼": "omo_withdraw",
            }
            pol_cols = [c for c in pol_map.keys() if c in tmp.columns]
            if pol_cols:
                pol = tmp[["date", *pol_cols]].rename(columns=pol_map).copy()
                for c in [c for c in pol.columns if c != "date"]:
                    pol[c] = pd.to_numeric(pol[c], errors="coerce")
                direct = direct.merge(pol, on="date", how="left")
            else:
                missing.update({"omo_net", "omo_inject", "omo_withdraw"})
    else:
        missing.update({"omo_net", "omo_inject", "omo_withdraw"})

    # LPR (monthly) -> daily forward fill.
    lpr_path = data_dir / "债券发行与到期20250818国债及政金债.xlsx"
    if lpr_path.exists():
        try:
            lpr = pd.read_excel(lpr_path, sheet_name=0)
        except Exception as exc:
            logging.warning("Failed reading LPR file %s: %s", lpr_path, exc)
            lpr = None
        if lpr is not None and not lpr.empty and "order_date_dt" in lpr.columns:
            sources.append(str(lpr_path))
            tmp = lpr.copy()
            tmp["date"] = _to_date(pd, tmp["order_date_dt"])
            keep = ["date"]
            if "year_1_lpr" in tmp.columns:
                tmp["lpr_1y"] = pd.to_numeric(tmp["year_1_lpr"], errors="coerce")
                keep.append("lpr_1y")
            else:
                missing.add("lpr_1y")
            if "year_5_lpr" in tmp.columns:
                tmp["lpr_5y"] = pd.to_numeric(tmp["year_5_lpr"], errors="coerce")
                keep.append("lpr_5y")
            else:
                missing.add("lpr_5y")
            tmp = tmp[keep].dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
            aligned = base_dates.merge(tmp, on="date", how="left").sort_values("date")
            for c in [c for c in aligned.columns if c != "date"]:
                aligned[c] = aligned[c].ffill()
            direct = direct.merge(aligned, on="date", how="left")
    else:
        missing.update({"lpr_1y", "lpr_5y"})

    # 国债净融资: best-effort scan.
    gov_found = False
    for p in sorted(data_dir.glob("*.xlsx")):
        try:
            xl = pd.ExcelFile(p)
        except Exception:
            continue
        for s in xl.sheet_names:
            try:
                hdr = pd.read_excel(p, sheet_name=s, nrows=0)
            except Exception:
                continue
            cols = [str(c) for c in hdr.columns]
            gov_cols = [c for c in cols if ("国债" in c) and ("净融资" in c or "净融资额" in c)]
            if not gov_cols:
                continue
            value_col = gov_cols[0]
            try:
                df = pd.read_excel(p, sheet_name=s)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            date_col = "日期" if "日期" in df.columns else ("时间" if "时间" in df.columns else df.columns[0])
            tmp = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "gov_net_financing"}).copy()
            tmp["date"] = _to_date(pd, tmp["date"])
            tmp["gov_net_financing"] = pd.to_numeric(tmp["gov_net_financing"], errors="coerce")
            tmp = tmp.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
            aligned = base_dates.merge(tmp, on="date", how="left").sort_values("date")
            aligned["gov_net_financing"] = aligned["gov_net_financing"].ffill()
            direct = direct.merge(aligned[["date", "gov_net_financing"]], on="date", how="left")
            sources.append(str(p))
            gov_found = True
            break
        if gov_found:
            break

    if not gov_found:
        direct["gov_net_financing"] = pd.NA
        missing.add("gov_net_financing")

    # Proxy: total net financing (daily).
    macro_day = read_macro_day(data_dir)
    proxy_col = "macro_day__净融资额(亿元)"
    if proxy_col in macro_day.columns:
        tmp = macro_day[["date", proxy_col]].rename(columns={proxy_col: "net_financing_total"}).copy()
        tmp = tmp.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
        aligned = base_dates.merge(tmp, on="date", how="left").sort_values("date")
        aligned["net_financing_total"] = aligned["net_financing_total"].ffill()
        direct = direct.merge(aligned[["date", "net_financing_total"]], on="date", how="left")
        sources.append("data/高频宏观指标_day.xlsx (net financing proxy)")
    else:
        missing.add("net_financing_total")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "direct_factors.csv"
    out_report = out_dir / "direct_factors_report.json"

    direct = direct.sort_values("date").reset_index(drop=True)
    direct.to_csv(out_csv, index=False, encoding="utf-8-sig")

    report = {
        "rows": int(direct.shape[0]),
        "columns": [c for c in direct.columns],
        "source_files_used": sources,
        "missing_expected_fields": sorted(missing),
        "note": "Low-frequency series (e.g., LPR) are aligned to daily dates and forward-filled.",
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return DirectFactorsArtifacts(csv=out_csv, report_json=out_report)


def export_macro_features(data_dir: Path, out_dir: Path) -> MacroArtifacts:
    """
    Step 5: build a daily macro feature table.

    - Daily macro: keep as-is, align to daily date frame.
    - Weekly macro: merge to daily date frame then forward-fill.
    """
    pd, _ = _require_pandas()

    base_dates = read_target_yield_10y(data_dir)[["date"]].copy().sort_values("date").reset_index(drop=True)
    macro_day = read_macro_day(data_dir)
    macro_week = read_macro_week(data_dir)

    df = base_dates.merge(macro_day, on="date", how="left").merge(macro_week, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    day_cols = [c for c in df.columns if c.startswith("macro_day__")]
    week_cols = [c for c in df.columns if c.startswith("macro_week__")]
    if day_cols:
        df[day_cols] = df[day_cols].ffill()
    if week_cols:
        df[week_cols] = df[week_cols].ffill()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "macro_features.csv"
    out_report = out_dir / "macro_features_report.json"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    report = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "day_feature_count": int(len(day_cols)),
        "week_feature_count": int(len(week_cols)),
        "columns": [c for c in df.columns],
        "notes": [
            "Weekly macro is merged on date then forward-filled to daily.",
            "Daily macro is aligned to daily date frame and forward-filled for continuity.",
        ],
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return MacroArtifacts(csv=out_csv, report_json=out_report)


def export_master_dataset(data_dir: Path, out_dir: Path) -> MasterArtifacts:
    """
    Step 6: build a master dataset by left-joining intermediate tables on `date`,
    using the target table as the base date frame.
    """
    pd, _ = _require_pandas()

    target = read_target_yield_10y(data_dir).copy()
    funding = export_funding_tables(data_dir=data_dir, out_dir=out_dir)
    demand = export_demand_tables(data_dir=data_dir, out_dir=out_dir)
    direct = export_direct_factors(data_dir=data_dir, out_dir=out_dir)
    macro = export_macro_features(data_dir=data_dir, out_dir=out_dir)

    liquidity = pd.read_csv(funding.liquidity_csv, parse_dates=["date"])
    demand_wide = pd.read_csv(demand.wide_csv, parse_dates=["date"])
    direct_factors = pd.read_csv(direct.csv, parse_dates=["date"])
    macro_features = pd.read_csv(macro.csv, parse_dates=["date"])

    master = target.merge(liquidity, on="date", how="left")
    master = master.merge(demand_wide, on="date", how="left")
    master = master.merge(direct_factors, on="date", how="left")
    master = master.merge(macro_features, on="date", how="left")
    master = master.sort_values("date").reset_index(drop=True)

    demand_cols = [c for c in ["银行", "基金", "保险", "券商", "理财子", "其他"] if c in master.columns]
    omo_cols = [c for c in master.columns if c.startswith("omo_")]
    flow_like_cols = [*demand_cols, *omo_cols]
    for c in ["net_financing_total", "gov_net_financing"]:
        if c in master.columns:
            flow_like_cols.append(c)

    if flow_like_cols:
        master[flow_like_cols] = master[flow_like_cols].fillna(0.0)

    # Forward fill mainly for low-frequency/level-like series (rates/macros/LPR/etc.).
    ffill_cols = [c for c in master.columns if c not in {"date", *set(flow_like_cols)}]
    if ffill_cols:
        master[ffill_cols] = master[ffill_cols].ffill()

    out_processed = out_dir / "processed"
    out_processed.mkdir(parents=True, exist_ok=True)
    out_csv = out_processed / "master_dataset.csv"
    out_report = out_processed / "master_dataset_report.json"
    master.to_csv(out_csv, index=False, encoding="utf-8-sig")

    report = {
        "rows": int(master.shape[0]),
        "cols": int(master.shape[1]),
        "null_counts_top": dict(
            master.isna().sum().sort_values(ascending=False).head(20).astype(int).to_dict()
        ),
        "inputs": {
            "target": "read_target_yield_10y()",
            "liquidity_features": str(funding.liquidity_csv),
            "demand_wide": str(demand.wide_csv),
            "direct_factors": str(direct.csv),
            "macro_features": str(macro.csv),
        },
        "notes": [
            "Base date frame is the target table dates.",
            "Demand/OMO/financing flow-like columns fill missing with 0.",
            "Other columns are forward-filled for continuity.",
        ],
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return MasterArtifacts(csv=out_csv, report_json=out_report)


def export_demand_tables(data_dir: Path, out_dir: Path) -> DemandArtifacts:
    """
    Build demand-side tables from 现券净买入 (by institution) and a mapping table.

    Outputs:
    - demand_long.csv: date,institution_name,category,net_buy
    - demand_wide.csv: date,银行,基金,保险,券商,理财子,其他 (daily sums)
    - institution_category_mapping.csv: institution_name,category
    - demand_build_report.json
    """
    pd, _ = _require_pandas()

    # Mapping file (acts like 机构分类.xlsx in your description)
    mapping_path = data_dir / "现券成交分机构统计20230401 - 20230630.xlsx"
    if not mapping_path.exists():
        raise FileNotFoundError(
            "Missing institution mapping source. Expected: data/现券成交分机构统计20230401 - 20230630.xlsx"
        )

    mapping_raw = pd.read_excel(mapping_path, sheet_name=0)
    if mapping_raw.empty or not {"机构类型", "机构分类"}.issubset(set(mapping_raw.columns)):
        raise ValueError(f"Unexpected mapping file format: {mapping_path}")

    mapping_df = (
        mapping_raw[["机构类型", "机构分类"]]
        .rename(columns={"机构类型": "institution_name", "机构分类": "category"})
        .dropna()
        .drop_duplicates(subset=["institution_name"])
        .reset_index(drop=True)
    )
    # Override: treat "理财" as its own category.
    mapping_df.loc[mapping_df["institution_name"].astype(str).str.contains("理财", na=False), "category"] = "理财子"

    mapping_dict = dict(zip(mapping_df["institution_name"].astype(str), mapping_df["category"].astype(str)))

    def infer_category(institution_name: str) -> str:
        name = str(institution_name)
        if name in mapping_dict:
            return mapping_dict[name]
        if "理财" in name:
            return "理财子"
        if "银行" in name:
            return "银行"
        if "基金" in name:
            return "基金"
        if "保险" in name:
            return "保险"
        if "证券" in name or "券商" in name:
            return "券商"
        return "其他"

    # Collect quarterly "详细" files (institution_type + bond dimensions).
    files = sorted(data_dir.glob("现券成交分机构统计*.xlsx"))
    used_detailed_files: list[str] = []
    frames = []
    category_fallback_path: Path | None = None

    for path in files:
        if path.name == mapping_path.name or path.name == "现券成交分机构统计20230701 - 20230930.xlsx":
            continue

        try:
            xl = pd.ExcelFile(path)
        except Exception:
            continue

        sheet = "数据" if "数据" in xl.sheet_names else 0
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        cols = {str(c) for c in df.columns}
        if {"交易日期", "机构分类", "净买入交易量（亿元）"}.issubset(cols) and "机构类型" not in cols:
            # This looks like an already category-level time series (not raw institutions).
            category_fallback_path = path
            continue

        if "交易日期" not in df.columns or "净买入交易量（亿元）" not in df.columns or "机构类型" not in df.columns:
            continue

        tmp = df[["交易日期", "机构类型", "净买入交易量（亿元）"]].copy()
        tmp = tmp.rename(columns={"交易日期": "date", "机构类型": "institution_name", "净买入交易量（亿元）": "net_buy"})
        tmp["date"] = _to_date(pd, tmp["date"])
        tmp["institution_name"] = tmp["institution_name"].astype(str).str.strip()
        tmp["net_buy"] = pd.to_numeric(
            tmp["net_buy"].replace({"--": pd.NA, "—": pd.NA, "-": pd.NA}),
            errors="coerce",
        ).fillna(0.0)
        tmp = tmp.dropna(subset=["date"])
        frames.append(tmp)
        used_detailed_files.append(str(path))

    if not frames:
        raise ValueError("No usable detailed 现券成交分机构统计*.xlsx files found for demand table.")

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["category"] = all_rows["institution_name"].map(infer_category)

    # Long: by date + institution + category.
    long_df = (
        all_rows.groupby(["date", "institution_name", "category"], as_index=False)[["net_buy"]]
        .sum(numeric_only=True)
        .sort_values(["date", "category", "institution_name"])
        .reset_index(drop=True)
    )

    cat_order = ["银行", "基金", "保险", "券商", "理财子", "其他"]

    # Wide from detailed data.
    detailed_wide = long_df.groupby(["date", "category"], as_index=False)[["net_buy"]].sum(numeric_only=True)
    detailed_wide = detailed_wide.pivot(index="date", columns="category", values="net_buy").reset_index()
    for c in cat_order:
        if c not in detailed_wide.columns:
            detailed_wide[c] = 0.0
    detailed_wide = detailed_wide[["date", *cat_order]].fillna(0.0).sort_values("date").reset_index(drop=True)

    # Optional fallback: category-level file to extend dates beyond detailed coverage.
    fallback_wide = None
    if category_fallback_path is not None:
        df = pd.read_excel(category_fallback_path, sheet_name=0)
        tmp = df[["交易日期", "机构分类", "净买入交易量（亿元）"]].copy()
        tmp = tmp.rename(columns={"交易日期": "date", "机构分类": "category", "净买入交易量（亿元）": "net_buy"})
        tmp["date"] = _to_date(pd, tmp["date"])
        tmp["category"] = tmp["category"].astype(str).str.strip()
        tmp["net_buy"] = pd.to_numeric(
            tmp["net_buy"].replace({"--": pd.NA, "—": pd.NA, "-": pd.NA}),
            errors="coerce",
        ).fillna(0.0)
        tmp = tmp.dropna(subset=["date"])
        tmp = tmp[tmp["category"].isin(["银行", "基金", "保险", "券商", "其他"])]
        fallback_wide = tmp.groupby(["date", "category"], as_index=False)[["net_buy"]].sum(numeric_only=True)
        fallback_wide = fallback_wide.pivot(index="date", columns="category", values="net_buy").reset_index()
        for c in cat_order:
            if c not in fallback_wide.columns:
                fallback_wide[c] = 0.0
        fallback_wide = fallback_wide[["date", *cat_order]].fillna(0.0).sort_values("date").reset_index(drop=True)

    # Prefer detailed values when overlapping.
    wide = detailed_wide
    if fallback_wide is not None:
        d = detailed_wide.set_index("date")
        f = fallback_wide.set_index("date")
        wide = d.combine_first(f)  # prefer detailed (left), fill missing from fallback
        wide = wide.reset_index().sort_values("date").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    long_csv = out_dir / "demand_long.csv"
    wide_csv = out_dir / "demand_wide.csv"
    mapping_csv = out_dir / "institution_category_mapping.csv"
    report_json = out_dir / "demand_build_report.json"

    long_df.to_csv(long_csv, index=False, encoding="utf-8-sig")
    wide.to_csv(wide_csv, index=False, encoding="utf-8-sig")
    mapping_df.sort_values(["category", "institution_name"]).to_csv(mapping_csv, index=False, encoding="utf-8-sig")

    report = {
        "mapping_source": str(mapping_path),
        "used_detailed_files": used_detailed_files,
        "used_category_fallback_file": str(category_fallback_path) if category_fallback_path else None,
        "rows_long": int(long_df.shape[0]),
        "rows_wide": int(wide.shape[0]),
        "categories": cat_order,
        "note": "Category mapping uses mapping file + heuristics; refine as needed.",
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return DemandArtifacts(long_csv=long_csv, wide_csv=wide_csv, mapping_csv=mapping_csv, report_json=report_json)


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

    target = read_target_yield_10y(data_dir)
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
