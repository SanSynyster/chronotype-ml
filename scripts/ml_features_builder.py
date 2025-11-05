#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_features_builder.py

Build a single **ML-ready** table from the merged EEG×behaviour trial-level file by
pivoting (channel × window) mean amplitudes into wide features, while preserving
per-trial behavioural/demographic context.

Input (default):
  data/merged/eeg_behav_trial_aligned.csv  ← preferred (1 row per channel×window per trial)
  data/merged/eeg_behav_trial.csv          ← fallback if aligned file is missing

Output (default):
  data/processed/ml_ready_features.csv
  data/processed/ml_ready_features.parquet
  reports/ml_features/feature_report.json (schema + basic stats)

Example:
  python scripts/ml_features_builder.py
  python scripts/ml_features_builder.py \
      --input data/merged/eeg_behav_trial_aligned.csv \
      --out-csv data/processed/ml_ready_features.csv \
      --channels Fz,FC1,FC2,FCz,Cz,CPz,Pz,POz \
      --windows FRN,P300

Notes:
- We DO NOT rescale/standardize here; keep raw µV features. Scaling is model-specific.
- We keep one row per (participant_id, Block, Trial). `global_trial_index` is retained.
- If a requested (channel,window) pair is missing for a trial, the feature is NaN.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List

import pandas as pd

# ---------------------------
# Utilities
# ---------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)


def parse_list(arg_val: str | None) -> List[str]:
    if not arg_val:
        return []
    return [x.strip() for x in arg_val.split(',') if x.strip()]


# ---------------------------
# Core builder
# ---------------------------

def build_ml_ready(
    df_long: pd.DataFrame,
    channels: List[str],
    windows: List[str],
) -> pd.DataFrame:
    """Pivot long trial table into wide feature matrix.

    Parameters
    ----------
    df_long : DataFrame
        Long table with columns: ['participant_id','Block','Trial','global_trial_index',
        'channel','window','mean_amp', ...behaviour cols...]
    channels : list[str]
        Channels to keep; if empty, infer from data.
    windows : list[str]
        Windows to keep (e.g., FRN, P300); if empty, infer from data.
    """

    # Basic type hygiene
    for col in ["participant_id", "Block", "Trial", "global_trial_index"]:
        if col in df_long.columns:
            df_long[col] = pd.to_numeric(df_long[col], errors="coerce")

    # Subset to requested channels/windows if provided
    if channels:
        df_long = df_long[df_long["channel"].isin(channels)]
    if windows:
        df_long = df_long[df_long["window"].isin(windows)]

    # Identify behavioural/context columns to preserve (one value per trial)
    base_keys = ["participant_id", "Block", "Trial", "global_trial_index"]
    # Known context columns present in the merged file (keep if they exist)
    candidate_context = [
        "good_trial", "trigger_numeric", "trigger_val", "feedback_valence", "behav_valence",
        "Option1", "Option2", "ActualValue1", "ActualValue2", "ChoiceMade", "CorrectChoice",
        "ResponseTime", "CurrentScore", "risky-choice", "feedback-condition",
        "forced and free risk trials", "Age", "Gender", "Chronotype",
    ]
    context_cols = [c for c in candidate_context if c in df_long.columns]

    # Create a tidy context table with one row per trial (first() is safe because we aligned already)
    ctx = (
        df_long[base_keys + context_cols]
        .drop_duplicates(base_keys)  # allows 1 row per trial
        .sort_values(base_keys)
        .reset_index(drop=True)
    )

    # Build feature name like Fz_FRN, FCz_P300, etc.
    df_long = df_long.copy()
    df_long["feature"] = df_long["channel"].astype(str) + "_" + df_long["window"].astype(str)

    # Pivot: one row per trial, columns = feature, values = mean_amp
    feat = (
        df_long.pivot_table(
            index=base_keys,
            columns="feature",
            values="mean_amp",
            aggfunc="mean",
        )
        .sort_index(axis=1)
        .reset_index()
    )

    # Merge features with context
    wide = ctx.merge(feat, on=base_keys, how="left")

    # Add lightweight QC columns
    feature_cols = [c for c in wide.columns if c not in base_keys + context_cols]
    wide["n_features_present"] = wide[feature_cols].notna().sum(axis=1)
    wide["feature_nan_ratio"] = wide[feature_cols].isna().mean(axis=1)

    # Sort for readability
    wide = wide.sort_values(base_keys).reset_index(drop=True)

    return wide


# ---------------------------
# Reporting helpers
# ---------------------------

def summarize_schema(df: pd.DataFrame) -> dict:
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    n_rows, n_cols = df.shape
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "dtypes": dtypes,
    }


def basic_stats(df: pd.DataFrame, feature_prefixes: List[str] | None = None) -> dict:
    if feature_prefixes is None:
        feature_prefixes = []

    out: dict = {}
    # Global stats
    out["global"] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "participants": int(df["participant_id"].nunique()) if "participant_id" in df else None,
    }

    # Feature-level missingness
    feat_cols = [
        c for c in df.columns
        if c not in {"participant_id", "Block", "Trial", "global_trial_index"}
        and not c.endswith("_id")
        and c not in {"Age", "Gender", "Chronotype", "ResponseTime", "CurrentScore",
                      "Option1", "Option2", "ActualValue1", "ActualValue2",
                      "risky-choice", "feedback-condition", "behav_valence",
                      "forced and free risk trials", "CorrectChoice", "ChoiceMade",
                      "good_trial", "trigger_numeric", "trigger_val", "feedback_valence",
                      "n_features_present", "feature_nan_ratio"}
    ]
    miss = df[feat_cols].isna().mean().sort_values(ascending=False)
    out["feature_missing_ratio_top10"] = miss.head(10).to_dict()

    # Per-window summaries (if possible)
    for w in ["FRN", "P300"]:
        w_cols = [c for c in feat_cols if c.endswith("_" + w) or c.split("_")[-1] == w]
        if w_cols:
            desc = df[w_cols].describe().T
            out[f"window_{w}_summary"] = desc[["mean", "std", "min", "max"]].to_dict(orient="index")

    return out


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Build ML-ready wide feature table from EEG×behaviour trials.")
    parser.add_argument("--input", type=str, default="data/merged/eeg_behav_trial_aligned.csv",
                        help="Path to trial-level merged file (aligned preferred). CSV expected.")
    parser.add_argument("--fallback", type=str, default="data/merged/eeg_behav_trial.csv",
                        help="Fallback path if --input is missing.")
    parser.add_argument("--out-csv", type=str, default="data/processed/ml_ready_features.csv",
                        help="Output CSV path.")
    parser.add_argument("--out-parquet", type=str, default="data/processed/ml_ready_features.parquet",
                        help="Output Parquet path.")
    parser.add_argument("--report-json", type=str, default="reports/ml_features/feature_report.json",
                        help="Path to write a small JSON report (schema + stats).")
    parser.add_argument("--channels", type=str, default="Fz,FC1,FC2,FCz,Cz,CPz,Pz,POz",
                        help="Comma-separated list of channels to include. Empty = infer all.")
    parser.add_argument("--windows", type=str, default="FRN,P300",
                        help="Comma-separated list of windows to include. Empty = infer all.")

    args = parser.parse_args()

    # Resolve input
    input_path = Path(args.input)
    if not input_path.exists():
        log(f"⚠️  Input not found: {input_path} → trying fallback {args.fallback}")
        input_path = Path(args.fallback)
        if not input_path.exists():
            raise SystemExit(f"❌ Neither input nor fallback exists: {args.input} | {args.fallback}")

    # IO dirs
    ensure_dirs(args.out_csv, args.out_parquet, args.report_json)

    log(f"✅ 📥 Loading {input_path}")
    df_long = pd.read_csv(input_path)

    # If user passed empty strings, infer from data
    channels = parse_list(args.channels)
    windows = parse_list(args.windows)
    if not channels:
        channels = sorted(df_long["channel"].dropna().unique().tolist())
        log(f"ℹ️  Inferred channels: {channels}")
    if not windows:
        windows = sorted(df_long["window"].dropna().unique().tolist())
        log(f"ℹ️  Inferred windows: {windows}")

    log("✅ 🧠 Building ML-ready wide feature table …")
    wide = build_ml_ready(df_long, channels=channels, windows=windows)

    # Save outputs
    log(f"✅ 💾 Writing CSV → {args.out_csv}")
    wide.to_csv(args.out_csv, index=False)

    log(f"✅ 💾 Writing Parquet → {args.out_parquet}")
    wide.to_parquet(args.out_parquet, index=False)

    # Report JSON
    rep = {
        "input": str(input_path),
        "channels": channels,
        "windows": windows,
        "schema": summarize_schema(wide),
        "stats": basic_stats(wide),
    }
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    log(f"✅ 📝 Wrote report → {args.report_json}")

    log("✅ Done.")


if __name__ == "__main__":
    main()
