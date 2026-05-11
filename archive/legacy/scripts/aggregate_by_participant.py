#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate trial-level features into participant-level features.

Default input:  data/processed/ml_ready_features.csv
Default output: data/processed/ml_ready_participant.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

DEFAULT_IN = "data/processed/ml_ready_features.csv"
DEFAULT_OUT = "data/processed/ml_ready_participant.csv"

# Columns we consider stable per participant (copied via mode)
STABLE_CATS = ["Chronotype", "Gender"]
STABLE_NUMS = ["Age"]  # should be constant; we still take mode

# Identifier and non-feature columns
ID_COLS = ["participant_id"]

# Trial indexing to drop from features
TRIAL_INDEX_COLS = ["Block", "Trial", "global_trial_index"]

# Columns that are explicit targets / directly encode targets at trial-level
LEAKY_COLS = [
    "behav_valence",
    "risky-choice",
    "feedback-condition",
    "forced and free risk trials",
    "ChoiceMade",
    "CorrectChoice",
    "ResponseTime",
    "CurrentScore",
    "good_trial",
    "trigger_numeric",
    "trigger_val",
]

# Task-structure numeric columns that should not inform participant traits
TASK_VALUE_COLS = ["Option1", "Option2", "ActualValue1", "ActualValue2"]

def most_frequent(series: pd.Series):
    if series.isna().all():
        return np.nan
    mode_vals = series.mode(dropna=True)
    return mode_vals.iloc[0] if not mode_vals.empty else np.nan

def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate trial-level to participant-level features.")
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN, help="Input CSV (trial-level).")
    ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT, help="Output CSV (participant-level).")
    ap.add_argument("--agg", default="mean", choices=["mean", "median"],
                    help="Aggregation function for numeric columns.")
    ap.add_argument("--keep-response-summaries", action="store_true",
                    help="Also add participant-level summaries for ResponseTime/CurrentScore (mean,std,count).")
    ap.add_argument("--add-eeg-std", action="store_true",
                    help="Also add per-participant std for EEG features (e.g., *_FRN, *_P300).")
    ap.add_argument("--verbose", action="store_true", help="Print extra info.")
    return ap.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    print(f"Loading {in_path}")
    df = pd.read_csv(in_path)

    if "participant_id" not in df.columns:
        raise SystemExit("participant_id column not found in input.")

    agg_func = "mean" if args.agg == "mean" else "median"

    # Identify numeric vs non-numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    drop_from_features = set(LEAKY_COLS + TRIAL_INDEX_COLS + ID_COLS + TASK_VALUE_COLS)
    drop_from_features.update(STABLE_NUMS)
    # numeric features to aggregate
    agg_feature_cols = [c for c in num_cols if c not in drop_from_features and c not in STABLE_NUMS]

    if args.verbose:
        print(f"- Numeric columns: {len(num_cols)}")
        print(f"- Candidate aggregated features: {len(agg_feature_cols)}")
        print(f"- Dropped (leaky/index/id): {sorted(list(drop_from_features))}")

    g = df.groupby("participant_id", dropna=False)

    # ---- named aggregation for numeric features ----
    named_aggs = {c: (c, agg_func) for c in agg_feature_cols}

    # Detect EEG amplitude columns (e.g., Fz_FRN, Cz_P300, CPz_P300, etc.)
    eeg_cols = [c for c in agg_feature_cols if re.search(r"_(FRN|P300)$", c)]
    if args.add_eeg_std and len(eeg_cols) > 0:
        for c in eeg_cols:
            named_aggs[f"{c}__std"] = (c, "std")

    # Optional summaries for RT/Score
    if args.keep_response_summaries:
        for c in ["ResponseTime", "CurrentScore"]:
            if c in df.columns:
                named_aggs[f"{c}__mean"] = (c, "mean")
                named_aggs[f"{c}__std"]  = (c, "std")
                named_aggs[f"{c}__count"] = (c, "count")

    part_num = g.agg(**named_aggs)

    # ---- behavioural summaries per participant (rates) ----
    beh_summ = []
    for pid, gdf in g:
        row = {"participant_id": pid}
        if "risky-choice" in gdf.columns:
            # risky-choice is numeric 0/1
            row["risky_choice_rate"] = gdf["risky-choice"].mean()
        if "behav_valence" in gdf.columns:
            row["gain_rate"] = (gdf["behav_valence"] == "gain").mean()
            row["loss_rate"] = (gdf["behav_valence"] == "loss").mean()
        if "CorrectChoice" in gdf.columns:
            row["correct_rate"] = (gdf["CorrectChoice"] == 1).mean()
        beh_summ.append(row)
    part_beh = pd.DataFrame(beh_summ).set_index("participant_id")
    # merge behavioural summaries into numeric aggregates
    part_num = part_num.join(part_beh, how="left")

    # ---- stable categorical/meta via mode ----
    keep_cols = [c for c in (STABLE_CATS + STABLE_NUMS) if c in df.columns]
    part_meta = g[keep_cols].agg(most_frequent)

    # Merge and finalize
    out = part_meta.join(part_num, how="left").reset_index()
    # Count features roughly as numeric columns excluding id/targets/stable meta
    non_feature = set(ID_COLS + STABLE_CATS + STABLE_NUMS)
    out["n_features"] = out.drop(columns=[c for c in out.columns if c in non_feature], errors="ignore").select_dtypes(include=[np.number]).shape[1]
    out["feature_nan_ratio"] = out.isna().mean(axis=1)

    print(f"Aggregated to participant-level: {out.shape[0]} participants × {out.shape[1]} columns (including EEG {agg_func} and {'std' if args.add_eeg_std else 'no std'})")

    out.to_csv(out_path, index=False)
    print(f"Wrote CSV → {out_path}")

    # Try Parquet (optional)
    try:
        pq_path = out_path.with_suffix(".parquet")
        out.to_parquet(pq_path, index=False)
        print(f"Wrote Parquet → {pq_path}")
    except Exception as e:
        print(f"(Parquet skipped: {e})")

if __name__ == "__main__":
    main()