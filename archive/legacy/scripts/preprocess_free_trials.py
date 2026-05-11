#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

SRC = Path("data/merged/eeg_behav_trial_aligned.csv")
DST_FREE = Path("data/merged/eeg_behav_trial_aligned_free.csv")
DST_AGG = Path("data/merged/eeg_behav_trial_aggregated_free.csv")

# Optional EEG selection (set to None to average across all)
EEG_FILTER = {"channel": "CPz", "window": "P300"}   # or: EEG_FILTER = None

def zscore(g):
    std = g.std(ddof=0)
    if std == 0 or np.isnan(std):
        return (g - g.mean())  # all zeros
    return (g - g.mean()) / std

def main():
    df = pd.read_csv(SRC)

    # --- keep only FREE trials (robust to case/NaN) ---
    mask = df["forced and free risk trials"].astype(str).str.lower().eq("free")
    df = df[mask].copy()
    print(f"✅ Kept only free trials: {len(df):,} rows remain.")

    # --- ensure numeric types for value columns ---
    for c in ["Option1", "Option2", "ResponseTime", "mean_amp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- feature engineering at TRIAL level ---
    df["OptionDiff"] = df["Option1"] - df["Option2"]
    df["AbsOptionDiff"] = (df["Option1"] - df["Option2"]).abs()
    df["ValueSum"] = df["Option1"] + df["Option2"]
    df["RT_log"] = np.log(df["ResponseTime"].fillna(0) + 1e-6)
    # per-participant RT z-score
    if "participant_id" in df.columns:
        df["RT_zscore"] = df.groupby("participant_id")["ResponseTime"].transform(zscore)
    else:
        df["RT_zscore"] = np.nan

    # per-participant mean_amp z-score (if column exists)
    if "mean_amp" in df.columns:
        df["mean_amp_z"] = df.groupby("participant_id")["mean_amp"].transform(zscore)

    # --- optional EEG filtering (e.g., CPz/P300) ---
    if EEG_FILTER is not None:
        for k, v in EEG_FILTER.items():
            if k in df.columns:
                df = df[df[k].astype(str) == str(v)]
        print(f"🎚️ EEG filter applied: {EEG_FILTER} -> {len(df):,} rows")

    # --- save the free-trials file (optional but handy) ---
    DST_FREE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DST_FREE, index=False)
    print(f"💾 wrote {DST_FREE}")

    # --- aggregate to 1 row per (participant, Trial) ---
    # If Option/RT columns are constant within a trial, 'first' is fine; mean_amp is averaged.
    group_keys = ["participant_id", "Trial"]
    missing_keys = [k for k in group_keys if k not in df.columns]
    if missing_keys:
        raise KeyError(f"Missing grouping columns: {missing_keys}")

    agg = (
        df.groupby(group_keys, as_index=False)
          .agg({
              "mean_amp": "mean" if "mean_amp" in df.columns else "first",
              "mean_amp_z": "mean" if "mean_amp_z" in df.columns else "first",
              "Option1": "first",
              "Option2": "first",
              "OptionDiff": "first",
              "AbsOptionDiff": "first",
              "ValueSum": "first",
              "ResponseTime": "first",
              "RT_log": "first",
              "RT_zscore": "first",
              "risky-choice": "first",
          })
    )

    # clean up: if columns don’t exist they become all-NaN; drop them
    drop_empty = [c for c in ["mean_amp_z"] if c in agg.columns and agg[c].isna().all()]
    agg = agg.drop(columns=drop_empty)

    DST_AGG.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(DST_AGG, index=False)
    print(f"✅ Saved clean aggregated dataset → {DST_AGG}")
    print(f"📐 Shape: {agg.shape[0]:,} rows × {agg.shape[1]} cols")

if __name__ == "__main__":
    main()