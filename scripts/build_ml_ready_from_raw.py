#!/usr/bin/env python3
"""Build an ML-ready trial table directly from local raw inputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PID_RE = re.compile(r"Epochs_Extracted_(\d+)_singletrial_means\.csv$", re.IGNORECASE)


def normalize_chronotype(value: object) -> str | float:
    label = str(value).strip().lower()
    if label in {"e", "evening", "eveningness"}:
        return "Evening"
    if label in {"m", "morning", "morningness"}:
        return "Morning"
    return np.nan


def trigger_column(df: pd.DataFrame) -> str:
    candidates = ["trigger", "triggers", "value", "type", "code", "marker", "event", "stim", "stimtype", "event_type", "trigger_code"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        vals = df[col].astype(str).str.extract(r"(\d+)")[0].dropna().unique()
        if len(vals) and set(vals).issubset({"1", "2"}):
            return col
    raise ValueError(f"Could not identify trigger column in columns {list(df.columns)}")


def load_eeg_participant(means_path: Path) -> pd.DataFrame:
    match = PID_RE.search(means_path.name)
    if not match:
        raise ValueError(f"Unexpected means filename: {means_path.name}")
    participant_id = int(match.group(1))
    trigger_path = means_path.with_name(means_path.name.replace("_singletrial_means.csv", "_triggers.csv"))
    if not trigger_path.exists():
        raise FileNotFoundError(trigger_path)

    means = pd.read_csv(means_path)
    means.columns = [c.strip().lower() for c in means.columns]
    triggers = pd.read_csv(trigger_path)
    triggers.columns = [c.strip().lower() for c in triggers.columns]
    trial_key = "trial" if "trial" in means.columns else "epoch"
    trigger_trial_key = trial_key if trial_key in triggers.columns else "epoch"
    trig_col = trigger_column(triggers.drop(columns=[trigger_trial_key], errors="ignore"))
    trigger_values = pd.to_numeric(triggers[trig_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    trigger_frame = pd.DataFrame({
        trial_key: pd.to_numeric(triggers[trigger_trial_key], errors="coerce"),
        "trigger_numeric": trigger_values,
    }).dropna(subset=[trial_key]).drop_duplicates(subset=[trial_key], keep="first")

    means[trial_key] = pd.to_numeric(means[trial_key], errors="coerce")
    merged = means.merge(trigger_frame, on=trial_key, how="left")
    merged["participant_id"] = participant_id
    merged["global_trial_index"] = merged[trial_key].astype("Int64")
    merged["feature"] = merged["channel"].astype(str).str.strip() + "_" + merged["window"].astype(str).str.strip()
    pivot = (
        merged.pivot_table(
            index=["participant_id", "global_trial_index"],
            columns="feature",
            values="mean_amp",
            aggfunc="mean",
        )
        .reset_index()
    )
    pivot.columns.name = None
    trigger_by_trial = merged.groupby(["participant_id", "global_trial_index"], as_index=False)["trigger_numeric"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )
    good_by_trial = merged.groupby(["participant_id", "global_trial_index"], as_index=False).agg(
        eeg_rows=("mean_amp", "size"),
        good_trial=("good_trial", "min") if "good_trial" in merged.columns else ("mean_amp", "size"),
    )
    out = pivot.merge(trigger_by_trial, on=["participant_id", "global_trial_index"], how="left")
    out = out.merge(good_by_trial, on=["participant_id", "global_trial_index"], how="left")
    return out


def build_behavior(behavior_path: Path, metadata_path: Path) -> pd.DataFrame:
    behavior = pd.read_excel(behavior_path)
    behavior = behavior[behavior["Trial"] <= 23].copy()
    behavior = behavior.sort_values(["UserID", "Block", "Trial"]).reset_index(drop=True)
    behavior["global_trial_index"] = behavior.groupby("UserID").cumcount() + 1
    behavior = behavior.rename(columns={"UserID": "participant_id", "Chronotype": "Chronotype_behavior"})
    behavior["Chronotype_behavior"] = behavior["Chronotype_behavior"].map(normalize_chronotype)

    metadata = pd.read_csv(metadata_path)
    metadata = metadata.rename(columns={"UserID": "participant_id"})
    keep = [
        "participant_id",
        "primary_chronotype",
        "behavior_chronotype",
        "summary_chronotype",
        "chronotype_conflict_behavior",
        "MEQ_MCTQ_status",
    ]
    behavior = behavior.merge(metadata[[c for c in keep if c in metadata.columns]], on="participant_id", how="left")
    behavior["Chronotype"] = behavior["primary_chronotype"].map(normalize_chronotype)
    return behavior


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML-ready trial table directly from raw local files.")
    parser.add_argument("--behavior", default="data/raw/raw_behavioral_trials.xlsx")
    parser.add_argument("--metadata", default="data/processed/participant_metadata.csv")
    parser.add_argument("--means-dir", default="data/raw/_singletrial_means")
    parser.add_argument("--out", default="data/processed/ml_ready_features.csv")
    parser.add_argument("--qc", default="reports/clean/raw_build/ml_ready_qc.json")
    args = parser.parse_args()

    behavior = build_behavior(Path(args.behavior), Path(args.metadata))
    eeg_parts = []
    for means_path in sorted(Path(args.means_dir).glob("*_singletrial_means.csv")):
        eeg_parts.append(load_eeg_participant(means_path))
    if not eeg_parts:
        raise FileNotFoundError(f"No *_singletrial_means.csv files found in {args.means_dir}")
    eeg = pd.concat(eeg_parts, ignore_index=True)
    merged = behavior.merge(eeg, on=["participant_id", "global_trial_index"], how="left")
    trig_valence = pd.to_numeric(merged["trigger_numeric"], errors="coerce").map({1: "gain", 2: "loss"})
    merged["behav_valence"] = merged["feedback-condition"].astype(str).str.lower().str.split("-").str[0]
    comparable = merged["behav_valence"].notna() & trig_valence.notna()
    merged["trigger_behavior_match"] = np.where(comparable, merged["behav_valence"].eq(trig_valence), np.nan)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    by_pid = merged.groupby("participant_id").agg(
        rows=("global_trial_index", "size"),
        eeg_rows_present=("eeg_rows", lambda s: int(s.notna().sum())),
        trigger_behavior_agreement=("trigger_behavior_match", "mean"),
        primary_chronotype=("Chronotype", "first"),
        behavior_chronotype=("Chronotype_behavior", "first"),
    ).reset_index()
    qc = {
        "output": args.out,
        "rows": int(merged.shape[0]),
        "columns": int(merged.shape[1]),
        "participants": int(merged["participant_id"].nunique()),
        "chronotype_counts_primary": merged.drop_duplicates("participant_id")["Chronotype"].value_counts(dropna=False).to_dict(),
        "trigger_behavior_agreement_mean": float(merged["trigger_behavior_match"].mean()),
        "participants_with_incomplete_eeg": by_pid.loc[by_pid["eeg_rows_present"].lt(by_pid["rows"]), ["participant_id", "rows", "eeg_rows_present"]].to_dict("records"),
        "low_trigger_agreement": by_pid.loc[by_pid["trigger_behavior_agreement"].lt(0.95), ["participant_id", "trigger_behavior_agreement"]].to_dict("records"),
        "chronotype_conflicts_behavior": by_pid.loc[by_pid["primary_chronotype"].ne(by_pid["behavior_chronotype"]), ["participant_id", "primary_chronotype", "behavior_chronotype"]].to_dict("records"),
    }
    qc_path = Path(args.qc)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    by_pid.to_csv(qc_path.with_name("ml_ready_by_participant.csv"), index=False)
    print(f"Wrote {out_path} with shape {merged.shape}")
    print(f"Wrote {qc_path}")


if __name__ == "__main__":
    main()
