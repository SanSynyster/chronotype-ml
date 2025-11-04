#!/usr/bin/env python3
"""
Merge per-trial EEG singletrial means with trigger valence (gain/loss),
and optionally align with behavioural trials to add feedback-condition.

- Assumes files like:
    data/raw/_singletrial_means/Epochs_Extracted_1002_singletrial_means.csv
    data/raw/_singletrial_means/Epochs_Extracted_1002_triggers.csv

- Trigger mapping (per your note):
    1 -> gain
    2 -> loss

- Behavioural file (optional):
    data/raw/raw_behavioral_trials.xlsx
  We'll build a global trial index per UserID by sorting by Block then Trial
  and merge on that index for consistency checks.

Outputs:
- Per-participant merged CSVs under: data/merged/singletrial_with_triggers/
- Combined dataset: data/merged/singletrial_all.csv
- Consistency report: data/merged/consistency_report.csv

Usage:
  python scripts/merge_singletrial_triggers.py \
    --means-dir data/raw/_singletrial_means \
    --out-dir data/merged/singletrial_with_triggers \
    --behav-xlsx data/raw/raw_behavioral_trials.xlsx  # optional
"""

from __future__ import annotations
import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# ------------------------ logging (emoji + pretty) ------------------------ #
LOGGER = logging.getLogger("chronotype")

LEVEL_EMOJI = {
    logging.DEBUG: "🐞",
    logging.INFO: "✅",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "💥",
}

class EmojiFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        emoji = LEVEL_EMOJI.get(record.levelno, "")
        record.msg = f"{emoji} {record.msg}"
        return super().format(record)

def setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(EmojiFormatter("%(asctime)s | %(levelname)s | %(message)s"))
    root.addHandler(ch)

# ----------------------------- helpers ------------------------------------ #

PID_RE = re.compile(r"Epochs_Extracted_(\d+)_", re.IGNORECASE)

def extract_pid(path: Path) -> Optional[str]:
    m = PID_RE.search(path.name)
    return m.group(1) if m else None

def load_means_and_triggers(means_path: Path, triggers_path: Path) -> pd.DataFrame:
    df_m = pd.read_csv(means_path)
    df_t = pd.read_csv(triggers_path)

    # Normalise column names
    df_m.columns = [c.strip().lower() for c in df_m.columns]
    df_t.columns = [c.strip().lower() for c in df_t.columns]

    # We expect a join key named 'trial' (your files have it). If not present, try 'epoch'.
    trial_key = None
    if "trial" in df_m.columns:
        trial_key = "trial"
    elif "epoch" in df_m.columns:
        trial_key = "epoch"
    else:
        raise ValueError(f"No 'trial' or 'epoch' column in {means_path}")

    if trial_key not in df_t.columns:
        # Some trigger exports use 'epoch' instead of 'trial'
        if "epoch" in df_t.columns:
            pass  # OK, we will use 'epoch' as the key
        else:
            raise ValueError(f"No '{trial_key}' (or 'epoch') column in {triggers_path}")

    # Detect trigger column: accept common variants or infer by content
    trigger_col = None
    candidates = [
        "trigger", "triggers", "value", "type", "code", "marker",
        "event", "stim", "stimtype", "event_type", "trigger_code",
    ]
    for cand in candidates:
        if cand in df_t.columns:
            trigger_col = cand
            break

    if trigger_col is None:
        # Heuristic: choose a column with only two unique non-null values that look like 1/2 (allow strings)
        for col in df_t.columns:
            if col in {trial_key, "epoch"}:
                continue
            vals = pd.Series(df_t[col].dropna().astype(str).str.extract(r"(\d+)")[0]).dropna()
            uniq = vals.unique()
            if set(uniq).issubset({"1", "2"}) and len(uniq) > 0:
                trigger_col = col
                break

    if trigger_col is None:
        raise ValueError(f"Could not locate a trigger column in {triggers_path}. Columns were: {list(df_t.columns)}")

    # Build a clean numeric trigger series from whatever format (e.g., '1', 'S 2', 1.0)
    trig_numeric = (
        pd.to_numeric(
            df_t[trigger_col].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce",
        )
    )

    # Prepare the trigger frame with the right join key, using an unambiguous trigger column name
    trig_frame = pd.DataFrame({
        trial_key: df_t[trial_key] if trial_key in df_t.columns else df_t["epoch"],
        "trigger_val": trig_numeric,
    })

    # In case of duplicate rows per trial/epoch, keep the first occurrence
    trig_frame = trig_frame.dropna(subset=["trigger_val"]).drop_duplicates(subset=[trial_key], keep="first")

    # Merge with an unambiguous trigger column name to avoid suffixing/collisions
    merged = df_m.merge(trig_frame[[trial_key, "trigger_val"]], left_on=trial_key, right_on=trial_key, how="left")

    LOGGER.debug("Merged means and triggers: %s rows × %s columns", len(merged), len(merged.columns))
    if "trigger_val" not in merged.columns:
        LOGGER.error("Merged frame has no 'trigger_val' column. Columns present: %s", list(merged.columns))

    # Map trigger to feedback valence
    valence_map = {1: "gain", 2: "loss"}
    merged["trigger_numeric"] = pd.to_numeric(merged.get("trigger_val"), errors="coerce")
    merged["feedback_valence"] = merged["trigger_numeric"].map(valence_map)

    # --- Cleanup & normalisation ---
    # Drop legacy textual 'trigger' column if it is entirely empty/whitespace/NA
    if "trigger" in merged.columns:
        _txt = merged["trigger"].astype(str).str.strip()
        if merged["trigger"].isna().all() or (_txt == "").all():
            merged = merged.drop(columns=["trigger"])
            LOGGER.debug("Dropped empty textual 'trigger' column after merge.")

    # Ensure good_trial is integer (0/1) if present
    if "good_trial" in merged.columns:
        merged["good_trial"] = (
            pd.to_numeric(merged["good_trial"], errors="coerce").fillna(0).astype(int)
        )

    # Enforce a clear column order when available
    desired_order = [
        col for col in [
            "participant_id",
            "subject",
            trial_key,             # 'trial' or 'epoch'
            "good_trial",
            "feedback_valence",
            "trigger_numeric",
            "channel",
            "window",
            "win_start_ms",
            "win_end_ms",
            "mean_amp",
            "trigger_val",
        ] if col in merged.columns
    ]
    other_cols = [c for c in merged.columns if c not in desired_order]
    merged = merged[desired_order + other_cols]

    # Small QC
    missing = merged["feedback_valence"].isna().sum()
    if missing:
        sample_vals = df_t.get(trigger_col, pd.Series(dtype=object)).head(5).tolist()
        LOGGER.warning(
            "Some trials have unmapped trigger values (n=%d) in %s (using column '%s', sample=%s)",
            missing, triggers_path.name, trigger_col, sample_vals,
        )

    return merged

def build_behav_index(behav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Behavioural file has columns:
      Block, Trial, ... , UserID, feedback-condition, forced and free risk trials, ...
    We construct a 1..N index per UserID ordered by Block then Trial.
    """
    needed = {"Block", "Trial", "UserID"}
    if not needed.issubset(set(behav_df.columns)):
        raise ValueError(f"Behavioural file missing required columns {needed}")

    behav_df = behav_df.copy()
    # Sort and enumerate per user
    behav_df.sort_values(["UserID", "Block", "Trial"], inplace=True)
    behav_df["global_trial_index"] = behav_df.groupby("UserID").cumcount() + 1

    # Derive a behavioural valence from 'feedback-condition' if present
    if "feedback-condition" in behav_df.columns:
        behav_df["behav_valence"] = (
            behav_df["feedback-condition"].astype(str).str.lower().str.split("-").str[0]
            .replace({"gain": "gain", "loss": "loss"})
        )
    else:
        behav_df["behav_valence"] = pd.NA

    return behav_df

# ------------------------------ main -------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Merge singletrial means with triggers (and behavioural, optional)")
    ap.add_argument("--means-dir", default="data/raw/_singletrial_means", help="Directory with *_singletrial_means.csv and *_triggers.csv")
    ap.add_argument("--out-dir", default="data/merged/singletrial_with_triggers", help="Output directory for per-participant merges")
    ap.add_argument("--behav-xlsx", default=None, help="Path to behavioural XLSX (optional)")
    ap.add_argument("--save-combined", default="data/merged/singletrial_all.csv", help="Path to save combined dataset")
    ap.add_argument("--save-consistency", default="data/merged/consistency_report.csv", help="Path for consistency checks")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    ap.add_argument("--id-start", type=int, default=None, help="First participant ID (inclusive) to attempt, e.g., 1001")
    ap.add_argument("--id-end", type=int, default=None, help="Last participant ID (inclusive) to attempt, e.g., 1056")
    ap.add_argument("--ids", type=str, default=None, help="Comma-separated list of participant IDs to attempt, e.g., 1001,1003,1010")
    ap.add_argument("--simple", action="store_true", help="Run with sensible defaults (IDs 1001-1056, no behavioural merge, default dirs)")
    args = ap.parse_args()

    setup_logging(args.log_level)

    # Simple mode: no flags -> run with baked-in defaults
    if len(sys.argv) == 1 or args.simple:
        LOGGER.info("No arguments provided or --simple set → running in SIMPLE mode (IDs 1001–1056, no behavioural merge).")
        # Force defaults for a one-command run
        args.id_start = 1001
        args.id_end = 1056
        args.ids = None
        args.behav_xlsx = None
        # Keep default dirs unless user supplied different ones
        if not args.means_dir:
            args.means_dir = "data/raw/_singletrial_means"
        if not args.out_dir:
            args.out_dir = "data/merged/singletrial_with_triggers"

    means_dir = Path(args.means_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which files to process
    requested_pids = None
    if args.ids:
        try:
            requested_pids = sorted({int(x.strip()) for x in args.ids.split(',') if x.strip()})
        except ValueError:
            LOGGER.error("Could not parse --ids. Use comma-separated integers, e.g., 1001,1002,1005")
            sys.exit(2)
    elif args.id_start is not None and args.id_end is not None:
        if args.id_end < args.id_start:
            LOGGER.error("--id-end (%s) cannot be smaller than --id-start (%s)", args.id_end, args.id_start)
            sys.exit(2)
        requested_pids = list(range(args.id_start, args.id_end + 1))

    if requested_pids is not None:
        LOGGER.info("Selecting participants by ID (%d requested) in range/list.", len(requested_pids))
        means_files = []
        for pid in requested_pids:
            candidate = means_dir / f"Epochs_Extracted_{pid}_singletrial_means.csv"
            if candidate.exists():
                means_files.append(candidate)
            else:
                LOGGER.warning("Missing means file for participant %s; skipping.", pid)
        if not means_files:
            LOGGER.error("No *_singletrial_means.csv files found for the requested IDs in %s", means_dir)
            sys.exit(2)
    else:
        LOGGER.info("Scanning means files in %s", means_dir)
        means_files = sorted(means_dir.glob("*_singletrial_means.csv"))
        if not means_files:
            LOGGER.error("No *_singletrial_means.csv files found in %s", means_dir)
            sys.exit(2)
    LOGGER.info("Found %d participant file(s) to process.", len(means_files))

    # Load behavioural (optional)
    behav_df = None
    if args.behav_xlsx:
        LOGGER.info("Loading behavioural file: %s", args.behav_xlsx)
        behav_df = pd.read_excel(args.behav_xlsx, sheet_name=0, engine="openpyxl")
        behav_df = build_behav_index(behav_df)
        LOGGER.info("Behavioural rows: %d  users: %d", len(behav_df), behav_df["UserID"].nunique())

    combined: List[pd.DataFrame] = []
    consistency_rows: List[Dict] = []

    for means_path in tqdm(means_files, desc="Merging participants"):
        pid = extract_pid(means_path)
        if not pid:
            LOGGER.warning("Could not infer participant id from %s, skipping.", means_path.name)
            continue

        trig_path = means_dir / f"Epochs_Extracted_{pid}_triggers.csv"
        if not trig_path.exists():
            LOGGER.warning("Missing triggers for %s, skipping.", pid)
            continue

        merged = load_means_and_triggers(means_path, trig_path)
        merged["participant_id"] = int(pid)

        # If behavioural is present, align by global trial index
        if behav_df is not None:
            user_behav = behav_df[behav_df["UserID"] == int(pid)].copy()
            # Check trial counts
            if user_behav.empty:
                LOGGER.warning("No behavioural rows for UserID=%s", pid)
            else:
                # Create a 1..N index in merged (sorted by 'trial' which should reflect epoch order)
                merged = merged.sort_values("trial").reset_index(drop=True)
                merged["global_trial_index"] = range(1, len(merged) + 1)
                merged = merged.merge(
                    user_behav[["global_trial_index","feedback-condition","behav_valence","forced and free risk trials","Block","Trial"]],
                    on="global_trial_index", how="left"
                )

                # Consistency check between trigger valence and behavioural valence (if present)
                if "behav_valence" in merged.columns and merged["behav_valence"].notna().any():
                    agree = (merged["behav_valence"] == merged["feedback_valence"])
                    n_agree = int(agree.sum())
                    n_total = int(merged["behav_valence"].notna().sum())
                    pct = (100.0 * n_agree / n_total) if n_total else 0.0
                    consistency_rows.append({
                        "participant_id": int(pid),
                        "n_trials_merged": len(merged),
                        "n_with_behav": n_total,
                        "n_valence_agree": n_agree,
                        "valence_agreement_pct": round(pct, 2),
                    })
                    LOGGER.info("PID %s: valence agreement %d/%d = %.2f%%", pid, n_agree, n_total, pct)

        # Save per-participant
        out_path = out_dir / f"Epochs_Extracted_{pid}_merged.csv"
        merged.to_csv(out_path, index=False)
        combined.append(merged)

    # Save combined dataset
    if combined:
        all_df = pd.concat(combined, ignore_index=True)
        Path(args.save_combined).parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(args.save_combined, index=False)
        LOGGER.info("Saved combined dataset: %s  (rows=%d, participants=%d)",
                    args.save_combined, len(all_df), all_df["participant_id"].nunique())

    # Save consistency report
    if consistency_rows:
        rep = pd.DataFrame(consistency_rows)
        Path(args.save_consistency).parent.mkdir(parents=True, exist_ok=True)
        rep.to_csv(args.save_consistency, index=False)
        LOGGER.info("Saved consistency report: %s", args.save_consistency)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()