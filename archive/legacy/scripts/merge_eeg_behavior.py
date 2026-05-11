#!/usr/bin/env python3
"""
Merge long-form EEG single-trial data with behavioural trials, per participant,
using a robust index alignment:

- EEG: data/merged/singletrial_all.csv   (many rows per trial: channel × window)
- BEHAV: data/raw/raw_behavioral_trials.xlsx  (one row per trial)

Strategy:
  For each participant_id (EEG) / UserID (behav):
    1) Behaviour: sort by [Block, Trial], add global_trial_index = 1..N
    2) EEG: sort by [trial] (or [epoch]), add global_trial_index = 1..N
    3) Merge on [participant_id/UserID, global_trial_index]
       → All behavioural columns (Age, Gender, Chronotype, options, choices, RT, risky-choice, feedback-condition, forced/free, etc.) are broadcast to all EEG rows of that trial in the merged outputs.

Outputs:
  - data/merged/eeg_behav_long.csv        (long, all channels/windows)
  - data/merged/eeg_behav_trial.csv       (per-trial collapse: mean per channel×window)
  - reports/eeg_behav_merge/per_participant_merge_report.csv
  - reports/eeg_behav_merge/mismatch_examples.csv
  - logs/eeg_behav_merge.log

Usage (defaults):
  python scripts/merge_eeg_behavior.py

Optional:
  python scripts/merge_eeg_behavior.py \
    --eeg data/merged/singletrial_all.csv \
    --behav data/raw/raw_behavioral_trials.xlsx \
    --out-long data/merged/eeg_behav_long.csv \
    --out-trial data/merged/eeg_behav_trial.csv \
    --report-dir reports/eeg_behav_merge \
    --log-level INFO \
    --min-agreement 0.8 \
    --balance-per-condition 0 \
    --trim-behaviour-to-23 {auto,always,never}  # default: auto
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------- logging with emojis ----------
LOGGER = logging.getLogger("eeg_behav_merge")
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

def setup_logging(level: str = "INFO", logfile: Path | None = None) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric)

    fmt = EmojiFormatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        root.addHandler(fh)

# ---------- helpers ----------
def lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def detect_trial_key(df: pd.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}
    if "trial" in cols: return "trial"
    if "epoch" in cols: return "epoch"
    raise ValueError("No 'trial' or 'epoch' column found in EEG frame")

def build_behav_index(behav: pd.DataFrame) -> pd.DataFrame:
    # Expect Block, Trial, UserID
    needed = {"Block", "Trial", "UserID"}
    if not needed.issubset(set(behav.columns)):
        raise ValueError(f"Behavioural file missing columns {needed}. Found: {list(behav.columns)}")

    b = behav.copy()
    b = b.sort_values(["UserID", "Block", "Trial"]).reset_index(drop=True)
    b["global_trial_index"] = b.groupby("UserID").cumcount() + 1

    # Behavioural valence from feedback-condition, if available
    if "feedback-condition" in b.columns:
        b["behav_valence"] = (
            b["feedback-condition"]
            .astype(str).str.lower().str.split("-").str[0]
            .replace({"gain": "gain", "loss": "loss"})
        )
    else:
        b["behav_valence"] = pd.NA
    return b

def derive_eeg_index(eeg_user: pd.DataFrame, trial_key: str) -> pd.DataFrame:
    """
    Build a per-participant, per-trial running index (1..N) and assign it to ALL rows
    of that trial. This function is careful to avoid length mismatches by mapping
    a trial->index dictionary back onto the full DataFrame.

    Example:
        If trials are [1,1,1,2,2,3,3,3] across rows (due to channels/windows),
        unique trials are [1,2,3] -> indices [1,2,3], and every row with trial==2
        gets global_trial_index==2.
    """
    e = eeg_user.copy()

    # Ensure trial_key is sortable and stable; mergesort keeps equal-key order stable
    e = e.sort_values(trial_key, kind="mergesort").reset_index(drop=True)

    # Coerce trials to integers when possible; fall back to original if coercion fails
    trial_series = pd.to_numeric(e[trial_key], errors="coerce")
    if trial_series.isna().all():
        # If we cannot coerce, just use rank() on the categorical order
        unique_trials = pd.Index(e[trial_key].astype(str)).unique()
        trial_to_idx = {t: i for i, t in enumerate(sorted(unique_trials), start=1)}
        e["global_trial_index"] = e[trial_key].astype(str).map(trial_to_idx).astype(int)
    else:
        # Normal numeric trial codes (possibly not starting at 1, possibly with gaps)
        unique_trials = pd.Index(trial_series.dropna().astype(int)).unique().sort_values()
        trial_to_idx = {int(t): i for i, t in enumerate(unique_trials, start=1)}
        e["global_trial_index"] = trial_series.astype("Int64").map(trial_to_idx).astype(int)

    return e

def modal_trigger_per_trial(eeg_user: pd.DataFrame, trial_key: str) -> pd.DataFrame:
    # Use trigger_numeric if available, else trigger_val
    if "trigger_numeric" in eeg_user.columns:
        trig = pd.to_numeric(eeg_user["trigger_numeric"], errors="coerce")
    else:
        trig = pd.to_numeric(eeg_user.get("trigger_val"), errors="coerce")

    d = pd.DataFrame({
        trial_key: eeg_user[trial_key],
        "trigger_modal": trig
    })
    d = (
        d.groupby(trial_key, as_index=False)["trigger_modal"]
         .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )
    return d

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Merge long-form EEG with behavioural trials via per-participant index alignment.")
    ap.add_argument("--eeg", default="data/merged/singletrial_all.csv")
    ap.add_argument("--behav", default="data/raw/raw_behavioral_trials.xlsx")
    ap.add_argument("--out-long", default="data/merged/eeg_behav_long.csv")
    ap.add_argument("--out-trial", default="data/merged/eeg_behav_trial.csv")
    ap.add_argument("--report-dir", default="reports/eeg_behav_merge")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    ap.add_argument("--min-agreement", type=float, default=0.8,
                    help="Minimum per-participant agreement (0..1) on valence (behaviour vs EEG) to include the participant in the aligned outputs. Set 0 to disable filtering.")
    ap.add_argument("--balance-per-condition", type=int, default=0,
                    help="If >0, downsample per participant to K trials per valence condition (gain/loss) after alignment. 0 disables balancing.")
    ap.add_argument("--trim-behaviour-to-23", dest="trim23", choices=["auto","always","never"], default="auto",
                    help="Trim behaviour to 23 trials per block to match EEG (due to missing post-block fixation). 'auto' trims when 368 vs 384 pattern detected; 'always' forces trim; 'never' disables.")
    args = ap.parse_args()

    setup_logging(args.log_level, logfile=Path("logs/eeg_behav_merge.log"))

    eeg_path = Path(args.eeg)
    behav_path = Path(args.behav)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    out_long_aligned = Path(str(args.out_long).replace(".csv", "_aligned.csv"))
    out_trial_aligned = Path(str(args.out_trial).replace(".csv", "_aligned.csv"))
    out_long_balanced = Path(str(args.out_long).replace(".csv", "_aligned_balanced.csv")) if args.balance_per_condition > 0 else None
    out_trial_balanced = Path(str(args.out_trial).replace(".csv", "_aligned_balanced.csv")) if args.balance_per_condition > 0 else None

    LOGGER.info(f"📥 Loading EEG merged (long) table: {eeg_path}")
    eeg = pd.read_csv(eeg_path)
    eeg = lower_cols(eeg)

    if "participant_id" not in eeg.columns:
        raise SystemExit("EEG file must contain 'participant_id'")

    trial_key = detect_trial_key(eeg)

    LOGGER.info(f"📥 Loading behavioural table: {behav_path}")
    behav = pd.read_excel(behav_path, sheet_name=0, engine="openpyxl")

    # ---- OPTIONAL TRIM: reduce behaviour to 23 trials/block to match EEG (23*16=368) ----
    # Detect canonical 368 (EEG unique trials) vs 384 (behaviour rows) per participant
    try:
        eeg_trial_counts = (eeg.groupby("participant_id")[trial_key]
                              .nunique().rename("eeg_trials"))
        behav_row_counts = (behav.groupby("UserID").size()
                              .rename("behav_rows"))
        counts = pd.concat([eeg_trial_counts, behav_row_counts], axis=1, join="inner").dropna()
    except Exception as e:
        LOGGER.warning(f"Could not compute per-participant counts for auto-trim detection: {e}")
        counts = pd.DataFrame()

    def do_trim_behavior(bdf: pd.DataFrame) -> pd.DataFrame:
        if {"Block", "Trial"}.issubset(bdf.columns):
            before = len(bdf)
            bdf = bdf[bdf["Trial"] <= 23].copy()
            LOGGER.info(f"🧮 Trimmed behaviour: kept Trial<=23 → {len(bdf)} rows (was {before}).")
            return bdf
        else:
            LOGGER.warning("Behaviour table missing Block/Trial columns; cannot trim to 23 per block.")
            return bdf

    auto_detected = False
    if not counts.empty:
        auto_detected = ((counts["eeg_trials"] == 368) & (counts["behav_rows"] == 384)).all()
        if args.trim23 == "auto" and auto_detected:
            LOGGER.info("🧩 Detected canonical 368 (EEG) vs 384 (behav) — enabling per-block trim Trial<=23 (auto).")
            behav = do_trim_behavior(behav)
        elif args.trim23 == "always":
            LOGGER.info("✂️ Forcing behaviour trim to Trial<=23 (per-block).")
            behav = do_trim_behavior(behav)
        elif args.trim23 == "never":
            LOGGER.info("🚫 Behaviour trim disabled (never).")
        else:
            LOGGER.info("ℹ️ Auto-trim not activated; proceeding without trim.")
    else:
        if args.trim23 == "always":
            LOGGER.info("✂️ Forcing behaviour trim to Trial<=23 (per-block).")
            behav = do_trim_behavior(behav)
        else:
            LOGGER.info("ℹ️ Could not evaluate auto-trim; leaving behaviour untrimmed.")

    behav_idx = build_behav_index(behav)

    # Prepare outputs and diagnostics
    merged_long_parts: List[pd.DataFrame] = []
    diag_rows: List[Dict] = []
    mismatch_samples: List[Dict] = []
    aligned_long_parts: List[pd.DataFrame] = []
    balanced_long_parts: List[pd.DataFrame] = []
    per_block_rows: List[Dict] = []

    eeg_pids = sorted(eeg["participant_id"].dropna().astype(int).unique().tolist())
    behav_pids = set(behav_idx["UserID"].dropna().astype(int).unique().tolist())

    LOGGER.info(f"EEG participants: {len(eeg_pids)} | Behaviour participants: {len(behav_pids)}")

    for pid in eeg_pids:
        user_eeg = eeg[eeg["participant_id"] == pid].copy()
        if user_eeg.empty:
            continue

        user_behav = behav_idx[behav_idx["UserID"] == pid].copy()
        if user_behav.empty:
            LOGGER.warning(f"Behaviour missing for participant {pid}; skipping merge (EEG retained separately).")
            diag_rows.append({
                "participant_id": pid,
                "status": "no_behaviour",
                "eeg_rows": len(user_eeg),
                "behav_trials": 0,
                "merged_rows": 0,
                "agreement_pct": np.nan,
            })
            continue

        # Index both sides
        user_eeg = derive_eeg_index(user_eeg, trial_key)

        # --- Sanity check counts before merging ---
        n_eeg_trials = int(user_eeg[trial_key].nunique())
        n_behav_trials = int(user_behav["global_trial_index"].nunique())
        if n_eeg_trials != n_behav_trials:
            LOGGER.warning(
                f"PID {pid}: EEG unique trials={n_eeg_trials} vs Behaviour trials={n_behav_trials}. "
                "Proceeding with index alignment; review reports for potential offsets or rejects."
            )

        # Merge ALL behavioural columns (except UserID) so we keep Age, Gender, Chronotype, options, RT, etc.
        behav_cols_to_merge = [c for c in user_behav.columns if c != "UserID"]
        if "global_trial_index" not in behav_cols_to_merge:
            behav_cols_to_merge = ["global_trial_index"] + behav_cols_to_merge

        merged_user = user_eeg.merge(
            user_behav[behav_cols_to_merge],
            on="global_trial_index",
            how="left"
        )

        # Agreement check (valence vs trigger)
        trig_series = pd.to_numeric(
            merged_user.get("trigger_numeric", merged_user.get("trigger_val")),
            errors="coerce"
        )
        behav_valence = merged_user["behav_valence"].astype("string")
        trig_valence = trig_series.map({1: "gain", 2: "loss"}).astype("string")

        comparable = behav_valence.notna() & trig_valence.notna()
        n_cmp = int(comparable.sum())
        n_agree = int((behav_valence[comparable] == trig_valence[comparable]).sum())
        agree_pct = (100.0 * n_agree / n_cmp) if n_cmp else np.nan

        # Per-block agreement diagnostics (only if Block present)
        if "Block" in merged_user.columns:
            # restrict to comparable rows
            mu_cmp = merged_user[comparable].copy()
            if not mu_cmp.empty:
                grp = mu_cmp.groupby("Block", as_index=False)
                for _, g in grp:
                    trig_series_blk = pd.to_numeric(g.get("trigger_numeric", g.get("trigger_val")), errors="coerce")
                    trig_valence_blk = trig_series_blk.map({1: "gain", 2: "loss"}).astype("string")
                    behav_valence_blk = g["behav_valence"].astype("string")
                    mask = behav_valence_blk.notna() & trig_valence_blk.notna()
                    n_cmp_blk = int(mask.sum())
                    if n_cmp_blk == 0:
                        continue
                    n_agree_blk = int((behav_valence_blk[mask] == trig_valence_blk[mask]).sum())
                    agree_blk = round(100.0 * n_agree_blk / n_cmp_blk, 2)
                    per_block_rows.append({
                        "participant_id": pid,
                        "block": int(g["Block"].iloc[0]),
                        "comparable_trials": n_cmp_blk,
                        "agree_trials": n_agree_blk,
                        "agreement_pct": agree_blk,
                    })

        # Collect a few mismatch examples
        if n_cmp and n_agree < n_cmp:
            mm = merged_user[comparable & (behav_valence != trig_valence)].head(10).copy()
            mm["participant_id"] = pid
            mismatch_samples.append(mm)

        # --- Build aligned-only subset (Option 1) ---
        mask_comparable = comparable
        mask_match = (behav_valence == trig_valence)
        aligned_user = merged_user[mask_comparable & mask_match].copy()

        # Participant-level gate by agreement threshold
        include_by_agreement = True
        if args.min_agreement > 0 and not np.isnan(agree_pct):
            include_by_agreement = (agree_pct / 100.0) >= args.min_agreement
        elif args.min_agreement > 0 and np.isnan(agree_pct):
            include_by_agreement = False  # no comparable trials → exclude if thresholding is on

        if not include_by_agreement:
            LOGGER.warning(f"PID {pid}: agreement {agree_pct:.2f}% < {args.min_agreement*100:.0f}% — excluded from ALIGNED outputs.")
        else:
            aligned_long_parts.append(aligned_user)

            # Optional per-condition balancing (gain/loss)
            if args.balance_per_condition > 0:
                k = args.balance_per_condition
                # Determine available per condition per participant
                # We balance at the trial level using global_trial_index uniqueness
                keep_cols = ["participant_id", "global_trial_index", "behav_valence"]
                trial_keys = aligned_user[keep_cols].drop_duplicates()
                samples = []
                for cond, grp in trial_keys.groupby("behav_valence"):
                    if grp.empty:
                        continue
                    # Sample without replacement up to k trials
                    take = min(len(grp), k)
                    samples.append(grp.sample(n=take, random_state=42))
                if samples:
                    sampled_trials = pd.concat(samples, ignore_index=True)
                    # Join back to full long rows for those sampled trial indices
                    balanced_user = aligned_user.merge(sampled_trials[["participant_id","global_trial_index","behav_valence"]],
                                                       on=["participant_id","global_trial_index","behav_valence"], how="inner")
                    balanced_long_parts.append(balanced_user)

        merged_long_parts.append(merged_user)

        diag_rows.append({
            "participant_id": pid,
            "status": "merged",
            "eeg_rows": len(user_eeg),
            "behav_trials": int(user_behav.shape[0]),
            "merged_rows": int(merged_user.shape[0]),
            "aligned_rows": int(aligned_user.shape[0]) if 'aligned_user' in locals() else 0,
            "comparable_trials": n_cmp,
            "agreement_pct": None if np.isnan(agree_pct) else round(agree_pct, 2),
        })

        LOGGER.info(f"PID {pid}: merged_rows={len(merged_user)} | behav_trials={len(user_behav)} | agreement={agree_pct if not np.isnan(agree_pct) else 'NA'}%")

    if not merged_long_parts:
        LOGGER.error("No participants were merged. Check IDs and inputs.")
        sys.exit(2)

    # Concatenate long-format merged
    merged_long = pd.concat(merged_long_parts, ignore_index=True)
    Path(args.out_long).parent.mkdir(parents=True, exist_ok=True)
    merged_long.to_csv(args.out_long, index=False)
    LOGGER.info(f"💾 Saved long-format merge: {args.out_long} (rows={len(merged_long)})")

    # Save aligned-only long output (Option 1)
    if aligned_long_parts:
        aligned_long = pd.concat(aligned_long_parts, ignore_index=True)
        out_long_aligned.parent.mkdir(parents=True, exist_ok=True)
        aligned_long.to_csv(out_long_aligned, index=False)
        LOGGER.info(f"💾 Saved aligned-only long: {out_long_aligned} (rows={len(aligned_long)})")
    else:
        aligned_long = None
        LOGGER.warning("No aligned participants passed the threshold; aligned outputs not created.")

    # Behaviour/meta columns to carry through in trial-level outputs if present
    BEHAV_META_COLS = [
        "Age", "Gender", "Chronotype", "feedback-condition", "behav_valence",
        "risky-choice", "forced and free risk trials",
        "Option1", "Option2", "ActualValue1", "ActualValue2",
        "ChoiceMade", "CorrectChoice", "ResponseTime", "CurrentScore"
    ]

    # Per-trial collapse (mean over rows per trial × channel × window)
    # You might later customise which features to average.
    numeric_cols = merged_long.select_dtypes(include=[np.number]).columns.tolist()
    group_keys = ["participant_id", "Block", "Trial", "global_trial_index", "channel", "window"]
    keep_keys = [k for k in group_keys if k in merged_long.columns]
    # Extend with available behaviour/meta columns
    keep_keys += [c for c in BEHAV_META_COLS if c in merged_long.columns]
    trial_agg = (
        merged_long.groupby(keep_keys, as_index=False)[numeric_cols]
                   .mean(numeric_only=True)
    )
    Path(args.out_trial).parent.mkdir(parents=True, exist_ok=True)
    trial_agg.to_csv(args.out_trial, index=False)
    LOGGER.info(f"💾 Saved per-trial aggregated: {args.out_trial} (rows={len(trial_agg)})")

    # Per-trial collapse for aligned-only
    if aligned_long is not None:
        numeric_cols_al = aligned_long.select_dtypes(include=[np.number]).columns.tolist()
        keep_keys_al = [k for k in group_keys if k in aligned_long.columns]
        keep_keys_al += [c for c in BEHAV_META_COLS if c in aligned_long.columns]
        trial_agg_al = (
            aligned_long.groupby(keep_keys_al, as_index=False)[numeric_cols_al]
                        .mean(numeric_only=True)
        )
        out_trial_aligned.parent.mkdir(parents=True, exist_ok=True)
        trial_agg_al.to_csv(out_trial_aligned, index=False)
        LOGGER.info(f"💾 Saved aligned per-trial aggregated: {out_trial_aligned} (rows={len(trial_agg_al)})")

    # Optional: balanced (K per condition) outputs
    if args.balance_per_condition > 0 and balanced_long_parts:
        balanced_long = pd.concat(balanced_long_parts, ignore_index=True)
        out_long_balanced.parent.mkdir(parents=True, exist_ok=True)
        balanced_long.to_csv(out_long_balanced, index=False)
        LOGGER.info(f"💾 Saved aligned+balanced long: {out_long_balanced} (rows={len(balanced_long)})")

        numeric_cols_ba = balanced_long.select_dtypes(include=[np.number]).columns.tolist()
        keep_keys_ba = [k for k in group_keys if k in balanced_long.columns]
        trial_agg_ba = (
            balanced_long.groupby(keep_keys_ba, as_index=False)[numeric_cols_ba]
                          .mean(numeric_only=True)
        )
        out_trial_balanced.parent.mkdir(parents=True, exist_ok=True)
        trial_agg_ba.to_csv(out_trial_balanced, index=False)
        LOGGER.info(f"💾 Saved aligned+balanced per-trial aggregated: {out_trial_balanced} (rows={len(trial_agg_ba)})")

    # Reports
    rep = pd.DataFrame(diag_rows)
    rep_path = report_dir / "per_participant_merge_report.csv"
    rep.to_csv(rep_path, index=False)
    LOGGER.info(f"💾 Saved merge report: {rep_path}")

    # Per-block report
    if per_block_rows:
        per_block_df = pd.DataFrame(per_block_rows)
        pb_path = report_dir / "per_block_agreement.csv"
        per_block_df.to_csv(pb_path, index=False)
        LOGGER.info(f"💾 Saved per-block agreement report: {pb_path}")
    else:
        LOGGER.info("No per-block agreement rows to save.")

    if mismatch_samples:
        mmex = pd.concat(mismatch_samples, ignore_index=True)
        mmex_path = report_dir / "mismatch_examples.csv"
        mmex.to_csv(mmex_path, index=False)
        LOGGER.info(f"💾 Saved mismatch examples: {mmex_path} (rows={len(mmex)})")
    else:
        LOGGER.info("No mismatch examples to save.")

    LOGGER.info("✅ Done (original, aligned, and optional balanced outputs written).")

if __name__ == "__main__":
    main()