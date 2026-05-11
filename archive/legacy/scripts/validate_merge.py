#!/usr/bin/env python3
"""
Validate the merged single-trial dataset against the raw inputs to catch
any merge mistakes (trial alignment, trigger mapping, coverage).

Simple defaults (no args):
  merged   = data/merged/singletrial_all.csv
  raw_dir  = data/raw/_singletrial_means
  out_dir  = reports/merge_validation

What it checks per participant_id:
  - Presence of raw means and triggers files
  - Trial counts: unique trials in raw MEANS vs raw TRIGGERS vs MERGED
  - Row counts: raw means rows vs merged rows
  - Trigger coverage: how many merged rows have trigger assigned
  - Trigger agreement: raw trigger per trial vs merged trigger per trial
  - Valence distribution (gain/loss) in raw triggers vs merged

Outputs:
  - per_participant_validation.csv  (one row per participant)
  - mismatched_trials.csv          (rows where raw!=merged trigger per trial)
  - missing_triggers.csv           (trials missing trigger in merged but present in raw)
  - coverage_summary.csv           (global sums)
  - logs at logs/validate_merge.log

Assumptions:
  - Raw file names: Epochs_Extracted_<PID>_singletrial_means.csv and
                     Epochs_Extracted_<PID>_triggers.csv in raw_dir
  - Trial key may be named 'trial' or 'epoch'.
  - Trigger column may be various names; we auto-detect and coerce to {1,2}.
  - In merged dataset, triggers live in 'trigger_numeric' or 'trigger_val'.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# ------------------------ logging (emoji + pretty) ------------------------ #
LOGGER = logging.getLogger("validate_merge")
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

# ------------------------------ helpers ---------------------------------- #

TRIGGER_COL_CANDIDATES = [
    "trigger", "triggers", "value", "code", "type", "marker", "event",
    "stim", "stimtype", "event_type", "trigger_code",
]

def lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def detect_trial_key(df: pd.DataFrame) -> str:
    if "trial" in df.columns:
        return "trial"
    if "epoch" in df.columns:
        return "epoch"
    raise ValueError("No trial/epoch column found")

def extract_numeric_trigger(series: pd.Series) -> pd.Series:
    # Allow strings like 'S 1', '1', 1.0 → 1 or 2; else NaN
    s = series.astype(str).str.extract(r"(\d+)")[0]
    return pd.to_numeric(s, errors="coerce")

def load_raw_triggers(trig_path: Path) -> Tuple[pd.DataFrame, str]:
    df_t = lower_cols(pd.read_csv(trig_path))
    trial_key = detect_trial_key(df_t)

    trig_col = None
    for c in TRIGGER_COL_CANDIDATES:
        if c in df_t.columns:
            trig_col = c
            break
    if trig_col is None:
        for c in df_t.columns:
            if c == trial_key:
                continue
            vals = extract_numeric_trigger(df_t[c]).dropna().unique()
            if set(map(int, vals)) <= {1, 2} and len(vals) > 0:
                trig_col = c
                break
    if trig_col is None:
        raise ValueError(f"Could not locate trigger column in {trig_path}")

    df_t["trigger_raw"] = extract_numeric_trigger(df_t[trig_col])
    df_t = df_t[[trial_key, "trigger_raw"]].copy()
    df_t = df_t.dropna(subset=["trigger_raw"]).drop_duplicates(subset=[trial_key], keep="first")
    return df_t, trial_key

def load_raw_means(means_path: Path) -> Tuple[pd.DataFrame, str]:
    df_m = lower_cols(pd.read_csv(means_path))
    trial_key = detect_trial_key(df_m)
    return df_m, trial_key

def derive_merged_trigger(merged_user: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = lower_cols(merged_user)
    trial_key = detect_trial_key(df)

    if "trigger_numeric" in df.columns:
        trig = pd.to_numeric(df["trigger_numeric"], errors="coerce")
    elif "trigger_val" in df.columns:
        trig = pd.to_numeric(df["trigger_val"], errors="coerce")
    else:
        trig = pd.Series(np.nan, index=df.index)

    # One trigger per trial (modal value across channel/window)
    per_trial = (
        pd.DataFrame({trial_key: df[trial_key], "trigger_merged": trig})
        .groupby(trial_key, as_index=False)["trigger_merged"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )
    return per_trial, trial_key

# ------------------------------ main flow -------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Validate merged single-trial dataset against raw means+triggers")
    ap.add_argument("--merged", default="data/merged/singletrial_all.csv")
    ap.add_argument("--raw-dir", default="data/raw/_singletrial_means")
    ap.add_argument("--out-dir", default="reports/merge_validation")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    ap.add_argument("--simple", action="store_true", help="Run with defaults (also used if no args)")
    args = ap.parse_args()

    if len(sys.argv) == 1 or args.simple:
        args.merged = args.merged or "data/merged/singletrial_all.csv"
        args.raw_dir = args.raw_dir or "data/raw/_singletrial_means"
        args.out_dir = args.out_dir or "reports/merge_validation"

    setup_logging(args.log_level, logfile=Path("logs/validate_merge.log"))

    merged_path = Path(args.merged)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"📥 Loading merged dataset: {merged_path}")
    merged = pd.read_csv(merged_path)
    if "participant_id" not in merged.columns:
        raise SystemExit("Merged file must contain 'participant_id'")

    merged_cols_lower = {c.lower() for c in merged.columns}
    if "trial" not in merged_cols_lower and "epoch" not in merged_cols_lower:
        raise SystemExit("Merged file must contain 'trial' or 'epoch'")

    pids = sorted(merged["participant_id"].dropna().astype(int).unique().tolist())
    LOGGER.info(f"Found {len(pids)} participant(s) in merged dataset")

    per_part_rows: List[Dict] = []
    mismatch_rows: List[Dict] = []
    missing_rows: List[Dict] = []

    for pid in pids:
        LOGGER.info(f"🔎 Validating participant {pid}")
        means_path = raw_dir / f"Epochs_Extracted_{pid}_singletrial_means.csv"
        trig_path  = raw_dir / f"Epochs_Extracted_{pid}_triggers.csv"

        merged_user = merged[merged["participant_id"] == pid].copy()
        n_rows_merged = len(merged_user)
        try:
            merged_per_trial, m_trial_key = derive_merged_trigger(merged_user)
        except Exception as e:
            LOGGER.error(f"Could not derive merged trigger for PID {pid}: {e}")
            continue

        # Defaults in case raw files missing
        raw_means_trials = set()
        raw_trig_trials = set()
        raw_gain = raw_loss = 0
        n_rows_means = 0
        n_trials_merged = int(merged_per_trial[m_trial_key].nunique())

        has_means = means_path.exists()
        has_trigs = trig_path.exists()

        if has_means:
            try:
                df_m, mkey_raw = load_raw_means(means_path)
                n_rows_means = len(df_m)
                raw_means_trials = set(pd.to_numeric(df_m[mkey_raw], errors="coerce").dropna().astype(int))
            except Exception as e:
                LOGGER.error(f"Error loading raw means for {pid}: {e}")
        else:
            LOGGER.warning(f"Means file missing for {pid}")

        if has_trigs:
            try:
                df_t, tkey_raw = load_raw_triggers(trig_path)
                raw_trig_trials = set(pd.to_numeric(df_t[tkey_raw], errors="coerce").dropna().astype(int))
                raw_gain = int((df_t["trigger_raw"] == 1).sum())
                raw_loss = int((df_t["trigger_raw"] == 2).sum())
            except Exception as e:
                LOGGER.error(f"Error loading raw triggers for {pid}: {e}")
        else:
            LOGGER.warning(f"Triggers file missing for {pid}")

        # Build raw per-trial trigger map
        raw_map: Dict[int, int] = {}
        if has_trigs:
            raw_map = df_t.set_index(tkey_raw)["trigger_raw"].to_dict()

        # Compare per-trial triggers
        merged_map = merged_per_trial.set_index(m_trial_key)["trigger_merged"].to_dict()
        for trial_id, raw_trig in raw_map.items():
            m_trig = merged_map.get(trial_id, np.nan)
            if np.isnan(m_trig):
                missing_rows.append({
                    "participant_id": pid,
                    "trial": trial_id,
                    "raw_trigger": int(raw_trig),
                    "merged_trigger": np.nan,
                    "issue": "missing_in_merged",
                })
            elif int(raw_trig) != int(m_trig):
                mismatch_rows.append({
                    "participant_id": pid,
                    "trial": trial_id,
                    "raw_trigger": int(raw_trig),
                    "merged_trigger": int(m_trig),
                    "issue": "valence_mismatch",
                })

        # Coverage and distribution in merged
        if "trigger_numeric" in merged_user.columns:
            trig_series = pd.to_numeric(merged_user["trigger_numeric"], errors="coerce")
        else:
            trig_series = pd.to_numeric(merged_user.get("trigger_val"), errors="coerce")
        merged_gain = int((trig_series == 1).sum())
        merged_loss = int((trig_series == 2).sum())
        merged_with_trig = int((~pd.isna(trig_series)).sum())

        n_mismatch = sum(1 for _ in mismatch_rows if _["participant_id"] == pid)
        n_missing  = sum(1 for _ in missing_rows if _["participant_id"] == pid)
        n_compare  = len(raw_map)
        agree_pct  = float((n_compare - n_mismatch) / n_compare * 100.0) if n_compare else np.nan

        per_part_rows.append({
            "participant_id": pid,
            "has_raw_means": has_means,
            "has_raw_triggers": has_trigs,
            "n_rows_raw_means": n_rows_means,
            "n_rows_merged": n_rows_merged,
            "n_trials_raw_means": len(raw_means_trials),
            "n_trials_raw_triggers": len(raw_trig_trials),
            "n_trials_merged": n_trials_merged,
            "raw_gain_trials": raw_gain,
            "raw_loss_trials": raw_loss,
            "merged_gain_rows": merged_gain,
            "merged_loss_rows": merged_loss,
            "n_trials_compared": n_compare,
            "n_trials_mismatch": n_mismatch,
            "n_trials_missing": n_missing,
            "agreement_pct": None if np.isnan(agree_pct) else round(agree_pct, 2),
        })

    # Write outputs
    per_part_df = pd.DataFrame(per_part_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_part_path = out_dir / "per_participant_validation.csv"
    per_part_df.to_csv(per_part_path, index=False)
    LOGGER.info(f"💾 Saved: {per_part_path}")

    if mismatch_rows:
        mm_df = pd.DataFrame(mismatch_rows)
        mm_path = out_dir / "mismatched_trials.csv"
        mm_df.to_csv(mm_path, index=False)
        LOGGER.info(f"💾 Saved: {mm_path} (n={len(mm_df)})")
    else:
        LOGGER.info("No per-trial trigger mismatches detected.")

    if missing_rows:
        ms_df = pd.DataFrame(missing_rows)
        ms_path = out_dir / "missing_triggers.csv"
        ms_df.to_csv(ms_path, index=False)
        LOGGER.info(f"💾 Saved: {ms_path} (n={len(ms_df)})")
    else:
        LOGGER.info("No missing triggers in merged for trials present in raw.")

    if not per_part_df.empty:
        cov = {
            "participants": int(per_part_df.shape[0]),
            "raw_means_present": int(per_part_df["has_raw_means"].sum()),
            "raw_triggers_present": int(per_part_df["has_raw_triggers"].sum()),
            "total_trials_compared": int(per_part_df["n_trials_compared"].sum()),
            "total_mismatches": int(per_part_df["n_trials_mismatch"].sum()),
            "total_missing": int(per_part_df["n_trials_missing"].sum()),
            "median_agreement_pct": float(per_part_df["agreement_pct"].dropna().median())
                                   if per_part_df["agreement_pct"].notna().any() else np.nan,
        }
        cov_path = out_dir / "coverage_summary.csv"
        pd.DataFrame([cov]).to_csv(cov_path, index=False)
        LOGGER.info(f"💾 Saved: {cov_path}")

    LOGGER.info("✅ Done.")

if __name__ == "__main__":
    main()