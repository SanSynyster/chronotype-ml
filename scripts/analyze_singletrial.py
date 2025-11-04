#!/usr/bin/env python3
"""
Analyze the combined single-trial EEG table and generate a compact QC pack
(summary tables + sanity plots) so we can monitor data health as it grows.

Defaults (no args):
  input  = data/merged/singletrial_all.csv
  outdir = reports/singletrial_qc

Outputs include:
  - schema_summary.csv              (columns, dtype, non-null, nunique)
  - missingness.csv                 (per-column NA % for key columns)
  - counts_by_participant.csv       (trial counts per participant)
  - counts_by_channel.csv           (rows per channel)
  - counts_by_window.csv            (rows per ERP window)
  - counts_by_valence.csv           (rows per trigger/valence)
  - participant_valence_pivot.csv   (participant × valence counts)
  - outliers_mean_amp.csv           (|z|>3 within (channel, window))
  - hist_mean_amp_FRN_FCz.png
  - hist_mean_amp_P300_Pz.png
  - box_FRN_FCz_by_valence.png
  - box_P300_Pz_by_valence.png
  - optional: valence_agreement.csv (if behavioural columns present)

Notes:
  * We assume the merged file from `merge_singletrial_triggers.py`.
  * "Valence" derived from `trigger_numeric` (1=gain, 2=loss) or
    `feedback_valence` if present.
  * Plots use matplotlib only (no seaborn), single-figure each, no styles.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ logging (emoji + pretty) ------------------------ #
LOGGER = logging.getLogger("singletrial")
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

# ------------------------------- utils ----------------------------------- #
KEY_COLS = [
    "participant_id", "subject", "trial", "epoch", "channel", "window",
    "win_start_ms", "win_end_ms", "mean_amp", "trigger_val", "trigger_numeric",
    "feedback_valence", "good_trial", "global_trial_index",
]

def read_input(path: Path) -> pd.DataFrame:
    LOGGER.info(f"📥 Loading {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

# ---------------------------- QC functions -------------------------------- #
def schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "null": int(s.isna().sum()),
            "nunique": int(s.nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values(["null", "nunique"], ascending=[False, True])

def derive_valence(df: pd.DataFrame) -> pd.Series:
    if "feedback_valence" in df.columns and df["feedback_valence"].notna().any():
        return df["feedback_valence"].astype(str).str.lower()
    if "trigger_numeric" in df.columns:
        return df["trigger_numeric"].map({1: "gain", 2: "loss"}).astype("object")
    if "trigger_val" in df.columns:
        return pd.to_numeric(df["trigger_val"], errors="coerce").map({1: "gain", 2: "loss"}).astype("object")
    return pd.Series(pd.NA, index=df.index, dtype="object")

def counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame({"value": [], "n": []})
    vc = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="n")
    return vc

def pivot_participant_valence(df: pd.DataFrame, valence: pd.Series) -> pd.DataFrame:
    tmp = pd.DataFrame({"participant_id": df.get("participant_id"), "valence": valence})
    pv = tmp.pivot_table(index="participant_id", columns="valence", aggfunc="size", fill_value=0)
    return pv.reset_index()

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score mean_amp within (channel, window); flag |z|>3."""
    if "mean_amp" not in df.columns or "channel" not in df.columns or "window" not in df.columns:
        return pd.DataFrame()
    grp = df[["participant_id", "trial", "channel", "window", "mean_amp"]].copy()
    # groupwise z; guard std=0
    def _z(x: pd.Series) -> pd.Series:
        mu = x.mean()
        sd = x.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return (x - mu)
        return (x - mu) / sd
    grp["z"] = grp.groupby(["channel", "window"])['mean_amp'].transform(_z)
    out = grp[np.abs(grp["z"]) > 3].copy()
    out.sort_values(["channel", "window", "z"], ascending=[True, True, False], inplace=True)
    return out

# ------------------------------ plotting ---------------------------------- #
def save_hist(series: pd.Series, title: str, path: Path) -> None:
    plt.figure()
    plt.hist(series.dropna(), bins=30)
    plt.title(title)
    plt.xlabel("mean_amp")
    plt.ylabel("count")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def save_box_by_valence(df: pd.DataFrame, title: str, path: Path) -> None:
    if "valence" not in df.columns or "mean_amp" not in df.columns:
        return
    data = [df.loc[df["valence"] == lab, "mean_amp"].dropna().values for lab in ["gain", "loss"]]
    plt.figure()
    plt.boxplot(data, tick_labels=["gain", "loss"])  # matplotlib 3.9 name
    plt.title(title)
    plt.ylabel("mean_amp")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# ------------------------------- main ------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Analyze merged single-trial EEG dataset and produce QC reports")
    ap.add_argument("--input", default="data/merged/singletrial_all.csv")
    ap.add_argument("--out-dir", default="reports/singletrial_qc")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    ap.add_argument("--simple", action="store_true", help="Run with defaults (also used if no args)")
    args = ap.parse_args()

    # Simple-by-default
    if len(sys.argv) == 1 or args.simple:
        args.input = args.input or "data/merged/singletrial_all.csv"
        args.out_dir = args.out_dir or "reports/singletrial_qc"

    setup_logging(args.log_level, logfile=Path("logs/singletrial_analyze.log"))

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_input(in_path)
    LOGGER.info("Data shape: %s rows × %s cols", len(df), len(df.columns))

    # Schema summary
    sch = schema_summary(df)
    sch.to_csv(out_dir / "schema_summary.csv", index=False)
    LOGGER.info("💾 Saved: %s", out_dir / "schema_summary.csv")

    # Missingness (only for key columns to avoid huge CSVs)
    miss_rows = []
    for c in KEY_COLS:
        if c in df.columns:
            miss = float(df[c].isna().mean()) * 100.0
            miss_rows.append({"column": c, "na_pct": round(miss, 3)})
    pd.DataFrame(miss_rows).to_csv(out_dir / "missingness.csv", index=False)
    LOGGER.info("💾 Saved: %s", out_dir / "missingness.csv")

    # Derived valence
    valence = derive_valence(df)

    # Counts
    c_part = counts(df, "participant_id");       c_part.to_csv(out_dir / "counts_by_participant.csv", index=False)
    c_chan = counts(df, "channel");             c_chan.to_csv(out_dir / "counts_by_channel.csv", index=False)
    c_win  = counts(df, "window");              c_win.to_csv(out_dir / "counts_by_window.csv", index=False)
    c_val  = counts(pd.DataFrame({"valence": valence}), "valence"); c_val.to_csv(out_dir / "counts_by_valence.csv", index=False)
    LOGGER.info("💾 Saved counts CSVs")

    # Participant × valence pivot
    pv = pivot_participant_valence(df, valence)
    pv.to_csv(out_dir / "participant_valence_pivot.csv", index=False)
    LOGGER.info("💾 Saved: %s", out_dir / "participant_valence_pivot.csv")

    # FRN@FCz and P300@Pz subsets (if present)
    if set(["window", "channel", "mean_amp"]).issubset(df.columns):
        frn_fcz = df[(df["window"].astype(str).str.upper()=="FRN") & (df["channel"].astype(str).str.upper()=="FCZ")].copy()
        p300_pz = df[(df["window"].astype(str).str.upper()=="P300") & (df["channel"].astype(str).str.upper()=="PZ")].copy()

        if not frn_fcz.empty:
            frn_fcz = frn_fcz.assign(valence=valence.loc[frn_fcz.index])
            save_hist(frn_fcz["mean_amp"], "FRN @ FCz — mean_amp", out_dir / "hist_mean_amp_FRN_FCz.png")
            save_box_by_valence(frn_fcz, "FRN @ FCz by valence", out_dir / "box_FRN_FCz_by_valence.png")
            LOGGER.info("🖼️ Saved FRN@FCz plots")
        else:
            LOGGER.warning("No FRN@FCz rows found for plotting.")

        if not p300_pz.empty:
            p300_pz = p300_pz.assign(valence=valence.loc[p300_pz.index])
            save_hist(p300_pz["mean_amp"], "P300 @ Pz — mean_amp", out_dir / "hist_mean_amp_P300_Pz.png")
            save_box_by_valence(p300_pz, "P300 @ Pz by valence", out_dir / "box_P300_Pz_by_valence.png")
            LOGGER.info("🖼️ Saved P300@Pz plots")
        else:
            LOGGER.warning("No P300@Pz rows found for plotting.")

    # Outliers
    out = detect_outliers(df)
    if not out.empty:
        out.to_csv(out_dir / "outliers_mean_amp.csv", index=False)
        LOGGER.info("💾 Saved: %s (n=%d)", out_dir / "outliers_mean_amp.csv", len(out))
    else:
        LOGGER.info("No |z|>3 outliers for mean_amp within (channel, window).")

    # Behavioural agreement (optional)
    if "behav_valence" in df.columns:
        agree_mask = df["behav_valence"].notna() & valence.notna()
        if agree_mask.any():
            agree = (df.loc[agree_mask, "behav_valence"].astype(str).str.lower() == valence.loc[agree_mask])
            tab = pd.DataFrame({
                "n_with_behav": [int(agree_mask.sum())],
                "n_agree": [int(agree.sum())],
                "pct": [round(float(agree.mean())*100.0, 2)],
            })
            tab.to_csv(out_dir / "valence_agreement.csv", index=False)
            LOGGER.info("💾 Saved: %s", out_dir / "valence_agreement.csv")
        else:
            LOGGER.info("Behavioural columns present but no overlapping non-null rows for agreement.")

    LOGGER.info("✅ Done.")

if __name__ == "__main__":
    main()