

#!/usr/bin/env python3
"""
verify_deltas.py

Purpose
-------
Visual and tabular checks to compare RAW ERP amplitudes vs DELTA features,
so we can confirm the direction of FRN (FCz) and P300 (Pz) across conditions
(Gain vs Loss) and chronotype groups (Morning=0, Evening=1).

Inputs
------
- Clean participant-level CSV produced by preprocess script
  (default: data/participant_summary_clean.csv)

Outputs
-------
- reports/verify_deltas/raw_frn_fcz_by_condition.png
- reports/verify_deltas/raw_p300_pz_by_condition.png
- reports/verify_deltas/deltas_vs_raw_summary.csv
- logs/run.log (shared rotating file logger)

CLI
---
python scripts/verify_deltas.py \
  --input_csv data/participant_summary_clean.csv \
  --out_dir reports/verify_deltas \
  --log-level INFO --pretty

Notes
-----
- Uses the same emoji+progress logging style as the main preprocessing script.
- Robust to column name variants (lower/upper, stray spaces).
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------------ Logging (emoji + progress) ------------------------ #
LOGGER = logging.getLogger("chronotype")
PRETTY = True

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

def setup_logging(level: str = "INFO", pretty: bool = True) -> None:
    """Configure console + rotating file logging like the main script."""
    global PRETTY
    PRETTY = bool(pretty)
    numeric = getattr(logging, str(level).upper(), logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    ch.setFormatter(EmojiFormatter(fmt) if PRETTY else logging.Formatter(fmt))

    # File handler (rotate at ~1MB, keep 3). Fallback if RotatingFileHandler missing.
    from logging.handlers import RotatingFileHandler
    os.makedirs("logs", exist_ok=True)
    fh = RotatingFileHandler("logs/run.log", maxBytes=1_000_000, backupCount=3)
    fh.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric)
    root.addHandler(ch)
    root.addHandler(fh)

    # Quiet noisy libs
    for noisy in ("matplotlib", "pandas", "urllib3", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    LOGGER.info("Logging initialized → console + logs/run.log (level=%s, pretty=%s)", level, PRETTY)

# ------------------------------- Utilities --------------------------------- #

def clean_col(c: str) -> str:
    return re.sub(r"\s+", "_", c.strip().lower())

def cols_for_electrode(df: pd.DataFrame, ele: str) -> Dict[str, List[str]]:
    """Return dict of bin name -> list of matching columns for a given electrode.
    Expects cleaned names like: bin1_gain_and_correct_fcz, ..., bin4_loss_and_error_pz
    """
    ele = ele.lower()
    patt = re.compile(rf"^bin([1-4])_.*_{re.escape(ele)}$")
    bins: Dict[str, List[str]] = {f"bin{i}": [] for i in range(1, 5)}
    for c in df.columns:
        m = patt.match(c)
        if m:
            bins[f"bin{m.group(1)}"].append(c)
    LOGGER.debug("Electrode=%s matched columns: %s", ele, bins)
    return bins

def average_bins(df: pd.DataFrame, bin_cols: Dict[str, List[str]], which: Iterable[int]) -> pd.Series:
    """Average across the provided bin indices (e.g., [1,2] for Gain)."""
    chosen_cols: List[str] = []
    for i in which:
        chosen_cols.extend(bin_cols.get(f"bin{i}", []))
    if not chosen_cols:
        raise ValueError("No columns matched for the requested bins.")
    LOGGER.debug("Averaging bins %s over columns: %s", list(which), chosen_cols)
    return df[chosen_cols].mean(axis=1)

# ------------------------------ Core logic --------------------------------- #

def compute_raw_and_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw Gain/Loss means and delta features for FRN@FCz and P300@Pz.
    Returns a new DataFrame with additional columns:
      - frn_gain_fcz, frn_loss_fcz, delta_frn_fcz
      - p300_gain_pz, p300_loss_pz, delta_p300_pz
    Leaves existing columns intact.
    """
    work = df.copy()

    # Ensure chronotype_binary exists (0=M, 1=E)
    if "chronotype_binary" not in work.columns:
        # try to infer from label column
        label_col = next((c for c in work.columns if c.startswith("chronotype")), None)
        if label_col is None:
            raise ValueError("Missing 'chronotype_binary' and no chronotype label found.")
        labels = work[label_col].astype(str).str.upper().str[0].map({"E": 1, "M": 0})
        work["chronotype_binary"] = labels
        LOGGER.warning("chronotype_binary was missing; inferred from %s", label_col)

    # FCz (FRN window proxies): bins 1–4 exist as mean amplitudes per condition
    fcz_bins = cols_for_electrode(work, "fcz")
    pz_bins = cols_for_electrode(work, "pz")

    # Gain = average of bins 1 & 2; Loss = average of bins 3 & 4
    work["frn_gain_fcz"] = average_bins(work, fcz_bins, which=[1, 2])
    work["frn_loss_fcz"] = average_bins(work, fcz_bins, which=[3, 4])
    work["delta_frn_fcz"] = work["frn_loss_fcz"] - work["frn_gain_fcz"]

    work["p300_gain_pz"] = average_bins(work, pz_bins, which=[1, 2])
    work["p300_loss_pz"] = average_bins(work, pz_bins, which=[3, 4])
    work["delta_p300_pz"] = work["p300_gain_pz"] - work["p300_loss_pz"]

    LOGGER.info("➕ Added raw & delta columns for FRN@FCz and P300@Pz")
    return work


def group_summary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Group by chronotype and compute mean/SD/N for selected columns.
    Returns a wide table with columns like: <col>_mean, <col>_sd, <col>_n
    """
    grp = df.groupby("chronotype_binary")[cols]
    agg = grp.agg([("mean", "mean"), ("sd", lambda x: x.std(ddof=1)), ("n", "count")])
    # Flatten MultiIndex columns
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    out = agg.reset_index()
    return out

# --------------------------------- Plots ----------------------------------- #

def barplot_with_error(ax, labels: List[str], means: List[float], sds: List[float], ns: List[int], title: str, ylabel: str):
    x = np.arange(len(labels))
    errs = [s / np.sqrt(n) if n and n > 1 else 0.0 for s, n in zip(sds, ns)]  # SEM
    ax.bar(x, means)
    ax.errorbar(x, means, yerr=errs, fmt='none', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) RAW FRN at FCz: Gain vs Loss by chronotype
    fig, ax = plt.subplots(figsize=(6, 4))
    rows = []
    for g in [0, 1]:
        sub = df[df["chronotype_binary"] == g]
        means = [sub["frn_gain_fcz"].mean(), sub["frn_loss_fcz"].mean()]
        sds = [sub["frn_gain_fcz"].std(ddof=1), sub["frn_loss_fcz"].std(ddof=1)]
        ns = [len(sub), len(sub)]
        barplot_with_error(
            ax,
            labels=[f"M{'' if g==0 else ''} Gain" if g==0 else "E Gain", f"M Loss" if g==0 else "E Loss"],
            means=means,
            sds=sds,
            ns=ns,
            title="RAW FRN (FCz): Gain vs Loss by Chronotype",
            ylabel="Amplitude (µV)",
        )
        rows.append({"group": g, "frn_gain_fcz_mean": means[0], "frn_loss_fcz_mean": means[1]})
    fig.tight_layout()
    p1 = out_dir / "raw_frn_fcz_by_condition.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    LOGGER.info("🖼️ Saved: %s", p1)

    # 2) RAW P300 at Pz: Gain vs Loss by chronotype
    fig, ax = plt.subplots(figsize=(6, 4))
    rows2 = []
    for g in [0, 1]:
        sub = df[df["chronotype_binary"] == g]
        means = [sub["p300_gain_pz"].mean(), sub["p300_loss_pz"].mean()]
        sds = [sub["p300_gain_pz"].std(ddof=1), sub["p300_loss_pz"].std(ddof=1)]
        ns = [len(sub), len(sub)]
        barplot_with_error(
            ax,
            labels=["M Gain" if g==0 else "E Gain", "M Loss" if g==0 else "E Loss"],
            means=means,
            sds=sds,
            ns=ns,
            title="RAW P300 (Pz): Gain vs Loss by Chronotype",
            ylabel="Amplitude (µV)",
        )
        rows2.append({"group": g, "p300_gain_pz_mean": means[0], "p300_loss_pz_mean": means[1]})
    fig.tight_layout()
    p2 = out_dir / "raw_p300_pz_by_condition.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    LOGGER.info("🖼️ Saved: %s", p2)

    # Summaries CSV
    summary = pd.DataFrame(rows).merge(pd.DataFrame(rows2), on="group", how="outer")
    p3 = out_dir / "deltas_vs_raw_summary.csv"
    summary.to_csv(p3, index=False)
    LOGGER.info("📑 Saved: %s", p3)

# --------------------------------- CLI ------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Verify RAW vs DELTA ERP directions by chronotype")
    ap.add_argument("--input_csv", default="data/participant_summary_clean.csv", help="Clean participant CSV")
    ap.add_argument("--out_dir", default="reports/verify_deltas", help="Output directory for plots/CSV")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Logging verbosity")
    ap.add_argument("--pretty", dest="pretty", action="store_true", default=True, help="Enable emojis and progress bars (default)")
    ap.add_argument("--no-pretty", dest="pretty", action="store_false", help="Disable emojis and progress bars")
    args = ap.parse_args()

    setup_logging(args.log_level, args.pretty)

    src = Path(args.input_csv)
    out_dir = Path(args.out_dir)

    if not src.exists():
        LOGGER.error("Input CSV not found: %s", src)
        sys.exit(2)

    LOGGER.info("📥 Loading %s", src)
    df = pd.read_csv(src)

    # Clean column names for robustness
    df.columns = [clean_col(c) for c in df.columns]

    # Compute raw condition means and deltas
    LOGGER.info("🧠 Computing RAW (Gain/Loss) and DELTAs for FRN@FCz and P300@Pz …")
    df2 = compute_raw_and_deltas(df)

    # Quick table: group means for delta columns (sanity check)
    keep = [c for c in ["delta_frn_fcz", "delta_p300_pz"] if c in df2.columns]
    if keep:
        gtab = group_summary(df2, keep)
        out_dir.mkdir(parents=True, exist_ok=True)
        gpath = out_dir / "group_means_deltas.csv"
        gtab.to_csv(gpath, index=False)
        LOGGER.info("📈 Saved: %s", gpath)

    # Plots
    LOGGER.info("🎨 Making verification plots …")
    make_plots(df2, out_dir)

    LOGGER.info("✅ Done. Review figures in %s", out_dir)


if __name__ == "__main__":
    main()