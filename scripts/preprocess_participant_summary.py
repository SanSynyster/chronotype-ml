# scripts/preprocess_participant_summary.py
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import logging
import sys
from tqdm import tqdm

LOGGER = logging.getLogger("chronotype")

PRETTY = False

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

def setup_logging(level: str = "INFO", pretty: bool = False) -> None:
    import os
    from logging.handlers import RotatingFileHandler
    global PRETTY
    PRETTY = bool(pretty)
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    if PRETTY:
        formatter = EmojiFormatter(fmt)
    else:
        formatter = logging.Formatter(fmt)

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / "run.log"

    # Stream handler (console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # Rotating file handler
    file_handler = RotatingFileHandler(str(logfile), maxBytes=1_048_576, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    # Reduce log noise from external libraries
    for noisy in ["pandas", "matplotlib", "urllib3", "numexpr"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    LOGGER.info("Logging initialized → console + logs/run.log (level=%s, pretty=%s)", level, PRETTY)

# ---------- Utilities ----------
def clean_col(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()

def inspect_file(path: Path) -> None:
    if not path.exists():
        LOGGER.error(f"Not found: {path}")
        return
    xls = pd.ExcelFile(path)
    LOGGER.info(f"📄 === {path} ===")
    LOGGER.info(f"Sheets: {xls.sheet_names}")
    sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    LOGGER.info(f"Sheet: {sheet}")
    LOGGER.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    LOGGER.info("Columns:")
    for i, c in enumerate(df.columns, 1):
        LOGGER.info(f"  {i:2d}. {c}")
    LOGGER.info("Preview (first 5 rows):")
    LOGGER.info("\n%s", df.head(5).to_string(index=False))

def load_summary_xlsx(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    LOGGER.info("📥 Loaded %s (sheet=%s) with shape %s", path, sheet, df.shape)
    LOGGER.debug("Original columns: %s", list(df.columns))
    df.columns = [clean_col(c) for c in df.columns]
    LOGGER.debug("Cleaned columns: %s", list(df.columns))
    return df

def drop_known_noise(df: pd.DataFrame) -> pd.DataFrame:
    removed = []
    for col in ["erpset", "erpset_1", "erp_set", "erp_set_1", "unnamed_13", "unnamed_28", "unnamed_30"]:
        if col in df.columns:
            removed.append(col)
            df = df.drop(columns=[col])
    if removed:
        LOGGER.info("🧹 Dropped noise columns: %s", removed)
    else:
        LOGGER.debug("No noise columns to drop.")
    return df

def pick_chronotype_column(df: pd.DataFrame) -> str | None:
    candidates = [c for c in df.columns if "chronotype" in c]
    LOGGER.info("Searching for chronotype column in candidates: %s", candidates)
    for c in candidates:
        if df[c].astype(str).str.contains(r"\bE\b|\bM\b", case=False, regex=True).any():
            LOGGER.info("Selected chronotype column: %s", c)
            return c
    if not candidates:
        LOGGER.warning("No chronotype-like columns found.")
    else:
        LOGGER.warning("Using fallback chronotype column: %s", candidates[-1])
    return candidates[-1] if candidates else None

def normalize_chronotype(series: pd.Series):
    LOGGER.debug("Normalising chronotype labels. Unique raw values: %s", series.dropna().unique().tolist())
    def _norm(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().upper()
        if s in ["E", "EVENING", "EVENINGNESS", "EVENING-TYPE", "1"]:
            return "E"
        if s in ["M", "MORNING", "MORNINGNESS", "MORNING-TYPE", "0"]:
            return "M"
        return s
    labels = series.apply(_norm)
    binary = labels.map({"E": 1, "M": 0})
    LOGGER.info("Chronotype distribution after normalisation: %s", binary.value_counts(dropna=False).to_dict())
    return labels, binary

def cols_for_electrode(df: pd.DataFrame, ele: str) -> list[str]:
    patt = re.compile(rf"^bin[1-4]_.*_{ele.lower()}$")
    cols = [c for c in df.columns if patt.match(c)]
    LOGGER.debug("Electrode=%s matched columns: %s", ele, cols)
    return cols

def mean_bins(df: pd.DataFrame, cols: list[str], which=(1,2)) -> pd.Series:
    chosen = []
    for b in which:
        patt = re.compile(rf"^bin{b}_", re.IGNORECASE)
        chosen.extend([c for c in cols if patt.match(c)])
    if not chosen:
        return pd.Series(np.nan, index=df.index)
    LOGGER.debug("Averaging bins %s over columns: %s", which, chosen)
    return df[chosen].astype(float).mean(axis=1)

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("🧠 Computing derived ERP/RT features …")
    out = df.copy()

    # ERP contrasts
    fcz = cols_for_electrode(out, "fcz")
    pz  = cols_for_electrode(out, "pz")
    LOGGER.debug("Found FCz columns: %s", fcz)
    LOGGER.debug("Found Pz columns: %s", pz)

    frn_gain_fcz = mean_bins(out, fcz, (1,2))
    frn_loss_fcz = mean_bins(out, fcz, (3,4))
    out["delta_frn_fcz"] = frn_loss_fcz - frn_gain_fcz

    p300_gain_pz = mean_bins(out, pz, (1,2))
    p300_loss_pz = mean_bins(out, pz, (3,4))
    out["delta_p300_pz"] = p300_gain_pz - p300_loss_pz

    # RT contrast: error - correct (avg over gain/loss)
    rt_correct_cols = [c for c in [
        "response_time_after_gain_correct",
        "response_time_after_loss_correct",
    ] if c in out.columns]
    rt_error_cols = [c for c in [
        "response_time_after_gain_error",
        "response_time_after_loss_error",
    ] if c in out.columns]

    if rt_correct_cols and rt_error_cols:
        out["delta_rt"] = out[rt_error_cols].astype(float).mean(axis=1) - \
                          out[rt_correct_cols].astype(float).mean(axis=1)
    else:
        out["delta_rt"] = np.nan

    LOGGER.info("➕ Derived columns added: %s", [c for c in ["delta_frn_fcz","delta_p300_pz","delta_rt"] if c in out.columns])
    return out

def zscore(df: pd.DataFrame, protect: set[str]) -> pd.DataFrame:
    LOGGER.info("Z-scoring numeric columns (protecting: %s)…", protect)
    out = df.copy()
    numeric = out.select_dtypes(include=["number"]).columns.tolist()
    to_scale = [c for c in numeric if c not in protect]
    iter_cols = tqdm(to_scale, desc="🔄 Z-scoring", unit="col") if PRETTY else to_scale
    for c in iter_cols:
        mu = out[c].mean()
        sd = out[c].std(ddof=0)
        if pd.notnull(sd) and sd != 0:
            out[c] = (out[c] - mu) / sd
            LOGGER.debug("Z-scored %s (mu=%.4f, sd=%.4f)", c, mu, sd)
        else:
            LOGGER.debug("Skipped %s (sd=%s)", c, sd)
    LOGGER.info("Z-scoring complete. Scaled %d columns.", len(to_scale))
    return out

# ---------- Commands ----------
def cmd_inspect(args):
    LOGGER.info("Running INSPECT on %d file(s)…", len(args.files))
    files_iter = tqdm(args.files, desc="🔎 Inspecting", unit="file") if PRETTY else args.files
    for f in files_iter:
        inspect_file(Path(f))

def cmd_preprocess(args):
    src = Path(args.input)
    out_clean = Path(args.out_clean)
    out_z = Path(args.out_z)
    LOGGER.info("🚀 Running PREPROCESS: input=%s, out_clean=%s, out_z=%s", src, out_clean, out_z)

    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src.resolve()}")

    df = load_summary_xlsx(src)
    
    df = drop_known_noise(df)

    if "mean_risk_taking_only_free_tials" in df.columns and "mean_risk_taking_only_free_trials" not in df.columns:
        df = df.rename(columns={"mean_risk_taking_only_free_tials": "mean_risk_taking_only_free_trials"})

    chrono_col = pick_chronotype_column(df)
    if not chrono_col:
        raise ValueError("No chronotype column found (needs 'chronotype' in name).")
    df["chronotype_label"], df["chronotype_binary"] = normalize_chronotype(df[chrono_col])

    df = compute_derived_features(df)

    # clip delta_rt to +/- 3 SD (after derived features exist)
    if "delta_rt" in df.columns:
        m, s = df["delta_rt"].mean(), df["delta_rt"].std(ddof=0)
        df["delta_rt"] = df["delta_rt"].clip(lower=m - 3*s, upper=m + 3*s)
        LOGGER.debug("Clipped delta_rt to ±3 SD (mu=%.4f, sd=%.4f)", m, s)

    out_clean.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_clean, index=False)
    LOGGER.info(f"💾 Saved clean table: {out_clean}")

    df_z = zscore(df, protect={"chronotype_binary"})
    df_z.to_csv(out_z, index=False)
    LOGGER.info(f"💾 Saved z-scored features: {out_z}")

    LOGGER.info("Label counts: %s", df["chronotype_binary"].value_counts(dropna=False).to_dict())
    LOGGER.info("Delta columns present: %s", [c for c in ["delta_frn_fcz","delta_p300_pz","delta_rt"] if c in df.columns])

def cmd_qc(args):
    import matplotlib.pyplot as plt
    from scipy import stats

    src_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    LOGGER.info("Running QC: input_csv=%s, out_dir=%s", src_csv, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_csv.exists():
        raise FileNotFoundError(f"Clean CSV not found (run preprocess first): {src_csv.resolve()}")

    df = pd.read_csv(src_csv)

    # 1) Descriptive stats (numeric only)
    desc = df.select_dtypes(include=["number"]).describe().T
    desc_path = out_dir / "descriptives_numeric.csv"
    desc.to_csv(desc_path)
    LOGGER.info(f"📑 Saved: {desc_path}")

    # 2) Histograms for key contrasts
    hist_cols = [c for c in ["delta_frn_fcz", "delta_p300_pz", "delta_rt"] if c in df.columns]
    hist_iter = tqdm(hist_cols, desc="📊 Histograms", unit="fig") if PRETTY else hist_cols
    for col in hist_iter:
        plt.figure()
        df[col].plot(kind="hist", bins=20, edgecolor="black")
        plt.title(col)
        plt.xlabel(col); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{col}.png", dpi=150)
        plt.close()
        LOGGER.info(f"🖼️ Saved: {out_dir / f'hist_{col}.png'}")

    # 3) Correlation heatmap (numeric)
    num = df.select_dtypes(include=["number"])
    if num.shape[1] >= 2:
        corr = num.corr()
        plt.figure()
        im = plt.imshow(corr.values, aspect="auto")
        plt.title("Numeric correlation")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
        plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(out_dir / "corr_heatmap.png", dpi=150)
        plt.close()
        LOGGER.info(f"🌡️ Saved: {out_dir / 'corr_heatmap.png'}")

    # 4) Group-wise QC: E(1) vs M(0)
    if "chronotype_binary" in df.columns:
        g = df.groupby("chronotype_binary")
        group_desc = g[hist_cols].agg(["mean","std","median","count"])
        group_desc.to_csv(out_dir / "group_descriptives.csv")
        LOGGER.info(f"📑 Saved: {out_dir / 'group_descriptives.csv'}")

        # Boxplots per feature by group
        box_iter = tqdm(hist_cols, desc="📦 Boxplots", unit="fig") if PRETTY else hist_cols
        for col in box_iter:
            plt.figure()
            data = [df.loc[df["chronotype_binary"]==0, col].dropna(),
                    df.loc[df["chronotype_binary"]==1, col].dropna()]
            plt.boxplot(data, tick_labels=["M (0)", "E (1)"])
            plt.title(f"{col} by chronotype")
            plt.tight_layout()
            plt.savefig(out_dir / f"box_{col}_by_chronotype.png", dpi=150)
            plt.close()
            LOGGER.info(f"🖼️ Saved: {out_dir / f'box_{col}_by_chronotype.png'}")

        # t-tests + Cohen's d
        rows = []
        test_iter = tqdm(hist_cols, desc="🧪 Group tests", unit="feat") if PRETTY else hist_cols
        for col in test_iter:
            m = df.loc[df["chronotype_binary"]==0, col].dropna()
            e = df.loc[df["chronotype_binary"]==1, col].dropna()
            if len(m) >= 3 and len(e) >= 3:
                t, p = stats.ttest_ind(m, e, equal_var=False)
                # Cohen's d (Hedges’ g correction not necessary with ~equal n)
                pooled_sd = np.sqrt(((m.var(ddof=1)*(len(m)-1)) + (e.var(ddof=1)*(len(e)-1))) / (len(m)+len(e)-2))
                d = (e.mean() - m.mean()) / pooled_sd if pooled_sd > 0 else np.nan
                rows.append({"feature": col, "n_M": len(m), "n_E": len(e), "mean_M": m.mean(), "mean_E": e.mean(), "t": t, "p": p, "cohens_d_E_minus_M": d})
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "group_tests.csv", index=False)
            LOGGER.info(f"📈 Saved: {out_dir / 'group_tests.csv'}")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Chronotype preprocessing utilities")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Logging verbosity")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pretty", dest="pretty", action="store_true", default=True, help="Enable emojis and progress bars (default)")
    group.add_argument("--no-pretty", dest="pretty", action="store_false", help="Disable emojis and progress bars")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_inspect = sub.add_parser("inspect", help="Inspect one or more Excel files")
    p_inspect.add_argument("files", nargs="+", help="Paths to .xlsx files")
    p_inspect.set_defaults(func=cmd_inspect)

    p_prep = sub.add_parser("preprocess", help="Clean names, labels, derived features, z-score")
    p_prep.add_argument("--input", required=True, help="Path to participant_summary.xlsx")
    p_prep.add_argument("--out_clean", default="data/participant_summary_clean.csv", help="Cleaned table CSV")
    p_prep.add_argument("--out_z", default="data/participant_features_zscored.csv", help="Z-scored features CSV")
    p_prep.set_defaults(func=cmd_preprocess)

    p_qc = sub.add_parser("qc", help="Generate QC stats and plots from the cleaned CSV")
    p_qc.add_argument("--input_csv", default="data/participant_summary_clean.csv", help="Path to cleaned CSV")
    p_qc.add_argument("--out_dir", default="reports/qc", help="Directory to write QC outputs")
    p_qc.set_defaults(func=cmd_qc)

    # Composite command: run (preprocess then QC)
    from argparse import ArgumentDefaultsHelpFormatter
    p_run = sub.add_parser(
        "run",
        help="Run preprocess then QC in one go",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    # Inputs/outputs mirroring preprocess and qc
    p_run.add_argument("--input", default="data/raw/participant_summary.xlsx", help="Path to participant_summary.xlsx")
    p_run.add_argument("--out_clean", default="data/participant_summary_clean.csv", help="Cleaned table CSV")
    p_run.add_argument("--out_z", default="data/participant_features_zscored.csv", help="Z-scored features CSV")
    p_run.add_argument("--qc_out_dir", default="reports/qc", help="Directory to write QC outputs")
    p_run.add_argument("--no-qc", action="store_true", help="Skip QC step")
    p_run.add_argument("--inspect", action="store_true", help="Also run an initial inspect of the input file")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    setup_logging(args.log_level, args.pretty)

    # Default to the composite 'run' command if no subcommand is given
    if not getattr(args, "cmd", None):
      LOGGER.info("No subcommand provided; defaulting to 'run'.")
      # Reparse with 'run' as the default command using existing defaults
      args = argparse.Namespace(
          cmd="run",
          input="data/raw/participant_summary.xlsx",
          out_clean="data/participant_summary_clean.csv",
          out_z="data/participant_features_zscored.csv",
          qc_out_dir="reports/qc",
          no_qc=False,
          inspect=False,
      )
      # Ensure logging remains configured; then call cmd_run
      LOGGER.info("Command selected: %s", args.cmd)
      cmd_run(args)
    else:
      LOGGER.info("Command selected: %s", args.cmd)
      args.func(args)

from types import SimpleNamespace

def cmd_run(args):
    # Optional initial inspect
    if args.inspect:
        LOGGER.info("🔎 Running initial INSPECT …")
        cmd_inspect(SimpleNamespace(files=[args.input]))

    # Preprocess
    LOGGER.info("🏗️ Running PREPROCESS phase …")
    cmd_preprocess(SimpleNamespace(
        input=args.input,
        out_clean=args.out_clean,
        out_z=args.out_z,
    ))

    # QC (unless disabled)
    if not args.no_qc:
        LOGGER.info("🧪 Running QC phase …")
        cmd_qc(SimpleNamespace(
            input_csv=args.out_clean,
            out_dir=args.qc_out_dir,
        ))
    else:
        LOGGER.info("⏭️ QC step skipped by --no-qc")

if __name__ == "__main__":
    main()