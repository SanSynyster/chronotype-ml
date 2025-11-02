# scripts/preprocess_participant_summary.py
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

# ---------- Utilities ----------
def clean_col(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[^\w]+", "_", s)           # non-alnum -> underscore
    s = re.sub(r"_+", "_", s).strip("_")    # collapse + trim
    return s.lower()

def inspect_file(path: Path) -> None:
    if not path.exists():
        print(f"[!] Not found: {path}")
        return
    xls = pd.ExcelFile(path)
    print(f"\n=== {path} ===")
    print(f"Sheets: {xls.sheet_names}")
    sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    print(f"Sheet: {sheet}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    print("Columns:")
    for i, c in enumerate(df.columns, 1):
        print(f"  {i:2d}. {c}")
    print("Preview (first 5 rows):")
    print(df.head(5).to_string(index=False))

def load_summary_xlsx(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = [clean_col(c) for c in df.columns]
    return df

def drop_known_noise(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["erpset", "erpset_1", "erp_set", "erp_set_1", "unnamed_13", "unnamed_28", "unnamed_30"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def pick_chronotype_column(df: pd.DataFrame) -> str | None:
    candidates = [c for c in df.columns if "chronotype" in c]
    for c in candidates:
        if df[c].astype(str).str.contains(r"\bE\b|\bM\b", case=False, regex=True).any():
            return c
    return candidates[-1] if candidates else None

def normalize_chronotype(series: pd.Series) -> tuple[pd.Series, pd.Series]:
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
    return labels, binary

def cols_for_electrode(df: pd.DataFrame, ele: str) -> list[str]:
    patt = re.compile(rf"^bin[1-4]_.*_{ele.lower()}$")
    return [c for c in df.columns if patt.match(c)]

def mean_bins(df: pd.DataFrame, cols: list[str], which=(1,2)) -> pd.Series:
    chosen = []
    for b in which:
        patt = re.compile(rf"^bin{b}_", re.IGNORECASE)
        chosen.extend([c for c in cols if patt.match(c)])
    if not chosen:
        return pd.Series(np.nan, index=df.index)
    return df[chosen].astype(float).mean(axis=1)

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ERP contrasts
    fcz = cols_for_electrode(out, "fcz")
    pz  = cols_for_electrode(out, "pz")

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

    return out

def zscore(df: pd.DataFrame, protect: set[str]) -> pd.DataFrame:
    out = df.copy()
    numeric = out.select_dtypes(include=["number"]).columns.tolist()
    to_scale = [c for c in numeric if c not in protect]
    for c in to_scale:
        mu = out[c].mean()
        sd = out[c].std(ddof=0)
        if pd.notnull(sd) and sd != 0:
            out[c] = (out[c] - mu) / sd
    return out

# ---------- Commands ----------
def cmd_inspect(args):
    for f in args.files:
        inspect_file(Path(f))

def cmd_preprocess(args):
    src = Path(args.input)
    out_clean = Path(args.out_clean)
    out_z = Path(args.out_z)

    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src.resolve()}")

    df = load_summary_xlsx(src)
    df = drop_known_noise(df)

    # normalize known typo
    if "mean_risk_taking_only_free_tials" in df.columns and "mean_risk_taking_only_free_trials" not in df.columns:
        df = df.rename(columns={"mean_risk_taking_only_free_tials": "mean_risk_taking_only_free_trials"})

    chrono_col = pick_chronotype_column(df)
    if not chrono_col:
        raise ValueError("No chronotype column found (needs 'chronotype' in name).")
    df["chronotype_label"], df["chronotype_binary"] = normalize_chronotype(df[chrono_col])

    # Derived deltas
    df = compute_derived_features(df)

    # Save clean (unscaled)
    out_clean.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_clean, index=False)
    print(f"Saved clean table: {out_clean}")

    # Z-scored copy (keep label untouched)
    df_z = zscore(df, protect={"chronotype_binary"})
    df_z.to_csv(out_z, index=False)
    print(f"Saved z-scored features: {out_z}")

    # Quick check
    print("Label counts:", df["chronotype_binary"].value_counts(dropna=False).to_dict())
    print("Delta columns present:", [c for c in ["delta_frn_fcz","delta_p300_pz","delta_rt"] if c in df.columns])

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Chronotype preprocessing utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect one or more Excel files")
    p_inspect.add_argument("files", nargs="+", help="Paths to .xlsx files")
    p_inspect.set_defaults(func=cmd_inspect)

    p_prep = sub.add_parser("preprocess", help="Clean names, labels, derived features, z-score")
    p_prep.add_argument("--input", required=True, help="Path to participant_summary.xlsx")
    p_prep.add_argument("--out_clean", default="data/participant_summary_clean.csv",
                        help="Cleaned table CSV")
    p_prep.add_argument("--out_z", default="data/participant_features_zscored.csv",
                        help="Z-scored features CSV")
    p_prep.set_defaults(func=cmd_preprocess)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()