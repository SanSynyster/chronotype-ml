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
    for col in ["erpset", "erpset_1"]:
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

# ---------- Commands ----------
def cmd_inspect(args):
    for f in args.files:
        inspect_file(Path(f))

def cmd_preprocess(args):
    src = Path(args.input)
    out_clean = Path(args.out_clean)

    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src.resolve()}")

    df = load_summary_xlsx(src)
    df = drop_known_noise(df)

    # fix known typo: "tials" -> "trials"
    if "mean_risk_taking_only_free_tials" in df.columns and "mean_risk_taking_only_free_trials" not in df.columns:
        df = df.rename(columns={"mean_risk_taking_only_free_tials": "mean_risk_taking_only_free_trials"})

    chrono_col = pick_chronotype_column(df)
    if not chrono_col:
        raise ValueError("No chronotype column found (name must contain 'chronotype').")
    df["chronotype_label"], df["chronotype_binary"] = normalize_chronotype(df[chrono_col])

    out_clean.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_clean, index=False)

    print(f"Saved clean table: {out_clean}")
    print("Label counts:", df["chronotype_binary"].value_counts(dropna=False).to_dict())

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Chronotype preprocessing utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect one or more Excel files")
    p_inspect.add_argument("files", nargs="+", help="Paths to .xlsx files")
    p_inspect.set_defaults(func=cmd_inspect)

    p_prep = sub.add_parser("preprocess", help="Clean names + standardize chronotype labels")
    p_prep.add_argument("--input", required=True, help="Path to participant_summary.xlsx")
    p_prep.add_argument("--out_clean", default="data/participant_summary_clean.csv",
                        help="Where to write the cleaned table (CSV)")
    p_prep.set_defaults(func=cmd_preprocess)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()