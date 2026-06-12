#!/usr/bin/env python3
"""Validate the binary chronotype labels against the MEQ score, and export the
continuous MEQ/MCTQ scores de-identified.

Context
-------
`all final data.xlsx` stores several side-by-side table blocks. The `ERPset`
field is actually the participant's name (the EEG dataset name). The `meq` and
`MCTQ` numeric columns live in a different block from the `chronotype` column and
are NOT row-aligned with it, which is why earlier work declined to export them.
However, every block is keyed by participant name, so the blocks can be aligned
safely by name rather than by row position.

This script:
  1. reads the MEQ/MCTQ block (name -> meq, MCTQ),
  2. attaches the MEQ score to each analysed participant by joining the
     name to `participant_metadata.csv` (UserID, summary_erpset, label),
  3. checks that the binary chronotype label is consistent with the MEQ score
     under the standard Horne-Ostberg direction (higher MEQ = more morning),
  4. writes a DE-IDENTIFIED table (UserID + meq + MCTQ + label only, no names)
     to data/processed/. No participant names are written to disk.

The output directory is git-ignored; nothing here is committed automatically.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_name(value: object) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"^final\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def read_meq_block(final_path: str) -> pd.DataFrame:
    """Read (name, meq, MCTQ) from the block that contains the meq column.

    Columns are located by header name; the name key is the nearest ERPset-like
    column to the left of the meq column, so meq and its name stay row-aligned.
    """
    raw = pd.read_excel(final_path, header=0)
    cols = [str(c).strip().lower() for c in raw.columns]
    erp_pos = [i for i, c in enumerate(cols) if c.startswith("erpset")]
    meq_pos = cols.index("meq")
    mctq_pos = cols.index("mctq") if "mctq" in cols else None
    name_pos = max(i for i in erp_pos if i <= meq_pos)

    block = pd.DataFrame({
        "name_key": raw.iloc[:, name_pos].map(normalize_name),
        "meq": pd.to_numeric(raw.iloc[:, meq_pos], errors="coerce"),
    })
    if mctq_pos is not None:
        block["MCTQ"] = pd.to_numeric(raw.iloc[:, mctq_pos], errors="coerce")
    block = block[block["name_key"].ne("nan")].dropna(subset=["meq"]).drop_duplicates("name_key")
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate chronotype labels against MEQ score.")
    parser.add_argument("--final", default="data/raw/all final data.xlsx")
    parser.add_argument("--metadata", default="data/processed/participant_metadata.csv")
    parser.add_argument("--out", default="data/processed/participant_meq_scores.csv")
    parser.add_argument("--evening-max", type=float, default=41.0,
                        help="Standard MEQ upper bound for eveningness (<=41).")
    parser.add_argument("--morning-min", type=float, default=59.0,
                        help="Standard MEQ lower bound for morningness (>=59).")
    args = parser.parse_args()

    meq_block = read_meq_block(args.final)
    meta = pd.read_csv(args.metadata)
    meta["name_key"] = meta["summary_erpset"].map(normalize_name)

    joined = meta.merge(meq_block, on="name_key", how="left")
    matched = joined["meq"].notna()
    print(f"MEQ score attached for {int(matched.sum())} / {len(joined)} participants.")

    sub = joined[matched].copy()
    # Implied class under the standard direction, with an intermediate band.
    def implied(meq: float) -> str:
        if meq <= args.evening_max:
            return "Evening"
        if meq >= args.morning_min:
            return "Morning"
        return "Intermediate"

    sub["meq_implied_class"] = sub["meq"].map(implied)
    sub["label_meq_consistent"] = np.where(
        sub["meq_implied_class"].eq("Intermediate"),
        pd.NA,
        sub["meq_implied_class"].eq(sub["primary_chronotype"]),
    )

    by_label = sub.groupby("primary_chronotype")["meq"].agg(["count", "min", "max", "mean", "median"])
    print("\nMEQ by primary label:")
    print(by_label.to_string())

    decisive = sub[sub["meq_implied_class"].ne("Intermediate")]
    n_consistent = int((decisive["meq_implied_class"].eq(decisive["primary_chronotype"])).sum())
    print(f"\nDecisive participants (outside intermediate band): {len(decisive)}")
    print(f"Label consistent with MEQ direction: {n_consistent} / {len(decisive)}")
    inconsistent = decisive[decisive["meq_implied_class"].ne(decisive["primary_chronotype"])]
    if len(inconsistent):
        print("INCONSISTENT UserIDs:", inconsistent["UserID"].tolist())
    n_intermediate = int(sub["meq_implied_class"].eq("Intermediate").sum())
    print(f"Participants in MEQ intermediate band (42-58): {n_intermediate}")

    # De-identified export: no names.
    out_cols = ["UserID", "primary_chronotype", "meq", "meq_implied_class", "label_meq_consistent"]
    if "MCTQ" in sub.columns:
        out_cols.insert(3, "MCTQ")
    out = sub[out_cols].sort_values("UserID")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote de-identified MEQ scores to {out_path} (no participant names).")


if __name__ == "__main__":
    main()
