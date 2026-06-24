#!/usr/bin/env python3
"""Export a PII-free, anonymized participant-level table for public release.

The derived participant-level table contains only aggregated behavioural and ERP
features plus chronotype, age, and gender. It contains no names, emails, phone
numbers, or dates. This script additionally replaces the study `participant_id`
code with an anonymous label (P01..P39) so the released file cannot be linked
back to the raw study identifiers.

The output is written under data/public/ which remains git-ignored: releasing the
dataset is a deliberate decision for the authors to make, so this script prepares
the file but does not commit it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Export anonymized participant-level data.")
    parser.add_argument("--data", default="data/clean/chronotype_participant.csv")
    parser.add_argument("--out", default="data/public/chronotype_participant_public.csv")
    parser.add_argument("--id-col", default="participant_id")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    # Sanity check: refuse to export if any obvious PII column sneaks in.
    forbidden = [c for c in df.columns if any(k in c.lower() for k in ("name", "email", "phone", "erpset", "birth"))]
    if forbidden:
        raise SystemExit(f"Refusing to export: potential PII columns present: {forbidden}")

    df = df.sort_values(args.id_col).reset_index(drop=True)
    df.insert(0, "pid", [f"P{i + 1:02d}" for i in range(len(df))])
    df = df.drop(columns=[args.id_col])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with shape {df.shape} (study id replaced by anonymous pid)")
    print("Columns are aggregated features + chronotype/age/gender; no direct identifiers.")


if __name__ == "__main__":
    main()
