#!/usr/bin/env python3
"""Generate lightweight QC reports for clean modelling tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(path: Path, target: str | None) -> dict:
    df = pd.read_csv(path)
    out = {
        "path": str(path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_top20": df.isna().mean().sort_values(ascending=False).head(20).to_dict(),
    }
    if target and target in df.columns:
        out["target_counts"] = df[target].astype(str).value_counts(dropna=False).to_dict()
    if "participant_id" in df.columns:
        out["participants"] = int(df["participant_id"].nunique())
        out["rows_per_participant"] = df.groupby("participant_id").size().describe().to_dict()
    if "Chronotype" in df.columns:
        out["chronotype_counts"] = df["Chronotype"].astype(str).value_counts(dropna=False).to_dict()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean dataset QC summary.")
    parser.add_argument("--out", default="reports/clean/qc/summary.json")
    args = parser.parse_args()
    tables = [
        (Path("data/clean/chronotype_participant.csv"), "Chronotype"),
        (Path("data/clean/chronotype_compact_12.csv"), "Chronotype"),
        (Path("data/clean/risky_choice_value_history.csv"), "risky-choice"),
    ]
    summaries = [summarize(path, target) for path, target in tables if path.exists()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"tables": summaries}, indent=2), encoding="utf-8")

    lines = ["# QC Summary", ""]
    for item in summaries:
        lines.extend([
            f"## `{item['path']}`",
            "",
            f"- Rows: `{item['rows']}`",
            f"- Columns: `{item['columns']}`",
        ])
        if "participants" in item:
            lines.append(f"- Participants: `{item['participants']}`")
        if "target_counts" in item:
            lines.append(f"- Target counts: `{item['target_counts']}`")
        lines.append("")
    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
