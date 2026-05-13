#!/usr/bin/env python3
"""Build an exploratory performance-informed compact chronotype table.

This feature set is intentionally separate from the theory-driven compact set.
It uses features that were recurrently useful in exploratory model importance,
feature-pack performance, and group statistics in this dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PERFORMANCE_FEATURES = [
    "free_risky_rate",
    "gain_correct_risky_rate",
    "loss_error_risky_rate",
    "risk_after_loss_error_minus_gain_correct",
    "risky_late_minus_early",
    "Fz_FRN_error_minus_correct",
    "FCz_FRN_error_minus_correct",
    "Fz_FRN_loss_error_minus_gain_correct",
    "FCz_FRN_loss_error_minus_gain_correct",
    "Pz_P300_loss_minus_gain",
    "POz_P300_loss_minus_gain",
    "CPz_P300_error_minus_correct",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build performance-informed compact chronotype table.")
    parser.add_argument("--input", default="data/clean/chronotype_participant.csv")
    parser.add_argument("--out", default="data/clean/chronotype_compact_performance.csv")
    parser.add_argument("--manifest", default="data/clean/chronotype_compact_performance_manifest.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = ["participant_id", "Chronotype"]
    missing = [c for c in required + PERFORMANCE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing performance-informed features: {missing}")

    cols = required + PERFORMANCE_FEATURES
    out = df[cols].copy()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    manifest = {
        "input": args.input,
        "output": args.out,
        "target": "Chronotype",
        "rows": int(out.shape[0]),
        "features": PERFORMANCE_FEATURES,
        "rationale": "Exploratory performance-informed compact set based on recurrent feature-pack, importance, and group-stat signals. Not a replacement for the a priori theory-driven compact set.",
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with shape {out.shape}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
