#!/usr/bin/env python3
"""Build a smaller journal-oriented chronotype feature set.

The compact pack is deliberately limited to theory-driven behavioral adaptation
and ERP contrast features. This is more defensible than the broader 47-feature
pilot model for a 39-participant dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


COMPACT_FEATURES = [
    "post_error_slowing",
    "rt_slope",
    "risky_late_minus_early",
    "risk_after_loss_error_minus_gain_correct",
    "gain_correct_risky_rate",
    "loss_error_risky_rate",
    "Fz_FRN_error_minus_correct",
    "FCz_FRN_error_minus_correct",
    "Fz_FRN_loss_error_minus_gain_correct",
    "POz_P300_loss_minus_gain",
    "Pz_P300_loss_minus_gain",
    "CPz_P300_error_minus_correct",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact theory-driven chronotype model table.")
    parser.add_argument("--input", default="data/clean/chronotype_participant.csv")
    parser.add_argument("--out", default="data/clean/chronotype_compact_12.csv")
    parser.add_argument("--manifest", default="data/clean/chronotype_compact_12_manifest.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = ["participant_id", "Chronotype"]
    missing = [c for c in required + COMPACT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing compact features: {missing}")

    cols = required + COMPACT_FEATURES
    out = df[cols].copy()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    manifest = {
        "input": args.input,
        "output": args.out,
        "target": "Chronotype",
        "rows": int(out.shape[0]),
        "features": COMPACT_FEATURES,
        "rationale": "Theory-driven compact chronotype set: behavioral adaptation, post-error slowing, FRN contrasts, and P300 contrasts.",
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with shape {out.shape}")


if __name__ == "__main__":
    main()
