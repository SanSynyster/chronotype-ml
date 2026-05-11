#!/usr/bin/env python3
"""Build chronotype sensitivity datasets by excluding flagged participants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_compact_chronotype import COMPACT_FEATURES


SCENARIOS = {
    "exclude_1013": [1013],
    "exclude_label_conflicts": [1027, 1036],
    "exclude_all_flagged": [1013, 1027, 1036],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build participant-level chronotype sensitivity datasets.")
    parser.add_argument("--input", default="data/clean/chronotype_participant.csv")
    parser.add_argument("--outdir", default="data/clean/sensitivity")
    parser.add_argument("--manifest", default="data/clean/sensitivity/manifest.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "participant_id" not in df.columns:
        raise ValueError("Input must contain participant_id")

    required = ["participant_id", "Chronotype"] + COMPACT_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing compact features: {missing}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"input": args.input, "scenarios": {}}

    for name, excluded in SCENARIOS.items():
        out = df[~df["participant_id"].isin(excluded)].copy()
        participant_path = outdir / f"chronotype_participant_{name}.csv"
        compact_path = outdir / f"chronotype_compact_12_{name}.csv"
        out.to_csv(participant_path, index=False)
        out[["participant_id", "Chronotype"] + COMPACT_FEATURES].to_csv(compact_path, index=False)
        manifest["scenarios"][name] = {
            "excluded_participants": excluded,
            "rows": int(out.shape[0]),
            "chronotype_counts": out["Chronotype"].value_counts(dropna=False).to_dict(),
            "participant_path": str(participant_path),
            "compact_path": str(compact_path),
        }
        print(f"Wrote {participant_path} and {compact_path}")

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
