#!/usr/bin/env python3
"""Naive baselines that contextualize the risky-choice classifier.

The leakage-safe risky-choice model reaches balanced accuracy around 0.587 under
participant-grouped CV. On its own that number is hard to read, so this script
reports the trivial reference points a reviewer will ask for:

- majority-class: always predict the more common outcome
- persistence: predict the participant's previous-trial choice (autocorrelation)
- participant-mean (oracle): predict each participant's own mean rate, thresholded
  at 0.5. This is NOT achievable under grouped CV (it peeks at the held-out
  participant) and is reported only as an individual-differences ceiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Risky-choice naive baselines.")
    parser.add_argument("--data", default="data/clean/risky_choice_prechoice.csv")
    parser.add_argument("--target", default="risky-choice")
    parser.add_argument("--prev-col", default="PrevRisky")
    parser.add_argument("--group-col", default="participant_id")
    parser.add_argument("--outdir", default="reports/clean/risky_choice_baseline")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df[df[args.target].notna()].reset_index(drop=True)
    y = df[args.target].astype(int).to_numpy()

    base_rate = float(y.mean())
    majority = int(round(base_rate))
    results = {
        "data": args.data,
        "n_rows": int(len(y)),
        "base_rate_risky": round(base_rate, 4),
        "baselines": {},
    }

    # Majority-class.
    y_major = np.full_like(y, majority)
    results["baselines"]["majority_class"] = {
        "accuracy": round(accuracy_score(y, y_major), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, y_major), 4),
    }

    # Persistence: predict previous-trial choice.
    if args.prev_col in df.columns:
        mask = df[args.prev_col].notna().to_numpy()
        y_prev = df.loc[mask, args.prev_col].astype(int).to_numpy()
        results["baselines"]["persistence_prev_choice"] = {
            "n_evaluated": int(mask.sum()),
            "accuracy": round(accuracy_score(y[mask], y_prev), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y[mask], y_prev), 4),
        }

    # Participant-mean oracle (upper reference, not grouped-CV achievable).
    if args.group_col in df.columns:
        means = df.groupby(args.group_col)[args.target].transform("mean")
        y_pmean = (means.to_numpy() >= 0.5).astype(int)
        results["baselines"]["participant_mean_oracle"] = {
            "accuracy": round(accuracy_score(y, y_pmean), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y, y_pmean), 4),
            "note": "peeks at held-out participant; individual-differences ceiling only",
        }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
