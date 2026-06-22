#!/usr/bin/env python3
"""Multimodal chronotype decoding: behavioral GRU embedding + EEG EEGNet embedding.

Combines the two independent, leakage-safe per-subject representations and tests
chronotype with the shared permutation-clean nested-LOO evaluator. Reports each
modality alone and the fusion, so we can see whether the neural and behavioral
signals add. Requires the two upstream embedding tables to exist:
  reports/clean/risky_choice_seq/participant_embeddings.csv   (run risky_choice_seq.py)
  reports/clean/eeg_chronotype/subject_eeg_embeddings.csv     (run eeg_chronotype.py)

Run (env_dl):
    python scripts/dl/multimodal_chronotype.py --permutations 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import chrono_eval

OUTDIR = Path("reports/clean/multimodal_chronotype")
BEH = "reports/clean/risky_choice_seq/participant_embeddings.csv"
EEG = "reports/clean/eeg_chronotype/subject_eeg_embeddings.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args()

    beh = pd.read_csv(BEH)
    eeg = pd.read_csv(EEG)
    fused = beh.merge(eeg, on="participant_id")  # intersection of subjects
    beh_only = fused[["participant_id"] + [c for c in fused if c.startswith("gru_emb_")]]
    eeg_only = fused[["participant_id"] + [c for c in fused if c.startswith("eeg_emb_")]]

    runs = {
        "behavioral_only": beh_only,
        "eeg_only": eeg_only,
        "multimodal": fused,
    }
    results = {}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, emb in runs.items():
        out = chrono_eval.evaluate(emb, permutations=args.permutations, label=name)
        results[name] = out["result"]
        print(f"{name:16} AUC={out['result']['nested_loo_roc_auc']:.3f} "
              f"p={out['result']['perm_p_value']:.3f} "
              f"(n_feat={out['result']['n_features']}, n={out['result']['n_participants']})")

    (OUTDIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print("wrote", OUTDIR / "metrics.json")


if __name__ == "__main__":
    main()
