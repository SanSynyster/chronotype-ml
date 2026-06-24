#!/usr/bin/env python3
"""Principled multimodal fusion: behavioral GRU embedding + validated ERP features.

Unlike multimodal_chronotype.py (which fused a noisy *learned* EEG embedding and
diluted the behavioral signal), this fuses the GRU embedding with the project's
hand-crafted, FDR-validated FRN/P300 contrast features from chronotype_compact_12.
These are low-dimensional and individually proven to separate chronotypes, so they
are good fusion partners at n=39. The two signals are independent (behavioral vs.
neural), so they have a real chance to ADD rather than dilute.

Reports GRU-alone, ERP-alone, and fused, all through the shared permutation-clean
nested-LOO evaluator (chrono_eval.py).

Run (env_dl):
    python scripts/dl/fusion_gru_p300.py --permutations 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import chrono_eval

OUTDIR = Path("reports/clean/fusion_gru_p300")
GRU = "reports/clean/risky_choice_seq/participant_embeddings.csv"
COMPACT = "data/clean/chronotype_compact_12.csv"

# Validated neural ERP contrasts (Pz/POz P300 loss-minus-gain survived FDR).
ERP_FEATURES = [
    "Fz_FRN_error_minus_correct", "FCz_FRN_error_minus_correct",
    "Fz_FRN_loss_error_minus_gain_correct", "POz_P300_loss_minus_gain",
    "Pz_P300_loss_minus_gain", "CPz_P300_error_minus_correct",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args()

    gru = pd.read_csv(GRU)
    erp = pd.read_csv(COMPACT)[["participant_id"] + ERP_FEATURES]
    fused = gru.merge(erp, on="participant_id")

    gru_only = fused[["participant_id"] + [c for c in gru.columns if c.startswith("gru_emb_")]]
    erp_only = fused[["participant_id"] + ERP_FEATURES]

    runs = {"gru_behavioral": gru_only, "erp_p300": erp_only, "fused": fused}
    results = {}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, emb in runs.items():
        out = chrono_eval.evaluate(emb, permutations=args.permutations, label=name)
        results[name] = out["result"]
        r = out["result"]
        print(f"{name:15} AUC={r['nested_loo_roc_auc']:.3f} p={r['perm_p_value']:.3f} "
              f"BA={r['nested_loo_balanced_accuracy']:.3f} "
              f"MEQ_r={r['meq_pearson_r']:.2f} (n_feat={r['n_features']})", flush=True)

    (OUTDIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print("wrote", OUTDIR / "metrics.json")


if __name__ == "__main__":
    main()
