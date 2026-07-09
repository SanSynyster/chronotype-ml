#!/usr/bin/env python3
"""Robustness battery + bootstrap CIs for the fused chronotype result (n=39).

Since no new participants are obtainable, this reviewer-proofs the headline fused
(GRU behavioral + validated ERP P300/FRN) result on the fixed sample:

  1. Full model: nested-LOO AUC + 1000-permutation p (canonical number).
  2. Bootstrap 95% CI on the AUC (resampling participants from the out-of-fold scores).
  3. Sensitivity to participant exclusions (matching docs sensitivity scenarios plus
     the MEQ intermediate band) -- does the effect survive dropping flagged / soft-label
     participants?
  4. Leave-one-subject-out influence: drop each participant once; report the AUC range
     and the most influential participant (shows no single subject drives the result).

Run (env_dl):
    python scripts/dl/robustness.py --permutations 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import chrono_eval

GRU = "reports/clean/risky_choice_seq/participant_embeddings.csv"
COMPACT = "data/clean/chronotype_compact_12.csv"
MEQ = "data/processed/participant_meq_scores.csv"
ERP_FEATURES = ["Fz_FRN_error_minus_correct", "FCz_FRN_error_minus_correct",
                "Fz_FRN_loss_error_minus_gain_correct", "POz_P300_loss_minus_gain",
                "Pz_P300_loss_minus_gain", "POz_P300_error_minus_correct"]
OUTDIR = Path("reports/clean/robustness")
SEED = 0

EXCLUSIONS = {
    "exclude_1013": [1013],
    "exclude_label_conflicts": [1027, 1036],
    "exclude_all_flagged": [1013, 1027, 1036],
    "exclude_meq_intermediate": [1002, 1010, 1016, 1021, 1022, 1033,
                                 1034, 1040, 1041, 1042, 1046, 1056],
}


def fused_embedding() -> pd.DataFrame:
    gru = pd.read_csv(GRU)
    compact = pd.read_csv(COMPACT)
    features = [c for c in ERP_FEATURES if c in compact.columns]
    erp = compact[["participant_id"] + features]
    return gru.merge(erp, on="participant_id")


def bootstrap_ci(oof: pd.DataFrame, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, s = oof["y"].to_numpy(), oof["score"].to_numpy()
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], s[b]))
    return (round(float(np.percentile(aucs, 2.5)), 3),
            round(float(np.percentile(aucs, 97.5)), 3),
            round(float(np.mean(aucs)), 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args()

    fused = fused_embedding()
    out = {}

    # 1 + 2: full model with permutation p, then bootstrap CI from its OOF scores.
    full = chrono_eval.evaluate(fused, permutations=args.permutations, label="fused")
    oof = full["oof"].rename(columns={"chronotype_evening": "y", "oof_decision_score": "score"})
    lo, hi, mean = bootstrap_ci(oof)
    out["full"] = {
        "n": int(len(fused)),
        "nested_loo_roc_auc": full["result"]["nested_loo_roc_auc"],
        "perm_p_value": full["result"]["perm_p_value"],
        "bootstrap_auc_95ci": [lo, hi], "bootstrap_auc_mean": mean,
    }
    print(f"full: AUC={out['full']['nested_loo_roc_auc']:.3f} "
          f"p={out['full']['perm_p_value']:.3f} "
          f"bootstrap95%=[{lo},{hi}]")

    # 3: exclusion sensitivity (point estimates, fast).
    out["exclusions"] = {}
    for name, drop in EXCLUSIONS.items():
        sub = fused[~fused["participant_id"].isin(drop)]
        res = chrono_eval.nested_auc(sub)
        n_ev = int(res["oof"]["y"].sum())
        out["exclusions"][name] = {"n": int(len(sub)), "n_evening": n_ev,
                                   "n_morning": int(len(sub) - n_ev),
                                   "nested_loo_roc_auc": round(res["roc_auc"], 3),
                                   "balanced_accuracy": round(res["balanced_accuracy"], 3)}
        print(f"{name:26} n={len(sub):2d} AUC={res['roc_auc']:.3f}")

    # 4: leave-one-subject-out influence.
    aucs = {}
    for pid in fused["participant_id"]:
        sub = fused[fused["participant_id"] != pid]
        aucs[int(pid)] = chrono_eval.nested_auc(sub)["roc_auc"]
    vals = np.array(list(aucs.values()))
    most_inf = min(aucs, key=aucs.get)  # subject whose removal most lowers AUC
    out["loo_influence"] = {
        "auc_min": round(float(vals.min()), 3), "auc_max": round(float(vals.max()), 3),
        "auc_range": round(float(vals.max() - vals.min()), 3),
        "most_influential_participant": most_inf,
        "auc_without_most_influential": round(aucs[most_inf], 3),
    }
    print(f"LOO influence: AUC range [{vals.min():.3f}, {vals.max():.3f}], "
          f"most influential = {most_inf}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "metrics.json").write_text(json.dumps(out, indent=2))
    print("wrote", OUTDIR / "metrics.json")


if __name__ == "__main__":
    main()
