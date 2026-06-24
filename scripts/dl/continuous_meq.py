#!/usr/bin/env python3
"""Predict the CONTINUOUS MEQ score (not the binary chronotype split).

Directly answers the dichotomization objection in docs/limitations.md (12/39 sit in
the MEQ intermediate band where Morning/Evening is "soft"): instead of a binary call,
predict each participant's actual MEQ. Nested leave-one-out Ridge regression
(regularization tuned in an inner CV), scored by predicted-vs-actual Pearson r with a
1000-iteration permutation p-value. Reports behavioral, ERP-P300, RL-parameter, and
fused (behavioral + ERP) feature sets so they are comparable to the classification side.

Run (env_dl):
    python scripts/dl/continuous_meq.py --permutations 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MEQ = "data/processed/participant_meq_scores.csv"
GRU = "reports/clean/risky_choice_seq/participant_embeddings.csv"
COMPACT = "data/clean/chronotype_compact_12.csv"
RL = "reports/clean/rl_model/participant_rl_params.csv"
ERP_FEATURES = ["Fz_FRN_error_minus_correct", "FCz_FRN_error_minus_correct",
                "Fz_FRN_loss_error_minus_gain_correct", "POz_P300_loss_minus_gain",
                "Pz_P300_loss_minus_gain", "CPz_P300_error_minus_correct"]
RL_PARAMS = ["alpha_gain", "alpha_loss", "lr_asymmetry", "beta", "bias"]
OUTDIR = Path("reports/clean/continuous_meq")
SEED = 0


def _estimator():
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    grid = {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}
    inner = KFold(n_splits=5, shuffle=True, random_state=SEED)
    return GridSearchCV(pipe, grid, scoring="neg_mean_squared_error", cv=inner, n_jobs=1)


def _loo_pred(X, y):
    return cross_val_predict(_estimator(), X, y, cv=LeaveOneOut(), n_jobs=1)


def evaluate(X, y, permutations, jobs):
    pred = _loo_pred(X, y)
    r = float(pearsonr(pred, y)[0])
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(y) for _ in range(permutations)]
    null = np.array(Parallel(n_jobs=jobs)(
        delayed(lambda yp: float(pearsonr(_loo_pred(X, yp), yp)[0]))(yp) for yp in perms))
    p = float((1 + np.sum(null >= r)) / (permutations + 1))
    return {"loo_pred_actual_r": round(r, 4), "perm_p_value": round(p, 4),
            "null_r_mean": round(float(null.mean()), 4),
            "null_r_95pct": round(float(np.percentile(null, 95)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    meq = pd.read_csv(MEQ)[["UserID", "meq"]].rename(columns={"UserID": "participant_id"})
    gru = pd.read_csv(GRU)
    erp = pd.read_csv(COMPACT)[["participant_id"] + ERP_FEATURES]
    rl = pd.read_csv(RL)[["participant_id"] + RL_PARAMS]

    base = meq.merge(gru, on="participant_id").merge(erp, on="participant_id").merge(
        rl, on="participant_id").dropna(subset=["meq"])
    y = base["meq"].to_numpy()
    gru_cols = [c for c in gru.columns if c.startswith("gru_emb_")]

    sets = {
        "behavioral_gru": gru_cols,
        "erp_p300": ERP_FEATURES,
        "rl_params": RL_PARAMS,
        "fused_gru_erp": gru_cols + ERP_FEATURES,
    }
    results = {"n_with_meq": int(len(y))}
    print(f"predicting continuous MEQ (n={len(y)})")
    for name, cols in sets.items():
        res = evaluate(base[cols].to_numpy(), y, args.permutations, args.jobs)
        results[name] = res
        print(f"{name:16} r={res['loo_pred_actual_r']:.3f} p={res['perm_p_value']:.3f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print("wrote", OUTDIR / "metrics.json")


if __name__ == "__main__":
    main()
