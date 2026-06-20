#!/usr/bin/env python3
"""Permutation-clean test: is chronotype decodable from risky-choice *dynamics*?

The causal GRU (scripts/dl/risky_choice_seq.py) is trained ONLY to predict trial-level
risky choice; it never sees chronotype. Its out-of-fold participant embeddings are
therefore a leakage-safe, chronotype-agnostic summary of each participant's
reinforcement/choice dynamics. We then ask whether a simple classifier on those
embeddings predicts chronotype above chance.

Why this is permutation-clean (fixes the earlier post-hoc PCA-dim selection):
  * The downstream classifier's hyperparameters (PCA dim, L2 strength C) are chosen
    INSIDE nested leave-one-out CV by an inner grid search -- never by looking at the
    outer score.
  * The ENTIRE nested procedure is re-run under every label permutation, so the null
    distribution already absorbs the model-selection step. The reported p-value needs
    no further multiple-comparison correction for the config grid.

Primary inferential statistic: nested-LOO ROC AUC for Evening-vs-Morning, with a
label-permutation p-value. Secondary corroboration: Pearson correlation between the
nested out-of-fold decision scores and the continuous MEQ score (lower MEQ = more
evening), which is an independent ground-truth axis.

Run (uses the isolated env_dl/ venv):
    python scripts/dl/chronotype_from_dynamics.py --permutations 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import risky_choice_seq as rcs

CHRONO = "data/clean/chronotype_participant.csv"
MEQ = "data/processed/participant_meq_scores.csv"
OUTDIR = Path("reports/clean/chronotype_from_dynamics")
SEED = 0

# Inner-CV grid (selected inside nested CV, not by hand).
PARAM_GRID = {"pca__n_components": [5, 10, 20], "clf__C": [0.1, 0.5, 1.0]}


def make_estimator() -> GridSearchCV:
    """Nested-CV inner estimator: standardize -> PCA -> L2 logistic, tuned by inner CV."""
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(svd_solver="full", random_state=SEED)),
        ("clf", LogisticRegression(max_iter=5000)),
    ])
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    return GridSearchCV(pipe, PARAM_GRID, scoring="roc_auc", cv=inner, n_jobs=1, refit=True)


def nested_oof_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Outer leave-one-out decision scores from the inner-tuned estimator."""
    return cross_val_predict(make_estimator(), X, y, cv=LeaveOneOut(),
                             method="decision_function", n_jobs=1)


def auc_from_scores(y: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores))


def balanced_acc_from_scores(y: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, (scores >= 0.0).astype(int)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--regenerate-embeddings", action="store_true",
                    help="recompute GRU embeddings instead of reading the cached CSV")
    args = ap.parse_args()

    # --- Load leakage-safe GRU embeddings (chronotype-agnostic features) ---
    cached = Path("reports/clean/risky_choice_seq/participant_embeddings.csv")
    if args.regenerate_embeddings or not cached.exists():
        df = pd.read_csv(rcs.DATA)
        df = df[df[rcs.TARGET].notna()].copy()
        df[rcs.FEATURES] = df[rcs.FEATURES].fillna(0.0)
        emb = rcs.compute_oof(df)["embeddings"]
    else:
        emb = pd.read_csv(cached)

    lab = pd.read_csv(CHRONO)[["participant_id", "Chronotype"]]
    meq = pd.read_csv(MEQ)[["UserID", "meq"]]
    data = emb.merge(lab, on="participant_id").merge(
        meq, left_on="participant_id", right_on="UserID", how="left")

    feat_cols = [c for c in emb.columns if c.startswith("gru_emb_")]
    X = data[feat_cols].to_numpy()
    y = (data["Chronotype"].str.lower() == "evening").astype(int).to_numpy()

    # --- Primary: nested-LOO AUC + label-permutation p-value ---
    obs_scores = nested_oof_scores(X, y)
    obs_auc = auc_from_scores(y, obs_scores)
    obs_ba = balanced_acc_from_scores(y, obs_scores)

    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(y) for _ in range(args.permutations)]

    def _perm_auc(yp):
        return auc_from_scores(yp, nested_oof_scores(X, yp))

    null = np.array(Parallel(n_jobs=args.jobs)(delayed(_perm_auc)(yp) for yp in perms))
    p_value = float((1 + np.sum(null >= obs_auc)) / (args.permutations + 1))

    # --- Secondary: corroborate against continuous MEQ (independent ground truth) ---
    has_meq = data["meq"].notna().to_numpy()
    # Evening-positive scores should track LOWER MEQ -> expect negative r.
    r_meq, p_meq = pearsonr(obs_scores[has_meq], data["meq"].to_numpy()[has_meq])

    result = {
        "design": "GRU embeddings (chronotype-agnostic) -> nested-LOO logistic; "
                  "model selection inside permutation, so p needs no extra correction.",
        "n_participants": int(len(y)),
        "n_evening": int(y.sum()),
        "n_morning": int((1 - y).sum()),
        "param_grid": PARAM_GRID,
        "primary": {
            "nested_loo_roc_auc": round(obs_auc, 4),
            "nested_loo_balanced_accuracy": round(obs_ba, 4),
            "permutations": args.permutations,
            "perm_p_value": round(p_value, 4),
            "null_auc_mean": round(float(null.mean()), 4),
            "null_auc_95pct": round(float(np.percentile(null, 95)), 4),
        },
        "secondary_meq": {
            "n_with_meq": int(has_meq.sum()),
            "pearson_r_score_vs_meq": round(float(r_meq), 4),
            "pearson_p": round(float(p_meq), 4),
            "expected_sign": "negative (evening-positive score <-> lower MEQ)",
        },
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "metrics.json").write_text(json.dumps(result, indent=2))
    np.save(OUTDIR / "null_auc.npy", null)
    pd.DataFrame({"participant_id": data["participant_id"], "chronotype_evening": y,
                  "oof_decision_score": obs_scores, "meq": data["meq"]}).to_csv(
        OUTDIR / "oof_scores.csv", index=False)
    print(json.dumps(result, indent=2))
    print("wrote", OUTDIR / "metrics.json")


if __name__ == "__main__":
    main()
