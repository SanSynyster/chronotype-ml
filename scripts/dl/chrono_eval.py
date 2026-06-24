"""Shared permutation-clean chronotype evaluator.

Given a per-participant embedding table (participant_id + feature columns), test
whether chronotype is decodable from it. Hyperparameters are tuned INSIDE nested
leave-one-out CV and the whole nested procedure is re-run under every label
permutation, so the reported p-value already accounts for model selection and
needs no further multiple-comparison correction.

Used by chronotype_from_dynamics.py (behavioral GRU), eeg_chronotype.py (EEGNet),
and multimodal_chronotype.py (both), so the three results are directly comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CHRONO = "data/clean/chronotype_participant.csv"
MEQ = "data/processed/participant_meq_scores.csv"
SEED = 0
PARAM_GRID = {"pca__n_components": [5, 10, 20], "clf__C": [0.1, 0.5, 1.0]}


def _estimator(max_pca: int) -> GridSearchCV:
    grid = {"pca__n_components": [k for k in PARAM_GRID["pca__n_components"] if k <= max_pca],
            "clf__C": PARAM_GRID["clf__C"]}
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(svd_solver="full", random_state=SEED)),
        ("clf", LogisticRegression(max_iter=5000)),
    ])
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1, refit=True)


def _nested_scores(X, y, max_pca):
    return cross_val_predict(_estimator(max_pca), X, y, cv=LeaveOneOut(),
                             method="decision_function", n_jobs=1)


def _prepare(emb: pd.DataFrame):
    """Merge embedding with chronotype/MEQ labels; return X, y, data, max_pca."""
    feats = [c for c in emb.columns if c != "participant_id"]
    lab = pd.read_csv(CHRONO)[["participant_id", "Chronotype"]]
    meq = pd.read_csv(MEQ)[["UserID", "meq"]]
    data = emb.merge(lab, on="participant_id").merge(
        meq, left_on="participant_id", right_on="UserID", how="left")
    X = data[feats].to_numpy()
    y = (data["Chronotype"].str.lower() == "evening").astype(int).to_numpy()
    # Cap PCA so it stays valid inside the inner 4-fold CV of the outer LOO:
    # smallest inner-train ~ 0.75 * (n_samples - 1).
    inner_min = int(0.75 * (X.shape[0] - 1)) - 1
    max_pca = max(2, min(X.shape[1], inner_min))
    return X, y, data, max_pca


def nested_auc(emb: pd.DataFrame) -> dict:
    """Fast nested-LOO point estimate (no permutation). Returns AUC/BA + OOF scores."""
    X, y, data, max_pca = _prepare(emb)
    obs = _nested_scores(X, y, max_pca)
    return {"roc_auc": float(roc_auc_score(y, obs)),
            "balanced_accuracy": float(balanced_accuracy_score(y, (obs >= 0.0).astype(int))),
            "oof": pd.DataFrame({"participant_id": data["participant_id"],
                                 "y": y, "score": obs})}


def evaluate(emb: pd.DataFrame, permutations: int = 1000, jobs: int = -1,
             label: str = "embedding") -> dict:
    """emb: DataFrame with participant_id + numeric feature columns."""
    X, y, data, max_pca = _prepare(emb)

    obs = _nested_scores(X, y, max_pca)
    obs_auc = float(roc_auc_score(y, obs))
    obs_ba = float(balanced_accuracy_score(y, (obs >= 0.0).astype(int)))

    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(y) for _ in range(permutations)]
    null = np.array(Parallel(n_jobs=jobs)(
        delayed(lambda yp: float(roc_auc_score(yp, _nested_scores(X, yp, max_pca))))(yp)
        for yp in perms))
    p = float((1 + np.sum(null >= obs_auc)) / (permutations + 1))

    has = data["meq"].notna().to_numpy()
    r, pr = pearsonr(obs[has], data["meq"].to_numpy()[has])

    return {
        "result": {
            "label": label,
            "n_participants": int(len(y)),
            "n_features": int(X.shape[1]),
            "nested_loo_roc_auc": round(obs_auc, 4),
            "nested_loo_balanced_accuracy": round(obs_ba, 4),
            "permutations": permutations,
            "perm_p_value": round(p, 4),
            "null_auc_mean": round(float(null.mean()), 4),
            "null_auc_95pct": round(float(np.percentile(null, 95)), 4),
            "meq_pearson_r": round(float(r), 4),
            "meq_pearson_p": round(float(pr), 4),
        },
        "oof": pd.DataFrame({"participant_id": data["participant_id"],
                             "chronotype_evening": y, "oof_decision_score": obs,
                             "meq": data["meq"]}),
        "null": null,
    }
