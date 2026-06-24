#!/usr/bin/env python3
"""Comprehensive machine-learning analysis of chronotype from EEG/ERP + behaviour.

This implements the full, ISI-style ML workflow used as the paper's primary
analytic framework:

  - leakage-aware scikit-learn pipelines (imputation + scaling fit inside folds);
  - NESTED cross-validation: an outer repeated stratified k-fold estimates
    generalization while an inner k-fold tunes hyperparameters, so no test fold
    ever informs model selection;
  - multiple classifiers compared on identical folds (L2/L1 logistic regression,
    random forest, RBF SVM, histogram gradient boosting);
  - a full metric suite (balanced accuracy, accuracy, ROC AUC, sensitivity,
    specificity, macro F1) with cross-fold uncertainty;
  - out-of-fold predicted probabilities (averaged over repeats) for ROC and
    confusion-matrix reporting;
  - a label-permutation test of the tuned primary model;
  - model coefficients / importances for interpretability.

Outputs go to reports/clean/ml_full/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ID_LIKE = {"participant_id", "Block", "Trial", "global_trial_index", "UserID"}
POSITIVE = "Morning"  # positive class for sensitivity/specificity


def load_xy(path: str, target: str):
    df = pd.read_csv(path)
    df = df[df[target].notna()].reset_index(drop=True)
    y = df[target].astype(str).to_numpy()
    X = df.drop(columns=[c for c in [target, *ID_LIKE] if c in df.columns])
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return X, y, num, cat


def preprocessor(num, cat):
    t = []
    if num:
        t.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat:
        t.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(t)


def model_grids():
    return {
        "logreg_l2": (
            LogisticRegression(penalty="l2", max_iter=5000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "logreg_l1": (
            LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, class_weight="balanced"),
            {"clf__C": [0.05, 0.1, 0.5, 1.0]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1),
            {"clf__n_estimators": [300], "clf__max_depth": [2, 3, None], "clf__min_samples_leaf": [1, 2]},
        ),
        "svm_rbf": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
            {"clf__C": [0.5, 1.0, 5.0], "clf__gamma": ["scale", 0.05]},
        ),
        "hist_gbm": (
            HistGradientBoostingClassifier(random_state=42),
            {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [7, 15]},
        ),
    }


def metrics_row(y_true01, y_pred01, y_prob1):
    tn, fp, fn, tp = confusion_matrix(y_true01, y_pred01, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan       # recall for positive (Morning)
    spec = tn / (tn + fp) if (tn + fp) else np.nan       # recall for negative (Evening)
    row = {
        "balanced_accuracy": balanced_accuracy_score(y_true01, y_pred01),
        "accuracy": accuracy_score(y_true01, y_pred01),
        "sensitivity_morning": sens,
        "specificity_evening": spec,
        "macro_f1": f1_score(y_true01, y_pred01, average="macro"),
    }
    if len(np.unique(y_true01)) == 2:
        row["roc_auc"] = roc_auc_score(y_true01, y_prob1)
    return row


def nested_cv(X, y01, pre, estimator, grid, outer, inner, collect_oof=False):
    rows = []
    n = len(y01)
    oof_prob_sum = np.zeros(n)
    oof_count = np.zeros(n)
    best_params = []
    for tr, te in outer.split(X, y01):
        pipe = Pipeline([("pre", clone(pre)), ("clf", clone(estimator))])
        gs = GridSearchCV(pipe, grid, scoring="balanced_accuracy", cv=inner, n_jobs=-1)
        gs.fit(X.iloc[tr], y01[tr])
        best = gs.best_estimator_
        best_params.append(gs.best_params_)
        prob = best.predict_proba(X.iloc[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(metrics_row(y01[te], pred, prob))
        if collect_oof:
            oof_prob_sum[te] += prob
            oof_count[te] += 1
    df = pd.DataFrame(rows)
    oof = (oof_prob_sum / np.maximum(oof_count, 1)) if collect_oof else None
    return df, oof, best_params


def main() -> None:
    ap = argparse.ArgumentParser(description="Full nested-CV ML analysis for chronotype.")
    ap.add_argument("--data", default="data/clean/chronotype_compact_12.csv")
    ap.add_argument("--target", default="Chronotype")
    ap.add_argument("--primary", default="logreg_l2")
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--outer-repeats", type=int, default=10)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--permutations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="reports/clean/ml_full")
    args = ap.parse_args()

    X, y, num, cat = load_xy(args.data, args.target)
    classes = sorted(np.unique(y))  # ['Evening','Morning'] -> Evening=0, Morning=1
    y01 = (y == POSITIVE).astype(int)
    pre = preprocessor(num, cat)
    grids = model_grids()

    outer = RepeatedStratifiedKFold(n_splits=args.outer_splits, n_repeats=args.outer_repeats, random_state=args.seed)
    inner = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=args.seed)

    # 1) Multi-model comparison under nested CV.
    comparison = []
    oof_primary = None
    primary_params = None
    for name, (est, grid) in grids.items():
        df, oof, bps = nested_cv(X, y01, pre, est, grid, outer, inner, collect_oof=(name == args.primary))
        summary = {"model": name, "n_features": len(num) + len(cat)}
        for col in ["balanced_accuracy", "accuracy", "roc_auc", "sensitivity_morning", "specificity_evening", "macro_f1"]:
            if col in df:
                summary[f"{col}_mean"] = round(float(df[col].mean()), 4)
                summary[f"{col}_sd"] = round(float(df[col].std(ddof=1)), 4)
        comparison.append(summary)
        if name == args.primary:
            oof_primary = oof
            primary_params = bps
        print(f"{name}: BA={summary['balanced_accuracy_mean']:.3f} AUC={summary.get('roc_auc_mean','-')}")

    comp = pd.DataFrame(comparison).sort_values("balanced_accuracy_mean", ascending=False)

    # 2) Out-of-fold confusion + ROC inputs for the primary model.
    pred01 = (oof_primary >= 0.5).astype(int)
    cm = confusion_matrix(y01, pred01, labels=[0, 1])
    oof_df = pd.DataFrame({"y_true": y, "y_true01": y01, "oof_prob_morning": oof_primary, "y_pred": pred01})

    # 3) Permutation test of the tuned primary model (nested inner tuning, single outer pass).
    est, grid = grids[args.primary]
    perm_outer = StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=args.seed)

    def nested_ba(labels):
        df, _, _ = nested_cv(X, labels, pre, est, grid, perm_outer, inner, collect_oof=False)
        return float(df["balanced_accuracy"].mean())

    observed = nested_ba(y01)
    rng = np.random.default_rng(args.seed)
    null = np.array([nested_ba(rng.permutation(y01)) for _ in range(args.permutations)])
    perm_p = float((1 + np.sum(null >= observed)) / (args.permutations + 1))

    # 4) Interpretability: refit tuned primary on all data; pull coefficients.
    gs_full = GridSearchCV(Pipeline([("pre", clone(pre)), ("clf", clone(est))]), grid,
                           scoring="balanced_accuracy", cv=inner, n_jobs=-1).fit(X, y01)
    clf = gs_full.best_estimator_.named_steps["clf"]
    coef_table = None
    if hasattr(clf, "coef_"):
        feat_names = gs_full.best_estimator_.named_steps["pre"].get_feature_names_out()
        coef_table = pd.DataFrame({"feature": feat_names, "coefficient": clf.coef_.ravel()})
        coef_table["abs"] = coef_table["coefficient"].abs()
        coef_table = coef_table.sort_values("abs", ascending=False).drop(columns="abs")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(outdir / "model_comparison.csv", index=False)
    oof_df.to_csv(outdir / "oof_predictions_primary.csv", index=False)
    if coef_table is not None:
        coef_table.to_csv(outdir / "primary_coefficients.csv", index=False)
    result = {
        "data": args.data,
        "primary_model": args.primary,
        "classes": {"negative": classes[0], "positive": POSITIVE},
        "nested_cv": {"outer_splits": args.outer_splits, "outer_repeats": args.outer_repeats, "inner_splits": args.inner_splits},
        "primary_best_params_mode": pd.Series([json.dumps(p) for p in primary_params]).value_counts().idxmax(),
        "confusion_matrix_labels": ["Evening", "Morning"],
        "confusion_matrix": cm.tolist(),
        "permutation": {"observed_ba": round(observed, 4), "null_mean": round(float(null.mean()), 4),
                        "null_p95": round(float(np.quantile(null, 0.95)), 4), "p_value": round(perm_p, 4),
                        "n_permutations": args.permutations},
    }
    (outdir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== Model comparison (nested CV) ===")
    print(comp.to_string(index=False))
    print("\nConfusion matrix [rows=true Evening,Morning; cols=pred Evening,Morning]:")
    print(cm)
    print(f"\nPrimary ({args.primary}) permutation: observed BA={observed:.3f}, p={perm_p:.4f} ({args.permutations} perms)")
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
