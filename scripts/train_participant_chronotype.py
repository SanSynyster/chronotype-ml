#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train participant-level Chronotype classifier with stratified CV (or LOO) on aggregated features.

Input:  data/processed/ml_ready_participant.csv  (≈39 x features)
Output: reports/participant_chronotype/
  - metrics.json (per-model CV means)
  - metrics_per_fold.csv (all folds, all models)
  - confusion_matrix.png (best model)
  - rf_importances.png / rf_importances.csv (if RF is best)
  - roc_pr_curves.png (if binary)
  - model_card.json (config snapshot)
  - best_model.joblib (sklearn Pipeline)
  - best_params.json (if --tune used; best params for best model)

Extras:
  - `--cv-mode {skf, loo}` to switch CV strategy.
  - `--tune` for small GridSearchCV on logreg/rf/svm within folds.
  - `--models` can include `svm`.
"""

import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, LeaveOneOut, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import re
from typing import List
from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
import joblib

DEFAULT_IN = "data/processed/ml_ready_participant.csv"
DEFAULT_OUTDIR = "reports/participant_chronotype"

EEG_PATTERNS = (r"_FRN$", r"_P300$")
BEHAV_SUMMARY_COLS = ("risky_choice_rate", "gain_rate", "loss_rate", "correct_rate")
RESP_SUMMARY_COLS = ("ResponseTime__mean", "ResponseTime__std", "CurrentScore__mean", "CurrentScore__std")
TASK_DESIGN_PATTERNS = (r"^Option[12]$", r"^ActualValue[12]$", r"^Trial$", r"^Block$")

def _filter_columns(cols: List[str], feature_set: str, drop_task: bool) -> List[str]:
    """Return the columns to use for X based on feature_set."""
    def is_eeg(c: str) -> bool:
        return any(re.search(p, c) for p in EEG_PATTERNS)
    def is_beh(c: str) -> bool:
        return c in BEHAV_SUMMARY_COLS
    def is_resp(c: str) -> bool:
        return c in RESP_SUMMARY_COLS
    def is_task_design(c: str) -> bool:
        return any(re.search(p, c) for p in TASK_DESIGN_PATTERNS)

    base = []
    if feature_set == "all":
        base = [c for c in cols]
    elif feature_set == "eeg":
        base = [c for c in cols if is_eeg(c)]
    elif feature_set == "eeg+std":
        # keep all EEG cols including any *_std that slipped in
        base = [c for c in cols if is_eeg(c) or c.endswith("__std")]
    elif feature_set == "beh":
        base = [c for c in cols if is_beh(c)]
    elif feature_set == "beh+resp":
        base = [c for c in cols if is_beh(c) or is_resp(c)]
    elif feature_set == "compact":
        # small, stable set: EEG means + beh summaries (no RT/score)
        base = [c for c in cols if is_eeg(c) or is_beh(c)]
    else:
        base = [c for c in cols]

    if drop_task:
        base = [c for c in base if not is_task_design(c)]

    return base

def build_argparser():
    ap = argparse.ArgumentParser(description="Participant-level Chronotype baseline")
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    ap.add_argument("--outdir", dest="out_dir", default=DEFAULT_OUTDIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", type=int, default=5, help="StratifiedKFold splits")
    ap.add_argument("--models", default="logreg,rf",
                    help="Comma-separated from: logreg,rf,svm")
    ap.add_argument("--cv-mode", default="skf", choices=["skf", "loo"],
                    help="Cross-validation mode: stratified k-fold (skf) or leave-one-out (loo).")
    ap.add_argument("--tune", action="store_true",
                    help="Enable small GridSearchCV hyperparameter tuning within each training fold.")
    ap.add_argument("--feature-set", default="compact",
                    choices=["all","eeg","eeg+std","beh","beh+resp","compact"],
                    help="Which columns to use for X.")
    ap.add_argument("--keep-age-gender", action="store_true",
                    help="Include Age and Gender as features.")
    ap.add_argument("--drop-task-columns", action="store_true",
                    help="Drop task design columns (Option1/2, ActualValue1/2, Trial, Block) if present.")
    return ap

def main():
    args = build_argparser().parse_args()
    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Choose cross-validation splitter
    if args.cv_mode == "loo":
        splitter = LeaveOneOut()
        n_splits = df.shape[0]
    else:
        splitter = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
        n_splits = args.splits

    # --- Define target and feature set ---
    target = "Chronotype"
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not found in {in_path}")

    meta_cols = {"participant_id", target}
    demo_cols = []
    if args.keep_age_gender:
        for d in ("Age", "Gender"):
            if d in df.columns:
                demo_cols.append(d)

    # Candidate pool = all columns minus meta; then filter by feature-set
    candidate_cols = [c for c in df.columns if c not in meta_cols]
    selected_cols = _filter_columns(candidate_cols, args.feature_set, drop_task=args.drop_task_columns)

    # Always append chosen demographics at the end (if requested and present)
    for d in demo_cols:
        if d not in selected_cols:
            selected_cols.append(d)

    X = df[selected_cols].copy()
    y = df[target].astype(str)

    # Split columns by dtype for preprocessing
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Preprocessor: scale numeric, one-hot encode categoricals
    # Handle sklearn version differences for OneHotEncoder arg name
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop"
    )

    # Small hyperparameter grids
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    param_grids = {
        "logreg": {
            "clf__C": [0.1, 1.0, 3.0, 10.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
            "clf__max_iter": [1000],
            "clf__class_weight": ["balanced"],
        },
        "rf": {
            "clf__n_estimators": [300, 600, 1000],
            "clf__max_depth": [None, 6, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", None],
            "clf__class_weight": ["balanced_subsample"],
        },
        "svm": {
            "clf__C": [0.5, 1.0, 3.0, 10.0],
            "clf__gamma": ["scale", 0.1, 0.01],
            "clf__kernel": ["rbf"],
            "clf__class_weight": ["balanced"],
            "clf__probability": [True],
        },
    }

    models = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if name == "logreg":
            models["logreg"] = Pipeline([
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed))
            ])
        elif name == "rf":
            models["rf"] = Pipeline([
                ("pre", pre),
                ("clf", RandomForestClassifier(
                    n_estimators=400, max_depth=None, random_state=args.seed, class_weight="balanced_subsample"))
            ])
        elif name == "svm":
            models["svm"] = Pipeline([
                ("pre", pre),
                ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=args.seed))
            ])
        else:
            raise SystemExit(f"Unknown model '{name}'")

    summary_rows = []
    summary_rows_per_fold = []
    best_name, best_balacc, best_preds, best_truth = None, -1.0, None, None
    best_model, best_feat_names, best_rf_importances = None, None, None

    for mname, pipe in models.items():
        y_true_all, y_pred_all = [], []
        fold_metrics = []

        # Fit once per fold (small N=39)
        for fold, (tr, te) in enumerate(splitter.split(X, y if args.cv_mode != "loo" else None), start=1):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]

            if args.tune:
                grid = param_grids[mname]
                gs = GridSearchCV(
                    estimator=pipe,
                    param_grid=grid,
                    cv=inner_cv,
                    scoring="balanced_accuracy",
                    n_jobs=-1,
                    refit=True,
                    verbose=0,
                )
                gs.fit(Xtr, ytr)
                est = gs.best_estimator_
                fold_best_params = gs.best_params_
            else:
                pipe.fit(Xtr, ytr)
                est = pipe
                fold_best_params = None
            yhat = est.predict(Xte)

            acc = accuracy_score(yte, yhat)
            bal = balanced_accuracy_score(yte, yhat)
            f1m = f1_score(yte, yhat, average="macro")
            prec = precision_score(yte, yhat, average="macro", zero_division=0)
            rec  = recall_score(yte, yhat, average="macro", zero_division=0)

            fold_metrics.append(dict(fold=fold, acc=acc, bal_acc=bal, f1_macro=f1m,
                                     precision_macro=prec, recall_macro=rec))
            y_true_all.extend(yte.tolist())
            y_pred_all.extend(yhat.tolist())

            summary_rows_per_fold.append({
                "model": mname, "fold": fold,
                "acc": acc, "bal_acc": bal, "f1_macro": f1m,
                "precision_macro": prec, "recall_macro": rec,
                "best_params": json.dumps(fold_best_params) if fold_best_params is not None else ""
            })

        # Aggregate CV metrics
        cv_metrics = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
        cv_metrics = {k: float(v) for k, v in cv_metrics.items()}

        # Track best by balanced accuracy
        if cv_metrics["bal_acc"] > best_balacc:
            best_name = mname
            best_balacc = cv_metrics["bal_acc"]
            best_preds = np.array(y_pred_all)
            best_truth = np.array(y_true_all)

            # Refit best on full data (with tuning if requested) for artifacts/export
            try:
                if args.tune:
                    grid = param_grids[mname]
                    gs_full = GridSearchCV(
                        estimator=models[mname],
                        param_grid=grid,
                        cv=inner_cv if args.cv_mode != "loo" else StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed),
                        scoring="balanced_accuracy",
                        n_jobs=-1,
                        refit=True,
                        verbose=0,
                    )
                    gs_full.fit(X, y)
                    best_model = gs_full.best_estimator_
                    # Save tuned best params for the best model
                    (out_dir / "best_params.json").write_text(json.dumps(gs_full.best_params_, indent=2))
                    print("Best tuned params (full-data refit) for best model:", gs_full.best_params_)
                else:
                    best_model = models[mname]
                    best_model.fit(X, y)

                # Recover feature names after ColumnTransformer:
                preproc = best_model.named_steps.get("pre")
                feat_names = None
                try:
                    feat_names = preproc.get_feature_names_out()
                except Exception:
                    feat_names = np.array(num_cols + cat_cols)

                if best_name == "rf":
                    rf = best_model.named_steps["clf"]
                    importances = rf.feature_importances_
                    best_feat_names = list(map(str, feat_names))
                    best_rf_importances = importances
                else:
                    best_feat_names, best_rf_importances = None, None
            except Exception:
                pass

        summary_rows.append(dict(model=mname, **cv_metrics))

    # Save summary metrics
    summary_df = pd.DataFrame(summary_rows).sort_values("bal_acc", ascending=False)
    summary_json = summary_df.to_dict(orient="records")
    (out_dir / "metrics.json").write_text(json.dumps(summary_json, indent=2))

    # Save per-fold metrics
    pf_path = out_dir / "metrics_per_fold.csv"
    pd.DataFrame(summary_rows_per_fold).to_csv(pf_path, index=False)

    print("=== Participant-level Chronotype CV (mean over folds) ===")
    print(summary_df.to_string(index=False))
    # If tuning was enabled, remind user where best params were saved
    if args.tune and (out_dir / "best_params.json").exists():
        print(f"Best tuned params saved to: {out_dir / 'best_params.json'}")

    # Confusion matrix for the best model
    labels = sorted(pd.unique(y).tolist())
    cm = confusion_matrix(best_truth, best_preds, labels=labels)

    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix — best={best_name}")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    fig_path = out_dir / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    # ROC/PR if binary
    if len(labels) == 2:
        try:
            # Refit best on full data to get probabilities
            best_model.fit(X, y)
            y_proba = best_model.predict_proba(X)[:, list(best_model.classes_).index(labels[1])]
            # Plot curves
            fig = plt.figure()
            RocCurveDisplay.from_predictions((y == labels[1]).astype(int), y_proba)
            plt.title("ROC Curve (best model)")
            plt.tight_layout()
            roc_path = out_dir / "roc_pr_curves.png"
            fig.savefig(roc_path, dpi=160)
            plt.close(fig)

            fig = plt.figure()
            PrecisionRecallDisplay.from_predictions((y == labels[1]).astype(int), y_proba)
            plt.title("Precision-Recall Curve (best model)")
            plt.tight_layout()
            fig.savefig(out_dir / "pr_curve.png", dpi=160)
            plt.close(fig)
        except Exception:
            pass

    # RF importances if available
    if best_rf_importances is not None and best_feat_names is not None:
        imp = pd.DataFrame({
            "feature": best_feat_names,
            "importance": best_rf_importances
        }).sort_values("importance", ascending=False)
        imp.to_csv(out_dir / "rf_importances.csv", index=False)

        fig = plt.figure()
        top = imp.head(20)[::-1]  # plot top 20
        plt.barh(top["feature"], top["importance"])
        plt.title("Random Forest Importances (top 20)")
        plt.tight_layout()
        fig.savefig(out_dir / "rf_importances.png", dpi=160)
        plt.close(fig)

    try:
        joblib.dump(best_model, out_dir / "best_model.joblib")
    except Exception:
        pass

    # Model card
    card = {
        "n_participants": int(df.shape[0]),
        "features_used": selected_cols,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "cv_splits": args.splits,
        "seed": args.seed,
        "metrics": summary_json,
        "best_model": best_name,
        "confusion_matrix_labels": labels,
        "feature_set": args.feature_set,
        "keep_age_gender": bool(args.keep_age_gender),
        "drop_task_columns": bool(args.drop_task_columns),
        "selected_feature_columns": selected_cols,
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))

if __name__ == "__main__":
    main()