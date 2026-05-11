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

from sklearn.model_selection import StratifiedKFold, LeaveOneOut, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import warnings
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
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

# Safe f_classif that handles NaN/Inf produced when a feature is constant within a class
def f_classif_safe(X, y):
    F, p = f_classif(X, y)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    return F, p

# A SelectKBest that automatically uses 'all' if k >= n_features for the current fold
from sklearn.feature_selection import SelectKBest as _SelectKBest
class SafeKBest(_SelectKBest):
    def __init__(self, score_func=f_classif_safe, k=10):
        super().__init__(score_func=score_func, k=k)
    def fit(self, X, y=None, **fit_params):
        original_k = self.k
        # If requested k is >= number of columns after preprocessing in this fold, just keep all
        if isinstance(original_k, int) and original_k >= X.shape[1]:
            self.k = 'all'
        try:
            return super().fit(X, y, **fit_params)
        finally:
            # restore original k so repr/params stay stable
            self.k = original_k

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
    ap.add_argument("--repeats", type=int, default=1, help="If >1, use RepeatedStratifiedKFold with this many repeats")
    ap.add_argument("--models", default="logreg,rf",
                    help="Comma-separated from: logreg,rf,svm,hgb,stack")
    ap.add_argument('--cv-mode', choices=['skf', 'sgkf', 'loo'], default='skf',
                    help='skf=StratifiedKFold, sgkf=StratifiedGroupKFold (group-aware), loo=LeaveOneOut')
    ap.add_argument('--group-col', type=str, default='participant_id',
                    help='Grouping column (e.g., participant_id) for group-aware CV')
    ap.add_argument("--tune", action="store_true",
                    help="Enable small GridSearchCV hyperparameter tuning within each training fold.")
    ap.add_argument("--feature-set", default="compact",
                    choices=["all","eeg","eeg+std","beh","beh+resp","compact"],
                    help="Which columns to use for X.")
    ap.add_argument("--topk", type=int, default=0, help="If >0, apply SelectKBest(f_classif) after preprocessing to keep top-K features (performed inside each CV fold to avoid leakage).")
    ap.add_argument("--keep-age-gender", action="store_true",
                    help="Include Age and Gender as features.")
    ap.add_argument("--drop-task-columns", action="store_true",
                    help="Drop task design columns (Option1/2, ActualValue1/2, Trial, Block) if present.")
    ap.add_argument("--fast-tune", action="store_true", help="Use smaller grids and 2-fold inner CV for faster tuning.")
    return ap

def main():
    args = build_argparser().parse_args()
    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Prepare groups vector early so it's available for CV splitter selection
    groups = df[args.group_col].to_numpy() if args.group_col in df.columns else None

    # Choose cross-validation splitter
    if args.cv_mode == "loo":
        splitter = LeaveOneOut()
        total_folds = df.shape[0]
    elif args.cv_mode == "sgkf":
        # Group-aware stratified K-fold (requires groups)
        if args.group_col not in df.columns:
            raise SystemExit(f"--cv-mode sgkf requires a grouping column present in the data; '{args.group_col}' not found.")
        splitter = StratifiedGroupKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
        total_folds = args.splits if not (args.repeats and args.repeats > 1) else (args.splits * args.repeats)
    else:  # skf
        if args.repeats and args.repeats > 1:
            splitter = RepeatedStratifiedKFold(n_splits=args.splits, n_repeats=args.repeats, random_state=args.seed)
            total_folds = args.splits * args.repeats
        else:
            splitter = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
            total_folds = args.splits

    # --- Define target and feature set ---
    target = "Chronotype"
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not found in {in_path}")

    meta_cols = {"participant_id", target}
    # Demographic columns to optionally keep
    demo_cols = []
    if args.keep_age_gender:
        for cand in ["Age", "age", "ParticipantAge", "Gender", "gender", "Sex", "sex"]:
            if cand in df.columns:
                demo_cols.append(cand)

    # Force-include extra demographic/trait columns if present (optional)
    # Add/rename to match your actual column names
    for cand in ["MEQ_total", "MCTQ_midSleep", "MCTQ_socialJetlag"]:
        if cand in df.columns:
            demo_cols.append(cand)

    # Candidate pool = all columns minus meta; then filter by feature-set
    candidate_cols = [c for c in df.columns if c not in meta_cols]
    selected_cols = _filter_columns(candidate_cols, args.feature_set, drop_task=args.drop_task_columns)

    # Always append chosen demographics at the end (if requested and present)
    for d in demo_cols:
        if d in df.columns and d not in selected_cols:
            selected_cols.append(d)

    # Safety: intersect with actual columns to avoid KeyError
    missing = [c for c in selected_cols if c not in df.columns]
    if missing:
        print(f"[warn] Dropping {len(missing)} missing columns from selected feature list: {missing}")
    filtered_selected_cols = [c for c in selected_cols if c in df.columns]

    X = df[filtered_selected_cols].copy()
    y = df[target].astype(str)

    # Group vector for group-aware CV
    if args.group_col in df.columns:
        groups = df[args.group_col].to_numpy()
    else:
        groups = None



    # Identify final demographic columns chosen above
    chosen_demo = [c for c in demo_cols if c in df.columns]

    # Split demo into numeric vs categorical using df dtypes
    demo_num = [c for c in chosen_demo if pd.api.types.is_numeric_dtype(df[c])]
    demo_cat = [c for c in chosen_demo if not pd.api.types.is_numeric_dtype(df[c])]

    # Split numeric/categorical feature lists to exclude demographics
    num_all = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_all = [c for c in X.columns if c not in num_all]
    num_cols = [c for c in num_all if c not in chosen_demo]
    cat_cols = [c for c in cat_all if c not in chosen_demo]

    # Pipeline for numeric (non-demo) features
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    if args.topk and args.topk > 0:
        # Apply SafeKBest ONLY to numeric (non-demo) branch
        num_pipe.steps.append(("kbest", SafeKBest(score_func=f_classif_safe, k=args.topk)))

    # OneHotEncoder with sklearn version compatibility
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", ohe, cat_cols),
            # demographics: encode categorical demos, passthrough numeric demos
            ("demo_cat", ohe, demo_cat),
            ("demo_num", "passthrough", demo_num),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Small hyperparameter grids
    inner_cv = StratifiedKFold(n_splits=(2 if args.fast_tune else 3), shuffle=True, random_state=args.seed)
    warnings.filterwarnings("ignore", message=r".*Features .* are constant.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"k=\d+ is greater than n_features=\d+.*", category=UserWarning)
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
        }
        ,
        "hgb": {
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_leaf_nodes": [15, 31],
            "clf__min_samples_leaf": [10, 20],
            "clf__max_depth": [None, 6]
        }
    }
    if args.fast_tune:
        if "rf" in param_grids:
            param_grids["rf"] = {
                "clf__n_estimators": [600, 1000],
                "clf__max_depth": [None, 6],
                "clf__min_samples_leaf": [1, 2],
                "clf__max_features": ["sqrt"],
                "clf__class_weight": ["balanced_subsample"],
            }
        if "logreg" in param_grids:
            param_grids["logreg"] = {
                "clf__C": [0.5, 1.0, 3.0],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs"],
                "clf__max_iter": [1000],
                "clf__class_weight": ["balanced"],
            }
        if "svm" in param_grids:
            param_grids["svm"] = {
                "clf__C": [0.5, 1.0, 3.0],
                "clf__gamma": ["scale", 0.01],
                "clf__kernel": ["rbf"],
                "clf__class_weight": ["balanced"],
                "clf__probability": [True],
            }
        if "hgb" in param_grids:
            param_grids["hgb"] = {
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_leaf_nodes": [31],
                "clf__min_samples_leaf": [10, 20],
                "clf__max_depth": [None],
            }

    # Helper to build pipeline with optional SelectKBest
    def make_pipe(clf):
        steps = [("pre", pre)]
        # Drop constant features per fold (after preprocessing) to avoid zero within-class variance
        steps.append(("var", VarianceThreshold(threshold=0.0)))
        steps.append(("clf", clf))
        return Pipeline(steps)

    models = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if name == "logreg":
            models["logreg"] = make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed))
        elif name == "rf":
            models["rf"] = make_pipe(RandomForestClassifier(
                n_estimators=400, max_depth=None, random_state=args.seed, class_weight="balanced_subsample"))
        elif name == "svm":
            models["svm"] = make_pipe(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=args.seed))
        elif name == "hgb":
            models["hgb"] = make_pipe(HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.1,
                max_depth=None,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=args.seed,
            ))
        elif name == "stack":
            # Stacking: RF + Logistic, meta Logistic; no tuning initially
            base_rf = RandomForestClassifier(
                n_estimators=600, max_features="sqrt",
                class_weight="balanced_subsample", random_state=args.seed)
            base_log = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=args.seed)
            stacker = StackingClassifier(
                estimators=[("rf", base_rf), ("log", base_log)],
                final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=args.seed),
                passthrough=False,
                n_jobs=-1,
                stack_method="predict_proba",
            )
            models["stack"] = make_pipe(stacker)
        else:
            raise SystemExit(f"Unknown model '{name}'")

    summary_rows = []
    summary_rows_per_fold = []
    best_name, best_balacc, best_preds, best_truth = None, -1.0, None, None
    best_model, best_feat_names, best_rf_importances = None, None, None

    print(f"Models to run: {list(models.keys())}")
    for mname, pipe in models.items():
        y_true_all, y_pred_all = [], []
        fold_metrics = []

        # Fit once per fold (small N=39)
        if args.cv_mode == "sgkf":
            split_iter = splitter.split(X, y, groups)
        elif args.cv_mode == "loo":
            split_iter = splitter.split(X)
        else:
            split_iter = splitter.split(X, y)

        for fold, (tr, te) in enumerate(split_iter, start=1):
            print(f"[{mname}] Fold {fold}/{total_folds} — training...", flush=True)
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
            print(f"[{mname}] Fold {fold} — fit complete.", flush=True)
            yhat = est.predict(Xte)

            acc = accuracy_score(yte, yhat)
            bal = balanced_accuracy_score(yte, yhat)
            f1m = f1_score(yte, yhat, average="macro")
            prec = precision_score(yte, yhat, average="macro", zero_division=0)
            rec  = recall_score(yte, yhat, average="macro", zero_division=0)
            print(f"[{mname}] Fold {fold} — bal_acc={bal:.3f}, f1={f1m:.3f}", flush=True)

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
        print(f"[{mname}] CV summary — bal_acc={cv_metrics['bal_acc']:.3f}, acc={cv_metrics['acc']:.3f}, f1={cv_metrics['f1_macro']:.3f}", flush=True)

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

                # Recover feature names after preprocessing, and after optional selector
                preproc = best_model.named_steps.get("pre")
                sel = best_model.named_steps.get("sel")

                # Names after the ColumnTransformer
                try:
                    feat_after_pre = preproc.get_feature_names_out()
                except Exception:
                    feat_after_pre = np.array(num_cols + cat_cols, dtype=str)

                # If a selector exists, reduce names by its support mask
                if sel is not None and hasattr(sel, "get_support"):
                    try:
                        mask = sel.get_support()
                        if len(mask) == len(feat_after_pre):
                            feat_after_sel = np.array(feat_after_pre)[mask]
                        else:
                            # Fallback if shapes don't line up
                            feat_after_sel = np.array([f"f{i}" for i in range(np.sum(mask))], dtype=str)
                    except Exception:
                        feat_after_sel = np.array(feat_after_pre, dtype=str)
                else:
                    feat_after_sel = np.array(feat_after_pre, dtype=str)

                if best_name == "rf":
                    rf = best_model.named_steps["clf"]
                    importances = rf.feature_importances_
                    # Final safety: match lengths exactly
                    if len(importances) != len(feat_after_sel):
                        feat_names_final = np.array([f"f{i}" for i in range(len(importances))], dtype=str)
                    else:
                        feat_names_final = feat_after_sel
                    best_feat_names = list(map(str, feat_names_final))
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
    if args.topk and args.topk > 0: print(f"Feature selection: SelectKBest(f_classif, k={args.topk}) applied inside CV.")
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