#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose and understand the participant-level chronotype model.

Outputs (under --report-dir):
  - diag_classification_report.txt
  - diag_confusion_matrix.png
  - diag_feature_importances.csv / diag_feature_importances.png (if tree-based)
  - diag_permutation_importances.csv / diag_permutation_importances.png
  - diag_probability_hist.png (if predict_proba available)
  - diag_cv_summary.txt (if metrics_per_fold.csv exists)

Usage:
  python scripts/diagnose_model.py \
    --in data/processed/ml_ready_participant.csv \
    --report-dir reports/participant_chronotype
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.inspection import permutation_importance

def load_best_model(report_dir: Path):
    model_path = report_dir / "best_model.joblib"
    if not model_path.exists():
        raise SystemExit(f"[error] best_model.joblib not found in {report_dir}")
    return joblib.load(model_path)

def get_feature_names_from_pipeline(pipeline):
    """Best-effort recovery of final feature names after preprocessing."""
    pre = pipeline.named_steps.get("pre")
    if pre is None:
        return None
    try:
        names = pre.get_feature_names_out()
        return np.array(names, dtype=str)
    except Exception:
        # Fallback if not available
        return None

def save_confusion_matrix(y_true, y_pred, labels, out_path: Path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def save_barh(values, names, out_path: Path, title: str, top=30):
    order = np.argsort(values)[::-1]
    names = np.array(names, dtype=str)[order][:top][::-1]
    vals = np.array(values)[order][:top][::-1]
    fig = plt.figure()
    plt.barh(names, vals)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def summarize_cv_metrics(report_dir: Path):
    pf = report_dir / "metrics_per_fold.csv"
    if not pf.exists():
        return None
    df = pd.read_csv(pf)
    agg = (
        df.groupby("model")[["acc", "bal_acc", "f1_macro", "precision_macro", "recall_macro"]]
        .agg(["mean", "std"])
        .round(4)
    )
    txt = agg.to_string()
    (report_dir / "diag_cv_summary.txt").write_text(txt + "\n")
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/processed/ml_ready_participant.csv")
    ap.add_argument("--report-dir", dest="report_dir", default="reports/participant_chronotype")
    ap.add_argument("--target", default="Chronotype")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load data and artifacts
    if not in_path.exists():
        raise SystemExit(f"[error] data not found: {in_path}")
    df = pd.read_csv(in_path)
    if args.target not in df.columns:
        raise SystemExit(f"[error] target '{args.target}' not in {in_path}")

    model = load_best_model(report_dir)

    # Try to load model card (for metadata)
    model_card = {}
    mc_path = report_dir / "model_card.json"
    if mc_path.exists():
        with open(mc_path, "r") as f:
            model_card = json.load(f)

    y = df[args.target].astype(str)
    X_cols = [c for c in df.columns if c not in {args.target, "participant_id"}]
    # If the training script stored selected cols, prefer them (for stability)
    if "selected_feature_columns" in model_card:
        sel = [c for c in model_card["selected_feature_columns"] if c in df.columns]
        if sel:
            X_cols = sel

    X = df[X_cols].copy()

    # Basic sanity
    classes = sorted(pd.unique(y).tolist())

    # Predictions on full data (diagnostics only)
    y_pred = model.predict(X)
    bal = balanced_accuracy_score(y, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)

    # Classification report
    cls_rep = classification_report(y, y_pred, labels=classes, zero_division=0)
    # Class balance
    class_counts = y.value_counts().to_string()

    out_txt = []
    out_txt.append("=== Diagnostic Evaluation on Full Data (not CV) ===")
    out_txt.append(f"Balanced Accuracy: {bal:.4f}")
    out_txt.append(f"Precision (macro): {pr:.4f}")
    out_txt.append(f"Recall (macro):    {rc:.4f}")
    out_txt.append(f"F1 (macro):        {f1:.4f}")
    out_txt.append("")
    out_txt.append("=== Class Distribution (y) ===")
    out_txt.append(class_counts)
    out_txt.append("")
    out_txt.append("=== Classification Report ===")
    out_txt.append(cls_rep)

    (report_dir / "diag_classification_report.txt").write_text("\n".join(out_txt) + "\n")
    print("\n".join(out_txt))

    # Confusion matrix
    save_confusion_matrix(
        y_true=y, y_pred=y_pred, labels=classes,
        out_path=report_dir / "diag_confusion_matrix.png",
        title=f"Confusion Matrix (full-data) — best model"
    )

    # Probability histograms / calibration (if available)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            # For multi-class, plot each class probability histogram
            fig = plt.figure(figsize=(8, 4 + 1.8 * len(classes)))
            nrows = len(classes)
            for i, c in enumerate(classes, 1):
                plt.subplot(nrows, 1, i)
                plt.hist(proba[:, i - 1 if set(model.classes_) == set(classes) else list(model.classes_).index(c)], bins=15, alpha=0.8)
                plt.title(f"Predicted probability for class '{c}'")
                plt.xlabel("p(class)")
                plt.ylabel("count")
            plt.tight_layout()
            fig.savefig(report_dir / "diag_probability_hist.png", dpi=160)
            plt.close(fig)
        except Exception:
            warnings.warn("Could not produce probability histograms.", RuntimeWarning)

    # Feature importances (tree-based)
    feat_names = get_feature_names_from_pipeline(model)
    if feat_names is None:
        # Prefer actual dataframe column names if available
        try:
            feat_names = np.array(X.columns, dtype=str)
        except Exception:
            n_feats = getattr(model.named_steps.get("clf"), "n_features_in_", None)
            if n_feats is None:
                n_feats = X.shape[1]
            feat_names = np.array([f"f{i}" for i in range(int(n_feats))], dtype=str)

    clf = model.named_steps.get("clf", None)
    if clf is not None and hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        # Align lengths safely
        if len(importances) != len(feat_names):
            feat_names = np.array([f"f{i}" for i in range(len(importances))], dtype=str)
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}) \
                 .sort_values("importance", ascending=False)
        imp_df.to_csv(report_dir / "diag_feature_importances.csv", index=False)
        save_barh(
            values=imp_df["importance"].values,
            names=imp_df["feature"].values,
            out_path=report_dir / "diag_feature_importances.png",
            title="Model Feature Importances (tree-based)",
            top=30
        )

    # Permutation importance (model-agnostic)
    try:
        # Small number of repeats to keep it quick
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm = permutation_importance(
                model, X, y, scoring="balanced_accuracy", n_repeats=10, random_state=42, n_jobs=-1
            )
        perm_means = perm.importances_mean
        # Align safely with names
        if len(perm_means) != len(feat_names):
            feat_names_pi = np.array([f"f{i}" for i in range(len(perm_means))], dtype=str)
        else:
            feat_names_pi = feat_names
        pi_df = pd.DataFrame({"feature": feat_names_pi, "perm_importance_mean": perm_means}) \
                .sort_values("perm_importance_mean", ascending=False)
        pi_df.to_csv(report_dir / "diag_permutation_importances.csv", index=False)
        save_barh(
            values=pi_df["perm_importance_mean"].values,
            names=pi_df["feature"].values,
            out_path=report_dir / "diag_permutation_importances.png",
            title="Permutation Importances (balanced accuracy)",
            top=30
        )
    except Exception as e:
        warnings.warn(f"Permutation importance failed: {e}", RuntimeWarning)

    # Summarize CV metrics from training (if present)
    agg = summarize_cv_metrics(report_dir)
    if agg is not None:
        print("\n=== Cross-Validation Summary (from metrics_per_fold.csv) ===")
        print(agg)

    print(f"\n[done] Diagnostics written to: {report_dir}")

if __name__ == "__main__":
    main()