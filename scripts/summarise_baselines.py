#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, glob, os
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.inspection import permutation_importance
import warnings
def find_combo_dir(target: str, pack: str, split: str) -> Path:
    return IN_DIR / f"{target}__{pack}__{split}"

def plot_confusion_matrix_for_best(target: str, pack: str, split: str, out_dir: Path) -> None:
    """
    Load predictions.csv for the best classification combo and save a confusion matrix PNG.
    Robust to mixed dtypes (e.g., str vs float) by coercing to string labels.
    """
    combo_dir = find_combo_dir(target, pack, split)
    pred_path = combo_dir / "predictions.csv"
    if not pred_path.exists():
        print(f"⚠️  No predictions.csv for {target}__{pack}__{split}; skipping CM.")
        return
    try:
        dfp = pd.read_csv(pred_path)

        # Determine true/pred columns, fall back to first two
        y_true = dfp["y_true"] if "y_true" in dfp.columns else dfp.iloc[:, 0]
        y_pred = dfp["y_pred"] if "y_pred" in dfp.columns else dfp.iloc[:, 1]

        # Coerce to string labels and handle NaNs explicitly to avoid mixed-type comparisons
        y_true = y_true.fillna("NA").astype(str)
        y_pred = y_pred.fillna("NA").astype(str)

        # Build label set preserving appearance order across y_true then y_pred
        labels = pd.unique(pd.concat([y_true, y_pred], ignore_index=True))

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{target} — {pack} ({split})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f"{target}__{pack}__{split}__confusion_matrix.png"
        fig.savefig(png)
        plt.close(fig)
        print(f"🖼️  Saved {png}")
    except Exception as e:
        print(f"⚠️  Failed CM for {target}__{pack}__{split}: {e}")

def plot_feature_importance_for_best(target: str, pack: str, split: str, metrics_path: str, out_dir: Path) -> None:
    """
    Load saved RandomForest model (if available) and plot top-15 feature importances.
    Falls back to metrics.json['feature_names'] ordering if model is missing.
    """
    combo_dir = find_combo_dir(target, pack, split)
    rf_path = combo_dir.parent.parent.parent / "models" / "ml_baselines" / f"{target}__{pack}__{split}" / "random_forest.joblib"
    # NOTE: older train script saved models alongside reports — try that location too
    alt_rf_path = find_combo_dir(target, pack, split) / "random_forest.joblib"

    # Load feature names from metrics.json
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        feat_names = metrics.get("feature_names", [])
    except Exception:
        feat_names = []

    model = None
    for pth in [rf_path, alt_rf_path]:
        if pth.exists():
            try:
                model = joblib.load(pth)
                break
            except Exception:
                pass

    if model is None or not hasattr(model, "feature_importances_"):
        print(f"⚠️  RandomForest model with importances not found for {target}__{pack}__{split}; skipping FI.")
        return

    importances = np.asarray(model.feature_importances_)
    if len(feat_names) != len(importances):
        # best effort names
        feat_names = feat_names if feat_names and len(feat_names) == len(importances) else [f"f{i}" for i in range(len(importances))]

    order = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.barh(range(len(order)), importances[order][::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-15 RF importances — {target} / {pack} ({split})")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{target}__{pack}__{split}__rf_importances.png"
    fig.savefig(png)
    plt.close(fig)
    print(f"🖼️  Saved {png}")


# ---- Additional helper functions ----

def _load_rf_model_and_feats(target: str, pack: str, split: str, metrics_path: str):
    """Return (model, feature_names) if available, else (None, [])."""
    combo_dir = find_combo_dir(target, pack, split)
    rf_path = combo_dir.parent.parent.parent / "models" / "ml_baselines" / f"{target}__{pack}__{split}" / "random_forest.joblib"
    alt_rf_path = combo_dir / "random_forest.joblib"
    feat_names = []
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        feat_names = metrics.get("feature_names", []) or []
    except Exception:
        pass
    model = None
    for pth in [rf_path, alt_rf_path]:
        if pth.exists():
            try:
                model = joblib.load(pth)
                break
            except Exception:
                continue
    return model, feat_names


def compute_per_class_metrics(target: str, pack: str, split: str, out_dir: Path) -> None:
    """
    Load predictions.csv for a classification winner and emit per-class metrics CSV
    (precision, recall, f1, support) and a compact text report.
    """
    combo_dir = find_combo_dir(target, pack, split)
    pred_path = combo_dir / "predictions.csv"
    if not pred_path.exists():
        print(f"⚠️  No predictions.csv for {target}__{pack}__{split}; skipping per-class metrics.")
        return
    try:
        dfp = pd.read_csv(pred_path)
        y_true = dfp["y_true"] if "y_true" in dfp.columns else dfp.iloc[:, 0]
        y_pred = dfp["y_pred"] if "y_pred" in dfp.columns else dfp.iloc[:, 1]
        y_true = y_true.fillna("NA").astype(str)
        y_pred = y_pred.fillna("NA").astype(str)
        labels = pd.unique(pd.concat([y_true, y_pred], ignore_index=True))
        prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        per_cls = pd.DataFrame({
            "label": labels,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sup,
        })
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{target}__{pack}__{split}__per_class_metrics.csv"
        per_cls.to_csv(csv_path, index=False)
        # also dump a sklearn-style text report for convenience
        txt = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        with open(out_dir / f"{target}__{pack}__{split}__classification_report.txt", "w") as f:
            f.write(txt)
        print(f"📄  Saved per-class metrics → {csv_path}")
    except Exception as e:
        print(f"⚠️  Failed per-class metrics for {target}__{pack}__{split}: {e}")


def plot_permutation_importance(target: str, pack: str, split: str, metrics_path: str, out_dir: Path, task: str) -> None:
    """
    Compute and plot permutation importances on the test split using the saved RF model.
    Requires ml_subsets test CSV and feature_names in metrics.json.
    Fix: pass the true target vector y to permutation_importance (was None).
    """
    model, feat_names = _load_rf_model_and_feats(target, pack, split, metrics_path)
    if model is None or not feat_names:
        print(f"⚠️  Cannot compute PI for {target}__{pack}__{split}; missing model or feature names.")
        return

    subset_dir = Path("data/processed/ml_subsets")
    test_csv = subset_dir / f"{target}__{pack}__{split}__test.csv"
    if not test_csv.exists():
        print(f"⚠️  Test subset not found for PI: {test_csv}")
        return

    try:
        df = pd.read_csv(test_csv)

        # Ensure all features exist; intersect just in case
        missing = [c for c in feat_names if c not in df.columns]
        if missing:
            print(f"⚠️  Skipping PI for {target}__{pack}__{split}: missing features {missing[:5]}{'...' if len(missing) > 5 else ''}")
            return

        X = df[feat_names].copy()

        # Load y from the target column in the subset; coerce to appropriate dtype
        if target not in df.columns:
            print(f"⚠️  Skipping PI for {target}__{pack}__{split}: target column '{target}' not in test subset.")
            return
        y = df[target].copy()

        # Drop rows with any NaNs in X or y (keep alignment)
        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

        # Align target dtype to what the classifier/regressor expects.
        if task == "classification":
            # Coerce y to string so it matches estimators that learned string classes ('0.0','1.0', etc.).
            y = y.astype(str)
        else:
            # regression
            y = pd.to_numeric(y, errors="coerce")
            m2 = y.notna()
            X, y = X.loc[m2], y.loc[m2]

        if len(X) == 0:
            print(f"⚠️  Skipping PI for {target}__{pack}__{split}: no valid rows after NaN filtering.")
            return

        scoring = "accuracy" if task == "classification" else "r2"

        # Ensure column order matches what the model saw during training if available
        if hasattr(model, "feature_names_in_"):
            missing_from_X = [c for c in model.feature_names_in_ if c not in X.columns]
            if missing_from_X:
                print(f"⚠️  Skipping PI for {target}__{pack}__{split}: test set missing trained features: {missing_from_X[:5]}{'...' if len(missing_from_X) > 5 else ''}")
                return
            X = X.loc[:, list(model.feature_names_in_)]

        pi = permutation_importance(
            model,
            X,
            y,
            scoring=scoring,
            n_repeats=10,
            random_state=42,
        )
        importances = pi.importances_mean
        order = np.argsort(importances)[::-1][:15]
        names = [feat_names[i] for i in order]
        vals = importances[order]

        out_dir.mkdir(parents=True, exist_ok=True)
        pi_csv = out_dir / f"{target}__{pack}__{split}__perm_importances.csv"
        pd.DataFrame({"feature": names, "perm_importance": vals}).to_csv(pi_csv, index=False)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.barh(range(len(order)), vals[::-1])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(names[::-1])
        ax.set_xlabel("Permutation importance")
        ax.set_title(f"Top-15 Permutation Importances — {target} / {pack} ({split})")
        fig.tight_layout()
        png = out_dir / f"{target}__{pack}__{split}__perm_importances.png"
        fig.savefig(png)
        plt.close(fig)
        print(f"🖼️  Saved {png}")
    except Exception as e:
        print(f"⚠️  Failed permutation importances for {target}__{pack}__{split}: {e}")


def write_model_card(target: str, pack: str, split: str, task: str, metrics_row: dict) -> None:
    """Write a compact model card Markdown for the winner combo of a target."""
    mc_dir = OUT_DIR / "model_cards"
    mc_dir.mkdir(parents=True, exist_ok=True)
    cm_png = OUT_DIR / "figs" / "confusion_matrices" / f"{target}__{pack}__{split}__confusion_matrix.png"
    rf_png = OUT_DIR / "figs" / "feature_importances" / f"{target}__{pack}__{split}__rf_importances.png"
    pi_png = OUT_DIR / "figs" / "feature_importances" / f"{target}__{pack}__{split}__perm_importances.png"

    lines = [
        f"# Model Card — {target}",
        "",
        f"**Winner:** `{pack}` split by `{split}`  ",
        f"**Task:** {task}",
        "",
    ]
    if task == "classification":
        lines += [
            f"- Balanced Acc: {metrics_row.get('RF_balanced_accuracy')}",
            f"- Macro F1: {metrics_row.get('RF_macro_f1')}",
            f"- Accuracy: {metrics_row.get('RF_accuracy')}",
        ]
    else:
        lines += [
            f"- RMSE: {metrics_row.get('RF_rmse')}",
            f"- MAE: {metrics_row.get('RF_mae')}",
            f"- R²: {metrics_row.get('RF_r2')}",
        ]
    lines += [
        f"- n_features: {metrics_row.get('n_features')}",
        "",
        "## Diagnostics",
    ]
    if cm_png.exists():
        lines += [f"![Confusion Matrix]({cm_png})", ""]
    if rf_png.exists():
        lines += [f"![RF Importances]({rf_png})", ""]
    if pi_png.exists():
        lines += [f"![Permutation Importances]({pi_png})", ""]

    path = mc_dir / f"{target}__{pack}__{split}.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"🪪  Wrote model card → {path}")

def generate_winner_plots(leaderboard_df: pd.DataFrame) -> None:
    """
    For each target's best combo, emit confusion matrix, per-class metrics, RF feature-importance,
    permutation importance plots, and model card.
    """
    cm_dir = OUT_DIR / "figs" / "confusion_matrices"
    fi_dir = OUT_DIR / "figs" / "feature_importances"
    pcm_dir = OUT_DIR / "per_class_metrics"
    for _, r in leaderboard_df.iterrows():
        target = r["target"]
        pack = r["best_pack"]
        split = r["best_split"]
        metrics_path = r["metrics_path"]
        task = r["task"]
        if task == "classification":
            plot_confusion_matrix_for_best(target, pack, split, cm_dir)
            compute_per_class_metrics(target, pack, split, pcm_dir)
        # Always try to plot RF importances
        plot_feature_importance_for_best(target, pack, split, metrics_path, fi_dir)
        # Also try permutation importances (needs model + subset)
        plot_permutation_importance(target, pack, split, metrics_path, fi_dir, task)
        # Write a compact model card
        write_model_card(target, pack, split, task, r.to_dict())
from collections import defaultdict

IN_DIR = Path("reports/ml_baselines")
OUT_DIR = Path("reports/ml_baselines_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLS_TARGETS = {"risky-choice", "behav_valence", "Chronotype"}

def load_metrics():
    rows = []
    for p in sorted(glob.glob(str(IN_DIR / "*__*__*/metrics.json"))):
        with open(p, "r") as f:
            m = json.load(f)
        # infer target/pack/split from folder name
        tag = Path(p).parent.name  # e.g., risky-choice__eeg_only__participant
        target, pack, split = tag.split("__", 2)

        rec = {
            "target": target,
            "pack": pack,
            "split": split,
            "n_features": m.get("n_features"),
            "n_feat_names": len(m.get("feature_names", [])),
            "metrics_path": p,
        }

        if target in CLS_TARGETS:
            # pull classification metrics
            lr = m.get("logistic_regression", {})
            rf = m.get("random_forest", {})
            rec.update({
                "task": "classification",
                "LR_accuracy": lr.get("accuracy"),
                "LR_balanced_accuracy": lr.get("balanced_accuracy"),
                "LR_macro_f1": lr.get("macro_f1"),
                "LR_roc_auc": lr.get("roc_auc"),
                "RF_accuracy": rf.get("accuracy"),
                "RF_balanced_accuracy": rf.get("balanced_accuracy"),
                "RF_macro_f1": rf.get("macro_f1"),
                "RF_roc_auc": rf.get("roc_auc"),
            })
        else:
            # regression
            lin = m.get("linear_regression", {})
            rf = m.get("random_forest", {})
            rec.update({
                "task": "regression",
                "LIN_mae": lin.get("mae"),
                "LIN_rmse": lin.get("rmse"),
                "LIN_r2": lin.get("r2"),
                "RF_mae": rf.get("mae"),
                "RF_rmse": rf.get("rmse"),
                "RF_r2": rf.get("r2"),
            })

        rows.append(rec)
    return pd.DataFrame(rows)

def choose_winner(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    if g["task"].iloc[0] == "classification":
        # use RF primarily (it’s usually stronger), but report LR as well
        g["primary_score"] = g["RF_balanced_accuracy"]
        g["secondary"] = g["RF_macro_f1"]
        g["tertiary"] = g["RF_accuracy"]
        g = g.sort_values(by=["primary_score","secondary","tertiary"], ascending=[False, False, False])
    else:
        # regression: lower rmse better
        g["primary_score"] = -g["RF_rmse"]  # negate to sort descending
        g["secondary"] = -g["RF_mae"]
        g["tertiary"] = g["RF_r2"]
        g = g.sort_values(by=["primary_score","secondary","tertiary"], ascending=[False, False, False])
    return g

def main():
    df = load_metrics()
    if df.empty:
        raise SystemExit("No metrics found. Did you run train_baselines.py?")

    # Write the raw metrics table
    raw_csv = OUT_DIR / "all_metrics_flat.csv"
    df.to_csv(raw_csv, index=False)

    # Leaderboard per target
    boards = []
    for target, g in df.groupby("target", sort=False):
        win = choose_winner(g)
        best = win.iloc[0].to_dict()
        boards.append({
            "target": target,
            "best_pack": best["pack"],
            "best_split": best["split"],
            "task": best["task"],
            # classification winners
            "RF_balanced_accuracy": best.get("RF_balanced_accuracy"),
            "RF_macro_f1": best.get("RF_macro_f1"),
            "RF_accuracy": best.get("RF_accuracy"),
            # regression winners
            "RF_rmse": best.get("RF_rmse"),
            "RF_mae": best.get("RF_mae"),
            "RF_r2": best.get("RF_r2"),
            "n_features": best.get("n_features"),
            "metrics_path": best.get("metrics_path"),
        })

    lb = pd.DataFrame(boards)
    lb_csv = OUT_DIR / "leaderboard.csv"
    lb.to_csv(lb_csv, index=False)

    # Also provide a compact Markdown table for README or notes
    lines = ["# Baseline Leaderboard (RandomForest per target)", ""]
    for _, r in lb.iterrows():
        if r["task"] == "classification":
            lines += [
                f"## {r['target']}",
                f"- **Best pack:** `{r['best_pack']}` ({r['best_split']})",
                f"- **Balanced Acc:** {r['RF_balanced_accuracy']:.4f}  |  Macro F1: {r['RF_macro_f1']:.4f}  |  Acc: {r['RF_accuracy']:.4f}",
                f"- **n_features:** {int(r['n_features'])}",
                f"- **metrics:** `{r['metrics_path']}`",
                ""
            ]
        else:
            lines += [
                f"## {r['target']}",
                f"- **Best pack:** `{r['best_pack']}` ({r['best_split']})",
                f"- **RMSE:** {r['RF_rmse']:.4f}  |  MAE: {r['RF_mae']:.4f}  |  R²: {r['RF_r2']:.4f}",
                f"- **n_features:** {int(r['n_features'])}",
                f"- **metrics:** `{r['metrics_path']}`",
                ""
            ]
    md_path = OUT_DIR / "leaderboard.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ Wrote:\n  - {raw_csv}\n  - {lb_csv}\n  - {md_path}")

    try:
        # Re-load the leaderboard with metrics_path (already contained)
        generate_winner_plots(lb)
        print("✅ Plots, per-class metrics, permutation importances, and model cards generated.")
    except Exception as e:
        print(f"⚠️  Plot generation skipped due to error: {e}")

if __name__ == "__main__":
    main()