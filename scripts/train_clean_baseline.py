#!/usr/bin/env python3
"""Train leakage-aware baseline classifiers with cross-validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_cv(y: np.ndarray, groups: np.ndarray | None, requested_splits: int):
    if groups is not None:
        n_groups = len(np.unique(groups))
        return GroupKFold(n_splits=max(2, min(requested_splits, n_groups)))
    class_counts = pd.Series(y).value_counts()
    min_class = int(class_counts.min())
    return StratifiedKFold(n_splits=max(2, min(requested_splits, min_class)), shuffle=True, random_state=42)


def split_features(df: pd.DataFrame, target: str, group_col: str | None) -> tuple[pd.DataFrame, pd.Series, np.ndarray | None, list[str], list[str]]:
    drop_cols = [target]
    if group_col and group_col in df.columns:
        groups = df[group_col].to_numpy()
    else:
        groups = None
    X = df.drop(columns=drop_cols)
    id_like = {"participant_id", "Block", "Trial", "global_trial_index"}
    feature_cols = [c for c in X.columns if c not in id_like]
    X = X[feature_cols].copy()
    y = df[target].copy()
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return X, y, groups, numeric_cols, categorical_cols


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
            categorical_cols,
        ))
    return ColumnTransformer(transformers)


def model_specs() -> dict[str, object]:
    return {
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, random_state=42),
    }


def binary_scores(y_true, y_pred, y_prob) -> dict[str, float]:
    scores = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            scores["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            pass
    return scores


def evaluate(df: pd.DataFrame, target: str, group_col: str | None, splits: int) -> tuple[pd.DataFrame, dict]:
    X, y_raw, groups, numeric_cols, categorical_cols = split_features(df, target, group_col)
    y = y_raw.astype(str).to_numpy()
    cv = make_cv(y, groups, splits)
    preprocessor = make_preprocessor(numeric_cols, categorical_cols)

    rows = []
    predictions = []
    for model_name, clf in model_specs().items():
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        fold_iter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        for fold_idx, (train_idx, test_idx) in enumerate(fold_iter, start=1):
            pipe.fit(X.iloc[train_idx], y[train_idx])
            pred = pipe.predict(X.iloc[test_idx])
            prob = None
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                proba = pipe.predict_proba(X.iloc[test_idx])
                if proba.shape[1] == 2:
                    prob = proba[:, 1]
            scores = binary_scores(y[test_idx], pred, prob)
            scores.update({"model": model_name, "fold": fold_idx, "n_test": int(len(test_idx))})
            rows.append(scores)
            pred_frame = pd.DataFrame({"model": model_name, "fold": fold_idx, "y_true": y[test_idx], "y_pred": pred})
            if prob is not None:
                pred_frame["y_prob_positive"] = prob
            predictions.append(pred_frame)

    metrics = pd.DataFrame(rows)
    summary = {
        "target": target,
        "rows": int(df.shape[0]),
        "features": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "n_total": len(numeric_cols) + len(categorical_cols),
        },
        "group_col": group_col,
        "cv": type(cv).__name__,
        "by_model_mean": metrics.groupby("model").mean(numeric_only=True).to_dict(orient="index"),
    }
    return pd.concat(predictions, ignore_index=True), {"fold_metrics": metrics, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train clean baseline classifiers.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--group-col", default="participant_id")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--outdir", default="reports/clean")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Missing target: {args.target}")
    df = df[df[args.target].notna()].reset_index(drop=True)

    group_col = args.group_col if args.group_col in df.columns else None
    preds, results = evaluate(df, args.target, group_col, args.splits)

    outdir = Path(args.outdir) / Path(args.data).stem
    outdir.mkdir(parents=True, exist_ok=True)
    results["fold_metrics"].to_csv(outdir / "fold_metrics.csv", index=False)
    preds.to_csv(outdir / "predictions.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(results["summary"], indent=2), encoding="utf-8")

    print(f"Wrote reports to {outdir}")
    print(json.dumps(results["summary"]["by_model_mean"], indent=2))


if __name__ == "__main__":
    main()
