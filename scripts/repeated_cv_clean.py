#!/usr/bin/env python3
"""Repeated cross-validation for clean classifier datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_LIKE = {"participant_id", "Block", "Trial", "global_trial_index"}


def split_features(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    X = X[[c for c in X.columns if c not in ID_LIKE]].copy()
    y = df[target].astype(str).to_numpy()
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return X, y, num, cat


def make_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    transformers = []
    if num:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num))
    if cat:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(transformers)


def make_model(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    raise ValueError(f"Unsupported model: {name}")


def score_fold(y_true, y_pred, y_prob=None) -> dict:
    row = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            row["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            pass
    return row


def ci95(values: pd.Series) -> tuple[float, float]:
    vals = values.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated CV for a clean classifier dataset.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", default="logreg", choices=["logreg", "rf"])
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="reports/clean/repeated_cv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df[df[args.target].notna()].reset_index(drop=True)
    X, y, num, cat = split_features(df, args.target)
    min_class = int(pd.Series(y).value_counts().min())
    splits = max(2, min(args.splits, min_class))
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=args.repeats, random_state=args.seed)
    pipe = Pipeline([("pre", make_preprocessor(num, cat)), ("clf", make_model(args.model))])

    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        cur = clone(pipe)
        cur.fit(X.iloc[tr], y[tr])
        pred = cur.predict(X.iloc[te])
        prob = None
        if hasattr(cur.named_steps["clf"], "predict_proba"):
            proba = cur.predict_proba(X.iloc[te])
            if proba.shape[1] == 2:
                prob = proba[:, 1]
        row = score_fold(y[te], pred, prob)
        row.update({"fold": fold, "n_test": int(len(te))})
        rows.append(row)

    metrics = pd.DataFrame(rows)
    summary = {
        "data": args.data,
        "target": args.target,
        "model": args.model,
        "rows": int(df.shape[0]),
        "n_features": int(len(num) + len(cat)),
        "splits": splits,
        "repeats": args.repeats,
        "n_folds": int(metrics.shape[0]),
        "metrics": {},
    }
    for col in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]:
        if col in metrics:
            low, high = ci95(metrics[col])
            summary["metrics"][col] = {
                "mean": float(metrics[col].mean()),
                "std": float(metrics[col].std(ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
            }

    outdir = Path(args.outdir) / Path(args.data).stem / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(outdir / "fold_metrics.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
