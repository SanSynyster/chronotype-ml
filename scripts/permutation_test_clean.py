#!/usr/bin/env python3
"""Permutation-test clean classification datasets.

This estimates whether observed cross-validated balanced accuracy is above the
null distribution obtained by shuffling labels. It is especially important for
chronotype, where there are only 39 participants.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_LIKE = {"participant_id", "Block", "Trial", "global_trial_index"}


def split_features(df: pd.DataFrame, target: str, group_col: str | None):
    groups = df[group_col].to_numpy() if group_col and group_col in df.columns else None
    X = df.drop(columns=[target])
    feature_cols = [c for c in X.columns if c not in ID_LIKE]
    X = X[feature_cols].copy()
    y = df[target].astype(str).to_numpy()
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


def make_cv(y: np.ndarray, groups: np.ndarray | None, splits: int):
    if groups is not None:
        n_groups = len(np.unique(groups))
        return GroupKFold(n_splits=max(2, min(splits, n_groups)))
    min_class = int(pd.Series(y).value_counts().min())
    return StratifiedKFold(n_splits=max(2, min(splits, min_class)), shuffle=True, random_state=42)


def make_model(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, random_state=42)
    raise ValueError(f"Unknown model: {name}")


def cv_balanced_accuracy(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None, cv) -> float:
    scores = []
    split_iter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for train_idx, test_idx in split_iter:
        cur = clone(pipe)
        cur.fit(X.iloc[train_idx], y[train_idx])
        pred = cur.predict(X.iloc[test_idx])
        scores.append(balanced_accuracy_score(y[test_idx], pred))
    return float(np.mean(scores))


def run_test(
    df: pd.DataFrame,
    target: str,
    group_col: str | None,
    model_name: str,
    n_permutations: int,
    splits: int,
    seed: int,
) -> dict:
    X, y, groups, numeric_cols, categorical_cols = split_features(df, target, group_col)
    cv = make_cv(y, groups, splits)
    pipe = Pipeline([("pre", make_preprocessor(numeric_cols, categorical_cols)), ("clf", make_model(model_name))])
    observed = cv_balanced_accuracy(pipe, X, y, groups, cv)

    rng = np.random.default_rng(seed)
    null_scores = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(y)
        null_scores.append(cv_balanced_accuracy(pipe, X, shuffled, groups, cv))

    null = np.array(null_scores, dtype=float)
    p_value = float((1 + np.sum(null >= observed)) / (n_permutations + 1))
    return {
        "target": target,
        "model": model_name,
        "rows": int(df.shape[0]),
        "n_features": int(len(numeric_cols) + len(categorical_cols)),
        "group_col": group_col,
        "cv": type(cv).__name__,
        "splits": splits,
        "n_permutations": n_permutations,
        "seed": seed,
        "observed_balanced_accuracy": observed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": p_value,
        "null_scores": null_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation-test a clean classifier dataset.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--group-col", default="participant_id")
    parser.add_argument("--model", default="logreg", choices=["logreg", "rf", "hgb"])
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="reports/clean/permutation_tests")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Missing target: {args.target}")
    df = df[df[args.target].notna()].reset_index(drop=True)
    group_col = args.group_col if args.group_col and args.group_col in df.columns else None

    result = run_test(
        df=df,
        target=args.target,
        group_col=group_col,
        model_name=args.model,
        n_permutations=args.permutations,
        splits=args.splits,
        seed=args.seed,
    )

    outdir = Path(args.outdir) / Path(args.data).stem / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({"null_balanced_accuracy": result["null_scores"]}).to_csv(outdir / "null_scores.csv", index=False)

    printable = {k: v for k, v in result.items() if k != "null_scores"}
    print(json.dumps(printable, indent=2))
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
