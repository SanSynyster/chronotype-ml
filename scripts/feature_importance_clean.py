#!/usr/bin/env python3
"""Cross-validated permutation importance for clean classifier datasets.

Importance is computed on held-out folds by shuffling one original input column
at a time and measuring the drop in balanced accuracy. This keeps importance
interpretable at the feature-engineering level instead of transformed one-hot or
scaled columns.
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
    X_all = df.drop(columns=[target])
    feature_cols = [c for c in X_all.columns if c not in ID_LIKE]
    X = X_all[feature_cols].copy()
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


def make_cv(y: np.ndarray, groups: np.ndarray | None, splits: int):
    if groups is not None:
        n_groups = len(np.unique(groups))
        return GroupKFold(n_splits=max(2, min(splits, n_groups)))
    min_class = int(pd.Series(y).value_counts().min())
    return StratifiedKFold(n_splits=max(2, min(splits, min_class)), shuffle=True, random_state=42)


def permutation_importance_cv(
    df: pd.DataFrame,
    target: str,
    group_col: str | None,
    model_name: str,
    splits: int,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    X, y, groups, numeric_cols, categorical_cols = split_features(df, target, group_col)
    cv = make_cv(y, groups, splits)
    base_pipe = Pipeline([("pre", make_preprocessor(numeric_cols, categorical_cols)), ("clf", make_model(model_name))])
    rng = np.random.default_rng(seed)

    rows = []
    fold_rows = []
    split_iter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        pipe = clone(base_pipe)
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        pipe.fit(X_train, y_train)
        baseline_pred = pipe.predict(X_test)
        baseline_score = float(balanced_accuracy_score(y_test, baseline_pred))
        fold_rows.append({"fold": fold_idx, "baseline_balanced_accuracy": baseline_score, "n_test": int(len(test_idx))})

        for feature in X.columns:
            drops = []
            for repeat_idx in range(1, repeats + 1):
                X_perm = X_test.copy()
                X_perm[feature] = rng.permutation(X_perm[feature].to_numpy())
                perm_pred = pipe.predict(X_perm)
                perm_score = float(balanced_accuracy_score(y_test, perm_pred))
                drop = baseline_score - perm_score
                drops.append(drop)
                rows.append({
                    "feature": feature,
                    "fold": fold_idx,
                    "repeat": repeat_idx,
                    "baseline_balanced_accuracy": baseline_score,
                    "permuted_balanced_accuracy": perm_score,
                    "importance_drop": drop,
                })

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("feature", as_index=False)
        .agg(
            mean_importance=("importance_drop", "mean"),
            std_importance=("importance_drop", "std"),
            min_importance=("importance_drop", "min"),
            max_importance=("importance_drop", "max"),
            positive_rate=("importance_drop", lambda s: float((s > 0).mean())),
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    meta = {
        "target": target,
        "model": model_name,
        "rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "group_col": group_col,
        "cv": type(cv).__name__,
        "splits": splits,
        "repeats": repeats,
        "seed": seed,
        "baseline_fold_scores": fold_rows,
        "baseline_mean_balanced_accuracy": float(pd.DataFrame(fold_rows)["baseline_balanced_accuracy"].mean()),
    }
    return summary, {"detail": detail, "meta": meta}


def write_markdown(summary: pd.DataFrame, meta: dict, out_path: Path, top_n: int) -> None:
    cols = ["feature", "mean_importance", "std_importance", "positive_rate"]
    top = summary.head(top_n)[cols]
    lines = [
        "# Feature Importance",
        "",
        f"Model: `{meta['model']}`",
        f"Target: `{meta['target']}`",
        f"Baseline mean balanced accuracy: `{meta['baseline_mean_balanced_accuracy']:.4f}`",
        "",
        "| feature | mean_importance | std_importance | positive_rate |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['feature']} | {row['mean_importance']:.4f} | {row['std_importance']:.4f} | {row['positive_rate']:.4f} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validated permutation importance for clean datasets.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--group-col", default="participant_id")
    parser.add_argument("--model", default="logreg", choices=["logreg", "rf", "hgb"])
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="reports/clean/feature_importance")
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Missing target: {args.target}")
    df = df[df[args.target].notna()].reset_index(drop=True)
    group_col = args.group_col if args.group_col and args.group_col in df.columns else None

    summary, result = permutation_importance_cv(
        df=df,
        target=args.target,
        group_col=group_col,
        model_name=args.model,
        splits=args.splits,
        repeats=args.repeats,
        seed=args.seed,
    )

    outdir = Path(args.outdir) / Path(args.data).stem / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "importance_summary.csv", index=False)
    result["detail"].to_csv(outdir / "importance_detail.csv", index=False)
    (outdir / "meta.json").write_text(json.dumps(result["meta"], indent=2), encoding="utf-8")
    write_markdown(summary, result["meta"], outdir / "importance_summary.md", args.top_n)

    print(f"Wrote {outdir}")
    print(summary.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
