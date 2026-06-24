#!/usr/bin/env python3
"""Consolidated sensitivity matrix for the chronotype manuscript.

Reports, in one table and across the same four participant-exclusion scenarios,
both the secondary exploratory classifier (theory-driven compact_12 logistic
regression) and the primary neural group difference (posterior P300 loss-minus-
gain contrasts). This is the transparency table requested by reviewers: it shows
how the headline claims move when flagged participants are removed.

Scenarios:
- full                    : all 39 participants
- exclude_1013            : drop the EEG/trigger QC case
- exclude_label_conflicts : drop 1027, 1036 (raw-behaviour label disagreement)
- exclude_all_flagged     : drop all three
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from group_stats_chronotype import cohens_d, cohens_d_ci, hedges_g
from permutation_test_clean import run_test
from repeated_cv_clean import (
    ci95,
    make_model,
    make_preprocessor,
    split_features,
)
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline


SCENARIOS = {
    "full": {
        "participant": "data/clean/chronotype_participant.csv",
        "compact": "data/clean/chronotype_compact_12.csv",
        "excluded": [],
    },
    "exclude_1013": {
        "participant": "data/clean/sensitivity/chronotype_participant_exclude_1013.csv",
        "compact": "data/clean/sensitivity/chronotype_compact_12_exclude_1013.csv",
        "excluded": [1013],
    },
    "exclude_label_conflicts": {
        "participant": "data/clean/sensitivity/chronotype_participant_exclude_label_conflicts.csv",
        "compact": "data/clean/sensitivity/chronotype_compact_12_exclude_label_conflicts.csv",
        "excluded": [1027, 1036],
    },
    "exclude_all_flagged": {
        "participant": "data/clean/sensitivity/chronotype_participant_exclude_all_flagged.csv",
        "compact": "data/clean/sensitivity/chronotype_compact_12_exclude_all_flagged.csv",
        "excluded": [1013, 1027, 1036],
    },
}

P300_FEATURES = ["Pz_P300_loss_minus_gain", "POz_P300_loss_minus_gain"]


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def repeated_cv_ba(data_path: str, target: str, repeats: int, seed: int) -> tuple[float, float, float, float]:
    df = pd.read_csv(data_path)
    df = df[df[target].notna()].reset_index(drop=True)
    X, y, num, cat = split_features(df, target)
    min_class = int(pd.Series(y).value_counts().min())
    splits = max(2, min(5, min_class))
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)
    pipe = Pipeline([("pre", make_preprocessor(num, cat)), ("clf", make_model("logreg"))])
    scores = []
    for tr, te in cv.split(X, y):
        cur = clone(pipe)
        cur.fit(X.iloc[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], cur.predict(X.iloc[te])))
    s = pd.Series(scores)
    low, high = ci95(s)
    return float(s.mean()), float(s.std(ddof=1)), low, high


def p300_group_test(participant_path: str, feature: str) -> dict:
    df = pd.read_csv(participant_path)
    evening = pd.to_numeric(df.loc[df["Chronotype"].eq("Evening"), feature], errors="coerce").to_numpy(float)
    morning = pd.to_numeric(df.loc[df["Chronotype"].eq("Morning"), feature], errors="coerce").to_numpy(float)
    e = evening[~np.isnan(evening)]
    m = morning[~np.isnan(morning)]
    welch_p = float(stats.ttest_ind(e, m, equal_var=False).pvalue)
    d_low, d_high = cohens_d_ci(evening, morning)
    return {
        "feature": feature,
        "cohens_d": cohens_d(evening, morning),
        "cohens_d_ci95_low": d_low,
        "cohens_d_ci95_high": d_high,
        "hedges_g": hedges_g(evening, morning),
        "welch_p": welch_p,
        "n_evening": int(len(e)),
        "n_morning": int(len(m)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the consolidated chronotype sensitivity matrix.")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="reports/clean/sensitivity_matrix")
    args = parser.parse_args()

    classifier_rows = []
    p300_rows = []
    for name, cfg in SCENARIOS.items():
        # Secondary exploratory classifier.
        df_c = pd.read_csv(cfg["compact"])
        perm = run_test(
            df=df_c[df_c["Chronotype"].notna()].reset_index(drop=True),
            target="Chronotype",
            group_col=None,
            model_name="logreg",
            n_permutations=args.permutations,
            splits=5,
            seed=args.seed,
        )
        ba_mean, ba_sd, ba_low, ba_high = repeated_cv_ba(cfg["compact"], "Chronotype", args.repeats, args.seed)
        classifier_rows.append({
            "scenario": name,
            "excluded": ",".join(str(p) for p in cfg["excluded"]) or "none",
            "n": int(perm["rows"]),
            "repeated_cv_ba_mean": round(ba_mean, 4),
            "repeated_cv_ba_sd": round(ba_sd, 4),
            "repeated_cv_ba_ci95_low": round(ba_low, 4),
            "repeated_cv_ba_ci95_high": round(ba_high, 4),
            "perm_observed_ba": round(perm["observed_balanced_accuracy"], 4),
            "perm_p_value": round(perm["p_value"], 4),
        })

        # Primary neural group difference.
        for feature in P300_FEATURES:
            row = p300_group_test(cfg["participant"], feature)
            p300_rows.append({
                "scenario": name,
                "excluded": ",".join(str(p) for p in cfg["excluded"]) or "none",
                "feature": row["feature"],
                "n_evening": row["n_evening"],
                "n_morning": row["n_morning"],
                "cohens_d": round(row["cohens_d"], 3),
                "cohens_d_ci95_low": round(row["cohens_d_ci95_low"], 3),
                "cohens_d_ci95_high": round(row["cohens_d_ci95_high"], 3),
                "hedges_g": round(row["hedges_g"], 3),
                "welch_p": round(row["welch_p"], 4),
            })

    classifier = pd.DataFrame(classifier_rows)
    p300 = pd.DataFrame(p300_rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    classifier.to_csv(outdir / "classifier_sensitivity.csv", index=False)
    p300.to_csv(outdir / "p300_sensitivity.csv", index=False)

    lines = ["# Chronotype Sensitivity Matrix", ""]
    lines += ["## Secondary exploratory classifier (compact_12, logistic regression)", ""]
    lines += [df_to_markdown(classifier)]
    lines += ["", "## Primary neural group difference (posterior P300 loss-minus-gain)", ""]
    lines += [df_to_markdown(p300)]
    lines += [""]
    (outdir / "sensitivity_matrix.md").write_text("\n".join(lines), encoding="utf-8")

    print(classifier.to_string(index=False))
    print()
    print(p300.to_string(index=False))
    print(f"\nWrote {outdir / 'sensitivity_matrix.md'}")


if __name__ == "__main__":
    main()
