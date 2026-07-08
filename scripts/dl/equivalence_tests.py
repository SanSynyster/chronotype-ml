#!/usr/bin/env python3
"""TOST equivalence tests for participant-level FRN contrasts.

Assumptions and leakage notes
----------------------------
- Seed is fixed at 0. This script does not train predictive models or split trial
  data; it analyzes one participant-level row per subject from
  data/clean/chronotype_frn_core.csv.
- Morning and Evening groups are independent. Each participant appears in exactly
  one group, so no participant can be split across conditions or train/test sets.
- Equivalence bounds are specified as +/-0.5 pooled SD for each contrast, per the
  workplan. Bounds are therefore contrast-specific and expressed on the original
  microvolt scale.
- The tested effect is Evening minus Morning. TOST declares equivalence only when
  both one-sided Welch tests reject at alpha: mean difference > lower bound and
  mean difference < upper bound. The reported TOST p-value is max(p_lower,
  p_upper).

Run from repo root:
    env_bayes/bin/python scripts/dl/equivalence_tests.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 0
FRN_CONTRASTS = [
    "Fz_FRN_loss_minus_gain",
    "FCz_FRN_loss_minus_gain",
    "Cz_FRN_loss_minus_gain",
    "Fz_FRN_error_minus_correct",
    "FCz_FRN_error_minus_correct",
    "Cz_FRN_error_minus_correct",
    "Fz_FRN_loss_error_minus_gain_correct",
    "FCz_FRN_loss_error_minus_gain_correct",
    "Cz_FRN_loss_error_minus_gain_correct",
]


def _pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    return math.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))


def _welch_df(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1**2 / (n1**2 * (n1 - 1))) + (v2**2 / (n2**2 * (n2 - 1)))
    return float(num / den)


def tost_for_contrast(evening: np.ndarray, morning: np.ndarray, bound_sd: float, alpha: float) -> dict:
    diff = float(np.mean(evening) - np.mean(morning))
    pooled = _pooled_sd(evening, morning)
    lower_bound = -bound_sd * pooled
    upper_bound = bound_sd * pooled
    se = math.sqrt(np.var(evening, ddof=1) / len(evening) + np.var(morning, ddof=1) / len(morning))
    df = _welch_df(evening, morning)

    t_lower = (diff - lower_bound) / se
    p_lower = float(1.0 - stats.t.cdf(t_lower, df=df))
    t_upper = (diff - upper_bound) / se
    p_upper = float(stats.t.cdf(t_upper, df=df))
    tost_p = max(p_lower, p_upper)
    d = diff / pooled if pooled > 0 else float("nan")

    return {
        "n_evening": int(len(evening)),
        "n_morning": int(len(morning)),
        "evening_mean": round(float(np.mean(evening)), 4),
        "morning_mean": round(float(np.mean(morning)), 4),
        "mean_difference_evening_minus_morning": round(diff, 4),
        "pooled_sd": round(float(pooled), 4),
        "cohens_d_evening_minus_morning": round(float(d), 4),
        "equivalence_bounds_raw": [round(float(lower_bound), 4), round(float(upper_bound), 4)],
        "equivalence_bounds_sd": [-bound_sd, bound_sd],
        "welch_df": round(df, 4),
        "t_lower_diff_gt_lower": round(float(t_lower), 4),
        "p_lower_diff_gt_lower": round(p_lower, 4),
        "t_upper_diff_lt_upper": round(float(t_upper), 4),
        "p_upper_diff_lt_upper": round(p_upper, 4),
        "tost_p": round(float(tost_p), 4),
        "equivalent_at_alpha": bool(tost_p < alpha),
    }


def run_tost(frn_csv: Path, bound_sd: float, alpha: float) -> list[dict]:
    df = pd.read_csv(frn_csv)
    rows = []
    for col in FRN_CONTRASTS:
        evening = df.loc[df["Chronotype"].str.lower() == "evening", col].dropna().to_numpy(dtype=float)
        morning = df.loc[df["Chronotype"].str.lower() == "morning", col].dropna().to_numpy(dtype=float)
        result = tost_for_contrast(evening, morning, bound_sd, alpha)
        result["contrast"] = col
        rows.append(result)
    return rows


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# TOST Equivalence Tests for FRN Contrasts",
        "",
        f"Equivalence bounds: +/-{summary['bound_sd']} pooled SD per contrast.",
        f"Alpha: {summary['alpha']}.",
        "Effect direction: Evening minus Morning.",
        "",
        "| Contrast | Difference | d | Raw lower | Raw upper | TOST p | Equivalent? |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["contrasts"]:
        decision = "yes" if row["equivalent_at_alpha"] else "no"
        lines.append(
            f"| {row['contrast']} | {row['mean_difference_evening_minus_morning']:.4f} | "
            f"{row['cohens_d_evening_minus_morning']:.4f} | {row['equivalence_bounds_raw'][0]:.4f} | "
            f"{row['equivalence_bounds_raw'][1]:.4f} | {row['tost_p']:.4f} | {decision} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frn-csv", type=Path, default=Path("data/clean/chronotype_frn_core.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/clean/tost"))
    parser.add_argument("--bound-sd", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    np.random.seed(SEED)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": SEED,
        "source": str(args.frn_csv),
        "alpha": args.alpha,
        "bound_sd": args.bound_sd,
        "method": "Independent-groups Welch TOST; effect is Evening minus Morning; bounds are +/- bound_sd times pooled SD for each contrast.",
        "contrasts": run_tost(args.frn_csv, args.bound_sd, args.alpha),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, args.outdir / "summary.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
