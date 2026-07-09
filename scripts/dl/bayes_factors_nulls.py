#!/usr/bin/env python3
"""Bayes factors for planned null results.

Assumptions and leakage notes
----------------------------
- Seed is fixed at 0. This script does not fit predictive models or resplit data;
  it consumes participant-level summaries or already participant-grouped EEGNet
  permutation outputs generated elsewhere.
- FRN group differences are recomputed from one row per participant in
  data/clean/chronotype_frn_core.csv. Morning and Evening groups are independent;
  no participant contributes to both groups.
- FRN Bayes factors use pingouin.bayesfactor_ttest on the independent-samples
  Welch t statistic. Pingouin returns BF10 for a group difference, so this script
  reports BF01 = 1 / BF10 as evidence for the null.
- EEGNet chronotype Bayes factors use only the observed nested leave-one-out AUCs
  and their label-permutation null AUC distributions in reports/clean/eeg_chronotype.
  Because there is no closed-form JZS Bayes factor for AUC-vs-permutation-null,
  the reported BF01 is a transparent density-ratio approximation: H0 is a Normal
  approximation to the empirical permutation-null AUC distribution; H1 is a
  two-sided non-null prior over true AUC displacement from the null, Normal with
  prior SD set by --auc-prior-sd. The marginal H1 predictive density is therefore
  Normal(null_mean, sqrt(null_sd^2 + auc_prior_sd^2)). BF01 is density_H0(obs) /
  density_H1(obs). Larger BF01 indicates the observed AUC is more compatible with
  chance-level decoding than with a broad non-null decoding effect. This is a
  sensitivity-calibrated summary for reviewer context, not a replacement for the
  leakage-safe permutation p-value.

Run from repo root with the Bayesian environment:
    env_bayes/bin/python scripts/dl/bayes_factors_nulls.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

SEED = 0
FRN_CONTRASTS = [
    "Fz_FRN_error_minus_correct",
    "FCz_FRN_error_minus_correct",
    "Cz_FRN_error_minus_correct",
]


def _normal_pdf(x: float, mu: float, sd: float) -> float:
    return float(stats.norm.pdf(x, loc=mu, scale=sd))


def _bf01_auc_density_ratio(obs: float, null: np.ndarray, auc_prior_sd: float) -> dict:
    null = np.asarray(null, dtype=float)
    null_mean = float(np.mean(null))
    null_sd = float(np.std(null, ddof=1))
    if null_sd <= 0:
        raise ValueError("Permutation null AUC distribution has zero variance.")
    h0_density = _normal_pdf(obs, null_mean, null_sd)
    h1_sd = math.sqrt(null_sd**2 + auc_prior_sd**2)
    h1_density = _normal_pdf(obs, null_mean, h1_sd)
    bf01 = h0_density / h1_density
    return {
        "observed_auc": round(float(obs), 4),
        "null_auc_mean": round(null_mean, 4),
        "null_auc_sd": round(null_sd, 4),
        "auc_prior_sd_h1": round(float(auc_prior_sd), 4),
        "bf01_density_ratio": round(float(bf01), 4),
        "h0_density_at_observed": round(h0_density, 6),
        "h1_density_at_observed": round(h1_density, 6),
    }


def frn_bayes_factors(frn_csv: Path) -> list[dict]:
    df = pd.read_csv(frn_csv)
    out = []
    for col in FRN_CONTRASTS:
        morning = df.loc[df["Chronotype"].str.lower() == "morning", col].dropna().to_numpy()
        evening = df.loc[df["Chronotype"].str.lower() == "evening", col].dropna().to_numpy()
        t_res = stats.ttest_ind(evening, morning, equal_var=False)
        bf10 = float(pg.bayesfactor_ttest(t_res.statistic, nx=len(evening), ny=len(morning), paired=False))
        bf01 = 1.0 / bf10
        out.append({
            "contrast": col,
            "n_evening": int(len(evening)),
            "n_morning": int(len(morning)),
            "evening_mean": round(float(np.mean(evening)), 4),
            "morning_mean": round(float(np.mean(morning)), 4),
            "welch_t": round(float(t_res.statistic), 4),
            "welch_p": round(float(t_res.pvalue), 4),
            "bf10_group_difference": round(bf10, 4),
            "bf01_null": round(float(bf01), 4),
        })
    return out


def eegnet_bayes_factors(eeg_dir: Path, feedback_metrics: Path, auc_prior_sd: float) -> dict:
    metrics = json.loads((eeg_dir / "metrics.json").read_text())
    feedback = json.loads(feedback_metrics.read_text())
    embeddings = {}
    for name in ["mean", "contrast"]:
        obs = metrics["embeddings"][name]["nested_loo_roc_auc"]
        null = np.load(eeg_dir / f"null_auc_{name}.npy")
        entry = _bf01_auc_density_ratio(obs, null, auc_prior_sd)
        entry["permutation_p_value"] = metrics["embeddings"][name]["perm_p_value"]
        embeddings[name] = entry
    return {
        "chronotype_embeddings": embeddings,
        "positive_control_feedback_valence": {
            "source": str(feedback_metrics),
            "roc_auc": feedback["out_of_fold"]["roc_auc"],
            "balanced_accuracy": feedback["out_of_fold"]["balanced_accuracy"],
            "note": "Positive-control EEGNet valence decoding is reported for context, not converted to BF01 for the chronotype null.",
        },
    }


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Bayes Factors for Planned Nulls",
        "",
        "BF01 values greater than 1 favour the null over the tested alternative.",
        "",
        "## FRN Error-Minus-Correct Group Difference",
        "",
        "| Contrast | Evening mean | Morning mean | Welch t | p | BF01 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["frn_group_difference"]:
        lines.append(
            f"| {row['contrast']} | {row['evening_mean']:.4f} | {row['morning_mean']:.4f} | "
            f"{row['welch_t']:.4f} | {row['welch_p']:.4f} | {row['bf01_null']:.4f} |"
        )
    lines += [
        "",
        "## EEGNet Chronotype Decoding",
        "",
        "Permutation-null density-ratio BF01; see script header for assumptions.",
        "",
        "| Embedding | Observed AUC | Null mean | Null SD | Permutation p | BF01 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    eegnet = summary["eegnet_chronotype"]
    if "chronotype_embeddings" in eegnet:
        for name, row in eegnet["chronotype_embeddings"].items():
            lines.append(
                f"| {name} | {row['observed_auc']:.4f} | {row['null_auc_mean']:.4f} | "
                f"{row['null_auc_sd']:.4f} | {row['permutation_p_value']:.4f} | "
                f"{row['bf01_density_ratio']:.4f} |"
            )
        pc = eegnet["positive_control_feedback_valence"]
        lines += [
            "",
            "## Positive Control",
            "",
            f"EEGNet feedback-valence AUC = {pc['roc_auc']:.4f}; balanced accuracy = {pc['balanced_accuracy']:.4f}.",
        ]
    else:
        lines.append(f"| unavailable |  |  |  |  |  |")
        lines += ["", f"EEGNet Bayes factors unavailable: {eegnet['reason']}."]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frn-csv", type=Path, default=Path("data/clean/chronotype_frn_core.csv"))
    parser.add_argument("--eeg-dir", type=Path, default=Path("reports/clean/eeg_chronotype"))
    parser.add_argument("--feedback-metrics", type=Path, default=Path("reports/clean/eegnet_feedback/metrics.json"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/clean/bayes_nulls"))
    parser.add_argument("--auc-prior-sd", type=float, default=0.10)
    args = parser.parse_args()

    np.random.seed(SEED)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": SEED,
        "frn_group_difference": frn_bayes_factors(args.frn_csv),
        "eegnet_chronotype": (
            eegnet_bayes_factors(args.eeg_dir, args.feedback_metrics, args.auc_prior_sd)
            if (args.eeg_dir / "metrics.json").exists() and args.feedback_metrics.exists()
            else {"reason": "single-trial EEGNet outputs are absent; data/raw/shifted_set/*.set is unavailable in this workspace"}
        ),
        "method_notes": {
            "frn": "Independent-group Welch t statistic converted with pingouin.bayesfactor_ttest; BF01 = 1 / BF10.",
            "eegnet": "Permutation-null density-ratio BF01 for chance-level chronotype decoding; H1 prior SD over AUC displacement set by auc_prior_sd.",
        },
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, args.outdir / "summary.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
