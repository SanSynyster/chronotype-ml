#!/usr/bin/env python3
"""Continuous MEQ analysis of the posterior P300 feedback effect.

The primary group comparison dichotomizes chronotype into Morning/Evening, but
12 of 39 participants fall in the MEQ intermediate band where that split is soft.
This analysis instead relates the posterior P300 loss-minus-gain contrast to the
continuous MEQ score, which avoids the dichotomization entirely.

Under the standard Horne-Ostberg direction (higher MEQ = more morning) and the
group result (morning types show a more positive loss-minus-gain contrast), we
expect a positive MEQ-P300 correlation.

Outputs Pearson r with a percentile-bootstrap 95% CI, Spearman rho, and the OLS
slope per electrode to reports/clean/meq_p300/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


FEATURES = ["Pz_P300_loss_minus_gain", "POz_P300_loss_minus_gain"]


def pearson_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        bx, by = x[idx], y[idx]
        if np.std(bx) == 0 or np.std(by) == 0:
            boots[i] = np.nan
            continue
        boots[i] = np.corrcoef(bx, by)[0, 1]
    boots = boots[np.isfinite(boots)]
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous MEQ vs posterior P300.")
    parser.add_argument("--participant", default="data/clean/chronotype_participant.csv")
    parser.add_argument("--meq", default="data/processed/participant_meq_scores.csv")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="reports/clean/meq_p300")
    args = parser.parse_args()

    part = pd.read_csv(args.participant)
    if "meq" in part.columns:
        df = part.copy()
    else:
        meq = pd.read_csv(args.meq)[["UserID", "meq"]]
        df = part.merge(meq, left_on="participant_id", right_on="UserID", how="inner")

    rows = []
    for feat in FEATURES:
        sub = df[[feat, "meq"]].apply(pd.to_numeric, errors="coerce").dropna()
        x = sub["meq"].to_numpy(dtype=float)
        y = sub[feat].to_numpy(dtype=float)
        r, p_r = stats.pearsonr(x, y)
        rho, p_rho = stats.spearmanr(x, y)
        lo, hi = pearson_ci(x, y, args.n_boot, args.seed)
        slope, intercept, _, _, se = stats.linregress(x, y)
        rows.append({
            "feature": feat,
            "n": int(len(sub)),
            "pearson_r": round(float(r), 4),
            "pearson_ci95_low": round(lo, 4),
            "pearson_ci95_high": round(hi, 4),
            "pearson_p": round(float(p_r), 4),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(p_rho), 4),
            "ols_slope_per_meq_point": round(float(slope), 5),
            "ols_slope_se": round(float(se), 5),
        })

    out = pd.DataFrame(rows)
    # FDR across the two electrodes.
    p = out["pearson_p"].to_numpy(dtype=float)
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=int)
    ranks[order] = np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate((p * len(p) / ranks)[order][::-1])[::-1]
    fdr = np.empty(len(p))
    fdr[order] = np.clip(adj, 0, 1)
    out["pearson_p_fdr"] = np.round(fdr, 4)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "meq_p300_correlations.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(out.to_string(index=False))
    print(f"\nWrote {outdir / 'meq_p300_correlations.csv'}")


if __name__ == "__main__":
    main()
