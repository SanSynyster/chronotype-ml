#!/usr/bin/env python3
"""Classical Morning-vs-Evening group statistics for chronotype features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


FEATURES = [
    "free_risky_rate",
    "gain_correct_risky_rate",
    "loss_error_risky_rate",
    "post_error_slowing",
    "rt_slope",
    "risky_late_minus_early",
    "Fz_FRN_error_minus_correct",
    "FCz_FRN_error_minus_correct",
    "Fz_FRN_loss_error_minus_gain_correct",
    "POz_P300_loss_minus_gain",
    "Pz_P300_loss_minus_gain",
    "CPz_P300_error_minus_correct",
]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def fdr_bh(pvals: list[float]) -> list[float]:
    p = np.array(pvals, dtype=float)
    order = np.argsort(p)
    ranked = np.empty_like(p)
    n = len(p)
    prev = 1.0
    for idx in order[::-1]:
        rank = np.where(order == idx)[0][0] + 1
        val = min(prev, p[idx] * n / rank)
        ranked[idx] = val
        prev = val
    return ranked.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chronotype group statistics.")
    parser.add_argument("--data", default="data/clean/chronotype_participant.csv")
    parser.add_argument("--outdir", default="reports/clean/group_stats")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    rows = []
    for feature in [f for f in FEATURES if f in df.columns]:
        evening = pd.to_numeric(df.loc[df["Chronotype"].eq("Evening"), feature], errors="coerce").to_numpy(dtype=float)
        morning = pd.to_numeric(df.loc[df["Chronotype"].eq("Morning"), feature], errors="coerce").to_numpy(dtype=float)
        e_clean = evening[~np.isnan(evening)]
        m_clean = morning[~np.isnan(morning)]
        if len(e_clean) >= 2 and len(m_clean) >= 2:
            t_res = stats.ttest_ind(e_clean, m_clean, equal_var=False, nan_policy="omit")
            u_res = stats.mannwhitneyu(e_clean, m_clean, alternative="two-sided")
            p_t = float(t_res.pvalue)
            p_u = float(u_res.pvalue)
        else:
            p_t = np.nan
            p_u = np.nan
        rows.append({
            "feature": feature,
            "evening_mean": float(np.nanmean(evening)),
            "evening_sd": float(np.nanstd(evening, ddof=1)),
            "morning_mean": float(np.nanmean(morning)),
            "morning_sd": float(np.nanstd(morning, ddof=1)),
            "cohens_d_evening_minus_morning": cohens_d(evening, morning),
            "welch_p": p_t,
            "mannwhitney_p": p_u,
            "n_evening": int(len(e_clean)),
            "n_morning": int(len(m_clean)),
        })

    out = pd.DataFrame(rows)
    out["welch_p_fdr"] = fdr_bh(out["welch_p"].fillna(1).tolist())
    out = out.sort_values("welch_p")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "chronotype_group_stats.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
