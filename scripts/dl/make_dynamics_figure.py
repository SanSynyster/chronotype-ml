#!/usr/bin/env python3
"""Figure for idea #5: chronotype decoded from risky-choice dynamics.

Panel A: permutation null distribution of nested-LOO ROC AUC vs. the observed value.
Panel B: out-of-fold decision score vs. continuous MEQ (independent ground truth).

Reads reports/clean/chronotype_from_dynamics/{null_auc.npy,oof_scores.csv,metrics.json}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SRC = Path("reports/clean/chronotype_from_dynamics")
OUT = Path("docs/figures")


def main() -> None:
    metrics = json.loads((SRC / "metrics.json").read_text())
    null = np.load(SRC / "null_auc.npy")
    scores = pd.read_csv(SRC / "oof_scores.csv")
    obs = metrics["primary"]["nested_loo_roc_auc"]
    p = metrics["primary"]["perm_p_value"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4))

    axA.hist(null, bins=30, color="0.7", edgecolor="white")
    axA.axvline(obs, color="crimson", lw=2, label=f"observed AUC = {obs:.3f}")
    axA.axvline(np.percentile(null, 95), color="k", ls="--", lw=1, label="null 95th pct")
    axA.set_xlabel("nested-LOO ROC AUC (label permutations)")
    axA.set_ylabel("count")
    axA.set_title(f"A. Permutation test (p = {p:.3f})")
    axA.legend(fontsize=8, frameon=False)

    m = scores["meq"].notna()
    ev = scores["chronotype_evening"] == 1
    axB.scatter(scores.loc[m & ev, "meq"], scores.loc[m & ev, "oof_decision_score"],
                color="tab:blue", label="Evening", alpha=0.8)
    axB.scatter(scores.loc[m & ~ev, "meq"], scores.loc[m & ~ev, "oof_decision_score"],
                color="tab:orange", label="Morning", alpha=0.8)
    r, pr = pearsonr(scores.loc[m, "meq"], scores.loc[m, "oof_decision_score"])
    axB.set_xlabel("MEQ score (lower = more evening)")
    axB.set_ylabel("OOF evening decision score")
    axB.set_title(f"B. Corroboration vs. MEQ (r = {r:.2f}, p = {pr:.3f})")
    axB.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig9_chronotype_from_dynamics.{ext}", dpi=150)
    print("wrote", OUT / "fig9_chronotype_from_dynamics.png")


if __name__ == "__main__":
    main()
