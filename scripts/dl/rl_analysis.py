#!/usr/bin/env python3
"""Do the RL parameters differ by chronotype, and do they track MEQ?

Loads the per-participant RL parameters (rl_model.py), then:
  (1) group comparison Morning vs Evening per parameter: Mann-Whitney U, Cohen's d
      with a bootstrap 95% CI;
  (2) correlation of each parameter with the continuous MEQ score (Pearson + Spearman);
  (3) a permutation-clean predictive check: do the RL parameters jointly classify
      chronotype (shared nested-LOO evaluator)?

This turns "a model predicts chronotype" into an interpretable, mechanistic statement
about feedback-driven learning.

Run (env_dl):
    python scripts/dl/rl_analysis.py --permutations 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

import chrono_eval

PARAMS = ["alpha_gain", "alpha_loss", "lr_asymmetry", "beta", "bias"]
RL = "reports/clean/rl_model/participant_rl_params.csv"
CHRONO = "data/clean/chronotype_participant.csv"
MEQ = "data/processed/participant_meq_scores.csv"
OUTDIR = Path("reports/clean/rl_model")
SEED = 0


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def boot_d_ci(a, b, n=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    ds = [cohens_d(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n)]
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args()

    rl = pd.read_csv(RL)
    lab = pd.read_csv(CHRONO)[["participant_id", "Chronotype"]]
    meq = pd.read_csv(MEQ)[["UserID", "meq"]]
    df = rl.merge(lab, on="participant_id").merge(
        meq, left_on="participant_id", right_on="UserID", how="left")
    is_evening = df["Chronotype"].str.lower() == "evening"

    out = {"n_evening": int(is_evening.sum()), "n_morning": int((~is_evening).sum()),
           "parameters": {}}
    print(f"{'param':14} {'Morn':>7} {'Even':>7} {'d':>6} {'d_CI':>16} "
          f"{'MW_p':>7} {'MEQ_r':>7} {'MEQ_p':>7}")
    for p in PARAMS:
        ev = df.loc[is_evening, p].to_numpy()
        mo = df.loc[~is_evening, p].to_numpy()
        d = cohens_d(ev, mo)
        lo, hi = boot_d_ci(ev, mo)
        _, mw = mannwhitneyu(ev, mo, alternative="two-sided")
        m = df["meq"].notna()
        r, rp = pearsonr(df.loc[m, p], df.loc[m, "meq"])
        rho, _ = spearmanr(df.loc[m, p], df.loc[m, "meq"])
        out["parameters"][p] = {
            "morning_mean": round(float(mo.mean()), 4), "evening_mean": round(float(ev.mean()), 4),
            "cohens_d_evening_minus_morning": round(float(d), 3),
            "d_ci95": [round(lo, 3), round(hi, 3)], "mannwhitney_p": round(float(mw), 4),
            "meq_pearson_r": round(float(r), 3), "meq_pearson_p": round(float(rp), 4),
            "meq_spearman_rho": round(float(rho), 3),
        }
        print(f"{p:14} {mo.mean():7.3f} {ev.mean():7.3f} {d:6.2f} "
              f"[{lo:5.2f},{hi:5.2f}] {mw:7.3f} {r:7.2f} {rp:7.3f}")

    # Permutation-clean predictive check from the RL parameters.
    emb = df[["participant_id"] + PARAMS].copy()
    pred = chrono_eval.evaluate(emb, permutations=args.permutations, label="rl_params")
    out["predictive_chronotype"] = pred["result"]
    print(f"\nRL-params -> chronotype: AUC={pred['result']['nested_loo_roc_auc']:.3f} "
          f"p={pred['result']['perm_p_value']:.3f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "rl_chronotype_meq.json").write_text(json.dumps(out, indent=2))
    print("wrote", OUTDIR / "rl_chronotype_meq.json")


if __name__ == "__main__":
    main()
