#!/usr/bin/env python3
"""Does the posterior-P300 chronotype effect survive adjustment for sex and age?

The cohort has a sex/age imbalance across chronotype groups (Evening ~68% male and
~2 yr older than Morning), so a reviewer will ask whether the posterior-P300
loss-minus-gain group difference is confounded by sex or age. This fits OLS models
of the P300 contrast with chronotype plus sex and age covariates, and reports the
adjusted chronotype effect against the unadjusted one, for Pz and POz.

Run:  env/bin/python scripts/dl/p300_covariate_check.py
Writes reports/clean/p300_covariate/summary.{json,md}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = "data/clean/chronotype_participant.csv"
OUTDIR = Path("reports/clean/p300_covariate")
ELECTRODES = ["Pz_P300_loss_minus_gain", "POz_P300_loss_minus_gain"]


def partial_d(t, df):
    """Approximate Cohen's d for a regression coefficient from its t and residual df."""
    return float(2 * t / np.sqrt(df)) if df > 0 else np.nan


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(DATA)
    # code chronotype so a positive coefficient means Morning > Evening (matches the
    # manuscript sign: Evening more negative loss-minus-gain)
    d["morning"] = (d["Chronotype"].str.lower() == "morning").astype(int)
    d["male"] = (d["Gender"].astype(str).str.upper().str[0] == "M").astype(int)
    d["age_c"] = d["Age"] - d["Age"].mean()

    results = {"n": int(len(d)),
               "note": "morning=1 vs evening=0; positive beta_morning = Morning higher "
                       "loss-minus-gain (Evening more negative), matching the primary effect."}
    rows = []
    for col in ELECTRODES:
        dd = d.dropna(subset=[col]).copy()
        # unadjusted
        m0 = smf.ols(f"Q('{col}') ~ morning", data=dd).fit()
        # adjusted for sex + age
        m1 = smf.ols(f"Q('{col}') ~ morning + male + age_c", data=dd).fit()
        b0, p0 = m0.params["morning"], m0.pvalues["morning"]
        b1, p1, t1 = m1.params["morning"], m1.pvalues["morning"], m1.tvalues["morning"]
        entry = {
            "electrode": col, "n": int(len(dd)),
            "unadjusted_beta_morning": float(b0), "unadjusted_p": float(p0),
            "adjusted_beta_morning": float(b1), "adjusted_p": float(p1),
            "adjusted_partial_d": partial_d(t1, m1.df_resid),
            "sex_beta": float(m1.params["male"]), "sex_p": float(m1.pvalues["male"]),
            "age_beta": float(m1.params["age_c"]), "age_p": float(m1.pvalues["age_c"]),
            "model_r2": float(m1.rsquared),
        }
        rows.append(entry)
    results["models"] = rows

    (OUTDIR / "summary.json").write_text(json.dumps(results, indent=2))

    md = ["# Posterior-P300 chronotype effect adjusted for sex and age", "",
          f"N = {results['n']}. Coding: morning = 1, evening = 0; a positive "
          "`morning` coefficient means Morning shows a higher (less negative) "
          "loss-minus-gain P300, i.e. the primary effect direction.", "",
          "| Electrode | Unadj. β (Morning) | Unadj. p | Adj. β (Morning) | Adj. p | "
          "Adj. partial d | Sex p | Age p |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| {r['electrode'].split('_')[0]} | {r['unadjusted_beta_morning']:.3f} | "
            f"{r['unadjusted_p']:.4f} | {r['adjusted_beta_morning']:.3f} | "
            f"{r['adjusted_p']:.4f} | {r['adjusted_partial_d']:.2f} | "
            f"{r['sex_p']:.3f} | {r['age_p']:.3f} |")
    md += ["", "**Interpretation.** The chronotype (Morning vs Evening) effect on the "
           "posterior-P300 loss-minus-gain contrast is reported before and after "
           "adjusting for sex and age. If the adjusted `morning` coefficient stays "
           "large and significant while sex and age are non-significant, the P300 "
           "effect is not attributable to the sex/age imbalance between groups."]
    (OUTDIR / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nwrote", OUTDIR / "summary.json")


if __name__ == "__main__":
    main()
