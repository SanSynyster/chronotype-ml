#!/usr/bin/env python3
"""C-A: single-trial feedback P300 -> next-trial risky choice, moderated by chronotype.

New mechanistic result linking the neural finding to behaviour at the single-trial
level. The feedback-locked P300 on trial t is a POST-feedback signal that occurs
after the choice on trial t but BEFORE the choice on trial t+1, so using P300(t)
to predict risky-choice(t+1) is causally valid -- no leakage (unlike using same-trial
feedback to predict the same-trial choice).

Question: does trial-to-trial fluctuation in the feedback P300 amplitude predict how
the participant adjusts risk on the next trial, and does that coupling differ between
Morning and Evening chronotypes?

Design (leakage-safe):
  * Pair each trial t to the immediately-following trial t+1 within the same
    participant (global_trial_index diff == 1). The current (t+1) trial must be a
    FREE trial (risky-choice defined).
  * Predictor: Pz_P300 on trial t, z-scored WITHIN participant so the coupling
    reflects trial-to-trial variation, not between-subject amplitude offsets.
    (POz_P300 run as a secondary electrode.)
  * Target: risky-choice on trial t+1 (1 = chose high-magnitude box).
  * Controls: previous risky-choice (t), previous feedback valence (loss at t),
    trial progress.

Two analyses:
  (1) Two-stage (transparent, robust): fit a per-participant logistic slope of
      next-risky on P300z, then compare the slope distribution Evening vs Morning
      (Mann-Whitney + Cohen's d with bootstrap CI). Headline test.
  (2) Confirmatory GLMM: Binomial mixed model with a P300z x chronotype fixed
      interaction and random intercept + random P300z slope by participant.

Run (env, Python 3.11 -- needs statsmodels):
    env/bin/python scripts/dl/p300_risk_coupling.py
Writes reports/clean/p300_coupling/{summary.json,summary.md} and a figure into
docs/figures/fig10_p300_risk_coupling.{png,pdf}.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATA = "data/processed/ml_ready_features.csv"
CHRONO = "data/clean/chronotype_participant.csv"
OUTDIR = Path("reports/clean/p300_coupling")
FIGDIR = Path("docs/figures")
SEED = 0
N_BOOT = 5000


def build_pairs(electrode: str) -> pd.DataFrame:
    """Return one row per (t -> t+1) consecutive pair with a valid P300 on t."""
    df = pd.read_csv(DATA)
    lab = pd.read_csv(CHRONO).set_index("participant_id")["Chronotype"]
    lab = (lab.str.lower() == "evening").astype(int)  # 1 evening, 0 morning

    df = df.sort_values(["participant_id", "global_trial_index"]).reset_index(drop=True)
    # feedback valence at trial t: loss = feedback-condition in loss_*
    fc = df["feedback-condition"].astype(str).str.lower()
    df["loss_t"] = fc.str.contains("loss").astype(float)

    rows = []
    for pid, g in df.groupby("participant_id"):
        if pid not in lab.index:
            continue
        g = g.sort_values("global_trial_index")
        p300 = g[electrode].to_numpy(dtype=float)
        gti = g["global_trial_index"].to_numpy()
        risky = g["risky-choice"].to_numpy()  # NaN on forced trials
        free = (g["forced and free risk trials"].astype(str) == "free").to_numpy()
        loss_t = g["loss_t"].to_numpy()
        prog = (gti - gti.min()) / max(1, (gti.max() - gti.min()))
        for i in range(len(g) - 1):
            # consecutive trials, current-t P300 present, next-trial free w/ choice
            if gti[i + 1] - gti[i] != 1:
                continue
            if not np.isfinite(p300[i]):
                continue
            if not free[i + 1] or not np.isfinite(risky[i + 1]):
                continue
            if not np.isfinite(risky[i]):
                continue
            rows.append({
                "participant_id": pid,
                "chronotype": int(lab.loc[pid]),
                "p300_t": p300[i],
                "prev_risky": int(risky[i]),
                "loss_t": float(loss_t[i]),
                "progress": float(prog[i + 1]),
                "next_risky": int(risky[i + 1]),
            })
    pairs = pd.DataFrame(rows)
    # within-participant z-score of P300 (isolate trial-to-trial variation)
    pairs["p300z"] = pairs.groupby("participant_id")["p300_t"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0)
    )
    return pairs


def per_subject_slopes(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-participant coupling coefficients of next_risky on the feedback P300.

    Two coefficients per participant:
      * slope       -- overall coupling: next_risky ~ p300z + progress
      * valence_int -- valence-resolved coupling: the p300z:loss_t interaction from
                       next_risky ~ p300z * loss_t + progress. This is the single-trial
                       analogue of the significant loss-minus-gain P300 contrast: how
                       much more strongly the feedback P300 drives next-trial risk after
                       losses than after gains.
    """
    import statsmodels.api as sm

    out = []
    for pid, g in pairs.groupby("participant_id"):
        if g["next_risky"].nunique() < 2 or len(g) < 15:
            continue  # need outcome variation and enough trials to estimate a slope
        y = g["next_risky"].to_numpy()
        rec = {"participant_id": pid, "chronotype": int(g["chronotype"].iloc[0]), "n": len(g)}
        # overall coupling
        try:
            X = sm.add_constant(g[["p300z", "progress"]].to_numpy())
            res = sm.Logit(y, X).fit(disp=0, method="lbfgs", maxiter=200)
            slope = float(res.params[1])
            rec["slope"] = slope if (np.isfinite(slope) and abs(slope) < 20) else np.nan
        except Exception:
            rec["slope"] = np.nan
        # valence-resolved coupling (p300z x loss_t), needs both valences present
        try:
            if g["loss_t"].nunique() == 2:
                d = g.copy()
                d["p300z_x_loss"] = d["p300z"] * d["loss_t"]
                Xv = sm.add_constant(d[["p300z", "loss_t", "p300z_x_loss", "progress"]].to_numpy())
                resv = sm.Logit(y, Xv).fit(disp=0, method="lbfgs", maxiter=200)
                vi = float(resv.params[3])
                rec["valence_int"] = vi if (np.isfinite(vi) and abs(vi) < 20) else np.nan
            else:
                rec["valence_int"] = np.nan
        except Exception:
            rec["valence_int"] = np.nan
        out.append(rec)
    return pd.DataFrame(out)


def two_stage_on(slopes: pd.DataFrame, col: str, rng) -> dict:
    s = slopes.dropna(subset=[col])
    ev = s.loc[s.chronotype == 1, col].to_numpy()
    mo = s.loc[s.chronotype == 0, col].to_numpy()
    u, p = stats.mannwhitneyu(ev, mo, alternative="two-sided")
    d = cohens_d(ev, mo)
    lo, hi = boot_d_ci(ev, mo, rng)
    t_ev, p_ev = stats.ttest_1samp(ev, 0.0)
    t_mo, p_mo = stats.ttest_1samp(mo, 0.0)
    return {
        "coefficient": col,
        "n_evening": int(len(ev)), "n_morning": int(len(mo)),
        "evening_mean": float(np.mean(ev)), "morning_mean": float(np.mean(mo)),
        "evening_sd": float(np.std(ev, ddof=1)), "morning_sd": float(np.std(mo, ddof=1)),
        "cohens_d_evening_minus_morning": float(d), "d_ci95": [lo, hi],
        "mannwhitney_u": float(u), "mannwhitney_p": float(p),
        "evening_vs0_p": float(p_ev), "morning_vs0_p": float(p_mo),
    }


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan


def boot_d_ci(a, b, rng, n=N_BOOT):
    ds = []
    a, b = np.asarray(a), np.asarray(b)
    for _ in range(n):
        ds.append(cohens_d(rng.choice(a, len(a), replace=True),
                           rng.choice(b, len(b), replace=True)))
    return float(np.nanpercentile(ds, 2.5)), float(np.nanpercentile(ds, 97.5))




def glmm(pairs: pd.DataFrame) -> dict:
    """Confirmatory Binomial mixed model with P300z x chronotype interaction."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = pairs.copy()
    d["pid"] = d["participant_id"].astype("category")
    formula = "next_risky ~ p300z * chronotype + prev_risky + loss_t + progress"
    # random intercept + random p300z slope by participant
    vc = {"subj": "0 + C(pid)", "subj_slope": "0 + C(pid):p300z"}
    try:
        model = BinomialBayesMixedGLM.from_formula(formula, vc, d)
        res = model.fit_vb()
        names = list(res.model.exog_names)
        idx = names.index("p300z:chronotype")
        coef = float(res.fe_mean[idx])
        sd = float(res.fe_sd[idx])
        z = coef / sd if sd > 0 else np.nan
        p = float(2 * stats.norm.sf(abs(z)))
        main_idx = names.index("p300z")
        return {
            "converged": True,
            "interaction_coef_p300z_x_chronotype": coef,
            "interaction_sd": sd, "interaction_z": float(z), "interaction_p": p,
            "p300z_main_coef": float(res.fe_mean[main_idx]),
            "p300z_main_sd": float(res.fe_sd[main_idx]),
        }
    except Exception as e:  # GLMM is the fragile one; two-stage is the headline
        return {"converged": False, "error": str(e)}


def make_figure(slopes: pd.DataFrame, ts: dict, path_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharex=True)
    rng = np.random.default_rng(SEED)
    panels = [("slope", ts["overall"], "Overall coupling\n(P300$_t$ → next risky, logit)"),
              ("valence_int", ts["valence"],
               "Valence-resolved coupling\n(P300$_t$×loss interaction, logit)")]
    for ax, (col, res, ylab) in zip(axes, panels):
        s = slopes.dropna(subset=[col])
        for x, lab, c in [(0, "Morning", "#1f77b4"), (1, "Evening", "#d62728")]:
            v = s.loc[s.chronotype == x, col].to_numpy()
            jit = x + (rng.random(len(v)) - 0.5) * 0.16
            ax.scatter(jit, v, s=32, alpha=0.75, color=c, edgecolor="white", linewidth=0.6, zorder=3)
            ax.hlines(np.mean(v), x - 0.22, x + 0.22, color=c, linewidth=2.4, zorder=4)
        ax.axhline(0, color="grey", linewidth=0.9, linestyle="--", zorder=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Morning", "Evening"])
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(f"d = {res['cohens_d_evening_minus_morning']:.2f} "
                     f"[{res['d_ci95'][0]:.2f}, {res['d_ci95'][1]:.2f}], "
                     f"p = {res['mannwhitney_p']:.3f}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Feedback P300 → next-trial risk coupling by chronotype (Pz)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    fig.savefig(path_png.with_suffix(".pdf"))
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    results = {"seed": SEED, "n_boot": N_BOOT, "electrodes": {}}
    for elec in ["Pz_P300", "POz_P300"]:
        pairs = build_pairs(elec)
        slopes = per_subject_slopes(pairs)
        ts = {"overall": two_stage_on(slopes, "slope", rng),
              "valence": two_stage_on(slopes, "valence_int", rng)}
        gm = glmm(pairs) if elec == "Pz_P300" else {"skipped": "primary electrode only"}
        results["electrodes"][elec] = {
            "n_pairs": int(len(pairs)),
            "n_participants_with_slope": int(slopes["slope"].notna().sum()),
            "two_stage": ts,
            "glmm": gm,
        }
        if elec == "Pz_P300":
            make_figure(slopes, ts, FIGDIR / "fig10_p300_risk_coupling.png")

    (OUTDIR / "summary.json").write_text(json.dumps(results, indent=2))

    # human-readable summary
    pz = results["electrodes"]["Pz_P300"]
    ov = pz["two_stage"]["overall"]; va = pz["two_stage"]["valence"]; gm = pz["glmm"]

    def blk(res, name):
        return [
            f"### {name}",
            f"- Evening mean = {res['evening_mean']:.3f} (SD {res['evening_sd']:.3f}, n={res['n_evening']}); "
            f"Morning mean = {res['morning_mean']:.3f} (SD {res['morning_sd']:.3f}, n={res['n_morning']})",
            f"- Group difference: Cohen's d = {res['cohens_d_evening_minus_morning']:.3f} "
            f"[{res['d_ci95'][0]:.3f}, {res['d_ci95'][1]:.3f}], Mann-Whitney p = {res['mannwhitney_p']:.4f}",
            f"- Coupling vs 0: Evening p = {res['evening_vs0_p']:.3f}, Morning p = {res['morning_vs0_p']:.3f}",
            "",
        ]

    md = [
        "# C-A: single-trial feedback P300 -> next-trial risk coupling by chronotype",
        "",
        "Leakage-safe: the feedback-locked P300 at trial t predicts the *next* trial's "
        "risky choice (P300 occurs after the t choice, before the t+1 choice).",
        f"Pairs (Pz): {pz['n_pairs']} consecutive t->t+1 transitions; "
        f"{pz['n_participants_with_slope']} participants with an estimable coupling.",
        "",
        "## Two-stage (per-subject coupling, then group contrast) -- Pz",
    ]
    md += blk(ov, "Primary: overall coupling (P300 -> next risky)")
    md += blk(va, "Theory-matched: valence-resolved coupling (P300 x loss interaction)")
    md += ["## Confirmatory GLMM (Pz, Binomial mixed model, overall coupling)"]
    if gm.get("converged"):
        md.append(f"- P300z x chronotype interaction = {gm['interaction_coef_p300z_x_chronotype']:.3f} "
                  f"(SD {gm['interaction_sd']:.3f}, z = {gm['interaction_z']:.2f}, "
                  f"p = {gm['interaction_p']:.4f})")
    else:
        md.append(f"- GLMM did not converge cleanly ({gm.get('error','')[:80]}); "
                  f"two-stage result above is the headline.")
    md += ["",
           "**Reading:** a between-subject trait effect (aggregate P300 loss-minus-gain) can "
           "coexist with a weak within-subject single-trial coupling; report faithfully whichever "
           "way these land -- they bound the mechanistic claim and pre-empt the 'is it just a "
           "between-subject correlation?' reviewer question.",
           "",
           "Secondary electrode POz in summary.json. Figure: docs/figures/fig10_p300_risk_coupling.png"]
    (OUTDIR / "summary.md").write_text("\n".join(md))

    print("\n".join(md))
    print("\nwrote", OUTDIR / "summary.json")


if __name__ == "__main__":
    main()
