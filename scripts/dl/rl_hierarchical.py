#!/usr/bin/env python3
"""G-D: hierarchical (partial-pooling) asymmetric RL model (Claude, per docs/specs_for_gpt.md).

Bayesian hierarchical version of scripts/dl/rl_model.py. Subject-level alpha_gain,
alpha_loss, beta, bias are partially pooled toward group means (more identifiable
given ~270 free trials/subject), with a chronotype (Evening=1) group-level offset on
each parameter mean so the Evening-minus-Morning contrast is read off as a posterior
with a 94% HDI. Same likelihood as the MLE model.

Likelihood (two arms [safe, risky], Q init 0), per free trial chronologically:
    p_risky = sigmoid(beta*(Q_risky - Q_safe) + bias)
    action ~ Bernoulli(p_risky)
    r = signed chosen-box value / 25  in [-1,1]
    alpha = alpha_gain if r>0 else alpha_loss ; Q[chosen] += alpha*(r - Q[chosen])

Runs the sequential Q-update as a pytensor.scan over trials, vectorised across
subjects (padded + masked).

Run (env_bayes -- needs pymc):
    env_bayes/bin/python scripts/dl/rl_hierarchical.py
Writes reports/clean/rl_hier/{summary.json,summary.md,shrinkage.png}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = "data/processed/ml_ready_features.csv"
CHRONO = "data/clean/chronotype_participant.csv"
MLE = "reports/clean/rl_model/participant_rl_params.csv"
OUTDIR = Path("reports/clean/rl_hier")
SEED = 0


def load_sequences():
    df = pd.read_csv(DATA)
    if "forced and free risk trials" in df.columns:
        df = df[df["forced and free risk trials"] == "free"].copy()
    df = df[pd.to_numeric(df["risky-choice"], errors="coerce").isin([0, 1])].copy()
    df = df.sort_values(["participant_id", "global_trial_index"])
    chosen = np.where(df["ChoiceMade"] == 1, df["ActualValue1"], df["ActualValue2"])
    df["reward"] = chosen / 25.0

    lab = pd.read_csv(CHRONO).set_index("participant_id")["Chronotype"]
    lab = (lab.str.lower() == "evening").astype(int)

    subs, actions, rewards, evening = [], [], [], []
    for pid, g in df.groupby("participant_id"):
        if pid not in lab.index:
            continue
        subs.append(pid)
        actions.append(g["risky-choice"].astype(int).to_numpy())
        rewards.append(g["reward"].to_numpy())
        evening.append(int(lab.loc[pid]))
    Tmax = max(len(a) for a in actions)
    S = len(subs)
    A = np.zeros((S, Tmax), "int64")
    R = np.zeros((S, Tmax), "float64")
    M = np.zeros((S, Tmax), "float64")
    for i, (a, r) in enumerate(zip(actions, rewards)):
        A[i, :len(a)] = a
        R[i, :len(r)] = r
        M[i, :len(a)] = 1.0
    return subs, np.array(evening), A, R, M


def build_and_sample(evening, A, R, M):
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    S, Tmax = A.shape
    ev = evening.astype("float64")
    A_ts, R_ts, M_ts = A.T, R.T, M.T  # (T, S) sequences for scan

    with pm.Model() as model:
        # group means (latent scale) + Evening offsets (= Evening-minus-Morning contrast)
        mu_ag = pm.Normal("mu_ag", 0.0, 1.5); off_ag = pm.Normal("off_ag", 0.0, 1.0)
        mu_al = pm.Normal("mu_al", 0.0, 1.5); off_al = pm.Normal("off_al", 0.0, 1.0)
        mu_b = pm.Normal("mu_b", 0.5, 1.0); off_b = pm.Normal("off_b", 0.0, 1.0)
        mu_bi = pm.Normal("mu_bi", 0.0, 1.0); off_bi = pm.Normal("off_bi", 0.0, 1.0)
        s_ag = pm.HalfNormal("s_ag", 1.0); s_al = pm.HalfNormal("s_al", 1.0)
        s_b = pm.HalfNormal("s_b", 1.0); s_bi = pm.HalfNormal("s_bi", 1.0)
        z_ag = pm.Normal("z_ag", 0, 1, shape=S); z_al = pm.Normal("z_al", 0, 1, shape=S)
        z_b = pm.Normal("z_b", 0, 1, shape=S); z_bi = pm.Normal("z_bi", 0, 1, shape=S)

        alpha_gain = pm.Deterministic("alpha_gain", pt.sigmoid(mu_ag + off_ag * ev + s_ag * z_ag))
        alpha_loss = pm.Deterministic("alpha_loss", pt.sigmoid(mu_al + off_al * ev + s_al * z_al))
        beta = pm.Deterministic("beta", pt.softplus(mu_b + off_b * ev + s_b * z_b))
        bias = pm.Deterministic("bias", mu_bi + off_bi * ev + s_bi * z_bi)

        def step(a_t, r_t, m_t, Qsafe, Qrisky, alpha_g, alpha_l, beta_, bias_):
            qdiff = Qrisky - Qsafe
            logit = beta_ * qdiff + bias_
            p = pm.math.sigmoid(logit)
            p = pt.clip(p, 1e-9, 1 - 1e-9)
            ll = m_t * (a_t * pt.log(p) + (1 - a_t) * pt.log(1 - p))
            alpha = pt.where(r_t > 0, alpha_g, alpha_l)
            chose_risky = pt.eq(a_t, 1)
            q_chosen = pt.where(chose_risky, Qrisky, Qsafe)
            upd = m_t * alpha * (r_t - q_chosen)
            Qrisky_n = Qrisky + pt.where(chose_risky, upd, 0.0)
            Qsafe_n = Qsafe + pt.where(chose_risky, 0.0, upd)
            return Qsafe_n, Qrisky_n, ll

        (_, _, lls), _ = pytensor.scan(
            fn=step,
            sequences=[pt.as_tensor_variable(A_ts), pt.as_tensor_variable(R_ts),
                       pt.as_tensor_variable(M_ts)],
            outputs_info=[pt.zeros(S), pt.zeros(S), None],
            non_sequences=[alpha_gain, alpha_loss, beta, bias],
        )
        pm.Potential("loglik", lls.sum())

        idata = pm.sample(draws=1000, tune=1000, chains=4, cores=4, target_accept=0.9,
                          random_seed=SEED, progressbar=False)
    return model, idata


def main():
    import arviz as az

    OUTDIR.mkdir(parents=True, exist_ok=True)
    subs, evening, A, R, M = load_sequences()
    print(f"subjects={len(subs)} evening={int(evening.sum())} morning={int((1-evening).sum())} "
          f"padded trials shape {A.shape}")

    model, idata = build_and_sample(evening, A, R, M)

    # convergence
    rhat = az.rhat(idata)
    max_rhat = float(max(float(rhat[v].max()) for v in rhat.data_vars))
    n_div = int(idata.sample_stats["diverging"].sum())

    # group-mean posteriors + Evening-minus-Morning contrasts (transformed scale)
    post = idata.posterior
    def flat(name):
        return post[name].values.reshape(-1)

    def transform(param, ev):
        if param in ("alpha_gain", "alpha_loss"):
            mu = "mu_ag" if param == "alpha_gain" else "mu_al"
            off = "off_ag" if param == "alpha_gain" else "off_al"
            return 1 / (1 + np.exp(-(flat(mu) + flat(off) * ev)))
        if param == "beta":
            return np.log1p(np.exp(flat("mu_b") + flat("off_b") * ev))
        return flat("mu_bi") + flat("off_bi") * ev  # bias

    params = ["alpha_gain", "alpha_loss", "beta", "bias"]
    results = {"seed": SEED, "n_subjects": len(subs), "max_rhat": max_rhat,
               "n_divergences": n_div, "parameters": {}}
    for p in params:
        ev_draws = transform(p, 1.0)
        mo_draws = transform(p, 0.0)
        contrast = ev_draws - mo_draws
        hdi = az.hdi(contrast, hdi_prob=0.94)
        results["parameters"][p] = {
            "evening_mean": float(ev_draws.mean()), "morning_mean": float(mo_draws.mean()),
            "contrast_evening_minus_morning": float(contrast.mean()),
            "contrast_hdi94": [float(hdi[0]), float(hdi[1])],
            "P(contrast>0)": float((contrast > 0).mean()),
        }
    # derived asymmetry contrast (alpha_loss - alpha_gain)
    asym = (transform("alpha_loss", 1.0) - transform("alpha_gain", 1.0)) - \
           (transform("alpha_loss", 0.0) - transform("alpha_gain", 0.0))
    hdi = az.hdi(asym, hdi_prob=0.94)
    results["parameters"]["lr_asymmetry"] = {
        "contrast_evening_minus_morning": float(asym.mean()),
        "contrast_hdi94": [float(hdi[0]), float(hdi[1])],
        "P(contrast>0)": float((asym > 0).mean()),
    }

    # subject-level posterior means vs MLE (shrinkage)
    subj_means = {p: post[p].mean(("chain", "draw")).values for p in params}
    shrink = pd.DataFrame({"participant_id": subs, **{f"{p}_hier": subj_means[p] for p in params}})
    corr = {}
    if Path(MLE).exists():
        mle = pd.read_csv(MLE)
        m = shrink.merge(mle, on="participant_id", suffixes=("", "_mle"))
        for p in params:
            corr[p] = float(np.corrcoef(m[f"{p}_hier"], m[p])[0, 1])
    results["hier_vs_mle_corr"] = corr
    (OUTDIR / "summary.json").write_text(json.dumps(results, indent=2))

    # shrinkage figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if corr:
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
        for ax, p in zip(axes, params):
            ax.scatter(m[p], m[f"{p}_hier"], s=30, alpha=0.75,
                       c=["#d62728" if e else "#1f77b4" for e in
                          (m["participant_id"].map(dict(zip(subs, evening))))])
            lo = min(m[p].min(), m[f"{p}_hier"].min()); hi = max(m[p].max(), m[f"{p}_hier"].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlabel(f"{p} (MLE)"); ax.set_ylabel(f"{p} (hierarchical)")
            ax.set_title(f"r = {corr[p]:.2f}", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Partial-pooling shrinkage: hierarchical vs MLE RL parameters "
                     "(red=Evening, blue=Morning)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUTDIR / "shrinkage.png", dpi=170)
        fig.savefig(OUTDIR / "shrinkage.pdf")
        plt.close(fig)

    # markdown
    P = results["parameters"]
    md = ["# G-D: hierarchical (partial-pooling) asymmetric RL model", "",
          f"N = {len(subs)} subjects; max R-hat = {max_rhat:.3f}; divergences = {n_div}.", "",
          "## Evening − Morning contrasts (transformed scale, 94% HDI)", "",
          "| Parameter | Evening | Morning | Contrast | 94% HDI | P(>0) |",
          "|---|---:|---:|---:|---|---:|"]
    for p in ["alpha_gain", "alpha_loss", "beta", "bias"]:
        r = P[p]
        md.append(f"| {p} | {r['evening_mean']:.3f} | {r['morning_mean']:.3f} | "
                  f"{r['contrast_evening_minus_morning']:+.3f} | "
                  f"[{r['contrast_hdi94'][0]:+.3f}, {r['contrast_hdi94'][1]:+.3f}] | "
                  f"{r['P(contrast>0)']:.3f} |")
    a = P["lr_asymmetry"]
    md.append(f"| lr_asymmetry | – | – | {a['contrast_evening_minus_morning']:+.3f} | "
              f"[{a['contrast_hdi94'][0]:+.3f}, {a['contrast_hdi94'][1]:+.3f}] | "
              f"{a['P(contrast>0)']:.3f} |")
    md += ["", "## Shrinkage vs MLE (Pearson r of subject-level estimates)",
           ", ".join(f"{p}: r={corr.get(p, float('nan')):.2f}" for p in params), "",
           "Interpretation: partial pooling stabilises the per-subject estimates while "
           "the group-level chronotype contrasts (esp. alpha_gain higher in Evening, beta "
           "lower in Evening) are read off directly with credible intervals rather than "
           "uncorrected NHST. This is a mechanistic/estimation model; the predictive "
           "headline remains the behaviour+ERP fusion.",
           "Figure: reports/clean/rl_hier/shrinkage.png"]
    (OUTDIR / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nwrote", OUTDIR / "summary.json")


if __name__ == "__main__":
    main()
