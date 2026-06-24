#!/usr/bin/env python3
"""Asymmetric reward-learning (RL) model of risky choice, fit per participant.

Task: each free trial offers a low-magnitude (5) "safe" box and a high-magnitude
(25) "risky" box; signs are hidden and random, so there is no stable stimulus value
to learn -- a standard Q-learning-over-stimuli model is inappropriate. What CAN vary
between people is how strongly they update their risk preference from gains vs losses.

Model (2 arms: safe / risky), per participant:
    P(risky_t) = logistic( beta * (Q_risky - Q_safe) + bias )
    on the chosen arm:  Q <- Q + alpha * (r_t - Q),  with
        alpha = alpha_gain if r_t > 0 else alpha_loss
    r_t = signed value of the chosen box, scaled to [-1, 1] (/25).

Interpretable parameters per subject:
    alpha_gain, alpha_loss  -- learning rate from positive / negative outcomes
    lr_asymmetry = alpha_loss - alpha_gain  -- the key hypothesis (do evening types
                                               update more from losses?)
    beta  -- choice consistency / outcome sensitivity (inverse temperature)
    bias  -- baseline risk propensity

This connects directly to the feedback-locked FRN/P300 neural finding: both index how
feedback valence drives behavior.

Run (env_dl):
    python scripts/dl/rl_model.py
Writes reports/clean/rl_model/participant_rl_params.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA = "data/processed/ml_ready_features.csv"
OUTDIR = Path("reports/clean/rl_model")
SEED = 0
RESTARTS = 8


def _nll(params, actions, rewards):
    a_gain, a_loss, beta, bias = params
    q = np.zeros(2)  # [safe, risky]
    nll = 0.0
    for a, r in zip(actions, rewards):
        p_risky = 1.0 / (1.0 + np.exp(-(beta * (q[1] - q[0]) + bias)))
        p_risky = min(max(p_risky, 1e-9), 1 - 1e-9)
        nll -= np.log(p_risky if a == 1 else 1 - p_risky)
        alpha = a_gain if r > 0 else a_loss
        q[a] += alpha * (r - q[a])
    return nll


def fit_subject(actions, rewards, rng):
    bounds = [(0, 1), (0, 1), (0, 10), (-5, 5)]
    best = None
    for _ in range(RESTARTS):
        x0 = [rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 3), rng.uniform(-1, 1)]
        res = minimize(_nll, x0, args=(actions, rewards), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    a_gain, a_loss, beta, bias = best.x
    return {"alpha_gain": a_gain, "alpha_loss": a_loss,
            "lr_asymmetry": a_loss - a_gain, "beta": beta, "bias": bias,
            "nll": best.fun, "n_trials": len(actions)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA)
    df = df[df["forced and free risk trials"] == "free"].copy()
    df = df.sort_values(["participant_id", "global_trial_index"])
    # signed reward = chosen box value, scaled to [-1, 1]
    chosen = np.where(df["ChoiceMade"] == 1, df["ActualValue1"], df["ActualValue2"])
    df["reward"] = chosen / 25.0

    rows = []
    for pid, g in df.groupby("participant_id"):
        actions = g["risky-choice"].astype(int).to_numpy()
        rewards = g["reward"].to_numpy()
        p = fit_subject(actions, rewards, rng)
        p["participant_id"] = pid
        rows.append(p)

    out = pd.DataFrame(rows)[["participant_id", "alpha_gain", "alpha_loss",
                              "lr_asymmetry", "beta", "bias", "nll", "n_trials"]]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(OUTDIR / "participant_rl_params.csv", index=False)
    print(out.describe().round(3).to_string())
    print("\nwrote", OUTDIR / "participant_rl_params.csv")


if __name__ == "__main__":
    main()
