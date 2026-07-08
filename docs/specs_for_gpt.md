# Specs for GPT-owned analyses G-E and G-D (from Claude / C-SPEC)

These unblock the two GPT tasks that were waiting on data-loader details.
Everything below is verified against the current repo. Follow the coordination
contract in `docs/parallel_workplan.md` (branch `analysis-additions`, seed = 0,
env_bayes, write to `reports/clean/<name>/summary.json` + `.md`, do not edit
Claude-owned files). Put your scripts in `scripts/dl/`.

---

## G-E — P300 specification curve

**Goal:** show the primary Pz/POz P300 loss-minus-gain group effect is not an
artefact of one analysis window; recompute the group Cohen's d across a grid of
time windows and summary choices.

**Data / loader.** Use the existing epoch loader — do NOT re-read `.set` files by
hand:

```python
from scripts.dl.load_clean_epochs import load_all
d = load_all(tmin=-0.2, tmax=0.8, labeled_only=True)
# d["X"]         (N_trials, 64, 251) float32, microvolts, already baseline-corrected
# d["valence"]   (N,) 1=loss, 0=gain
# d["subject"]   (N,) participant id  -> aggregation group
# d["chronotype"](N,) 1=evening, 0=morning
# d["ch_names"]  list of 64 channel names ; d["times"] (251,) seconds
```

Load once with the widest window and slice windows from `d["times"]` in memory —
do not reload per window. **Assert** `"Pz"` and `"POz"` are in `d["ch_names"]`
before proceeding; if a name differs (e.g. case), stop and record it in the
summary rather than guessing.

**Per-trial amplitude → per-participant contrast → group d.** For a given
channel `ch` and window `[w0, w1]`:
1. per-trial amplitude = mean over samples with `w0 <= times <= w1` for that
   channel (or the peak variant, see grid);
2. per participant, `contrast_p = mean(amp | loss) - mean(amp | gain)`;
3. group effect = Cohen's d of `contrast_p` between Evening and Morning
   participants (Evening − Morning, to match the sign convention in
   `reports/clean/group_stats/`), with a bootstrap 95% CI (5000 resamples, seed 0)
   and a Welch p. The published anchor is Pz d ≈ −1.04, POz d ≈ −0.92 in the
   450–550 ms mean window — your grid must reproduce that cell as a sanity check.

**Grid (the "multiverse"):**
- channels: `Pz`, `POz`
- window centre: 400–600 ms; use sliding windows of width {50, 100 ms} stepped
  by 25 ms across that range (report each as `[w0, w1]`)
- summary: `mean` and `peak` (max positive amplitude in-window)
- The epochs are already baseline-corrected upstream, so do **not** add a second
  baseline; note this single baseline choice explicitly in the header rather than
  varying it.

**Outputs:** `reports/clean/spec_curve/summary.json` (one record per
channel×window×summary cell: window, d, ci_low, ci_high, welch_p, n_evening,
n_morning) + `summary.md` (how many cells reach d < −0.8 and p < .05; whether the
sign is stable) + `curve.png` (d ± CI across window centre, one line per
channel×summary, dashed line at the published anchor). Header docstring must state
the leakage stance: this is a descriptive group comparison, no CV, no prediction.

---

## G-D — Hierarchical (partial-pooling) reinforcement-learning model

**Goal:** replace the per-participant MLE RL fits with a Bayesian hierarchical
model so per-subject α_gain, α_loss, β, bias are shrunk toward group means
(more identifiable given ~270 free trials/subject), and report group-level
contrasts by chronotype with credible intervals.

**Likelihood — identical to the existing MLE model** (`scripts/dl/rl_model.py`,
which is the source of truth; do not change its parameterisation):

```
two arms: [safe, risky], Q initialised to 0
per free trial t (chronological):
    p_risky_t = logistic( beta * (Q_risky - Q_safe) + bias )
    choice a_t ~ Bernoulli(p_risky_t)          # a_t = 1 if risky chosen
    r_t = signed value of the CHOSEN box / 25   in [-1, 1]
    alpha = alpha_gain if r_t > 0 else alpha_loss
    Q[a_t] <- Q[a_t] + alpha * (r_t - Q[a_t])
```

**Data.** `data/processed/ml_ready_features.csv`, filtered to
`df["forced and free risk trials"] == "free"`, sorted by
`["participant_id", "global_trial_index"]`. Per participant:
`actions = df["risky-choice"].astype(int)`;
`chosen = where(ChoiceMade==1, ActualValue1, ActualValue2)`; `reward = chosen/25`.
(Exactly the preprocessing in `rl_model.py:70-85`.) Chronotype label per
participant from `data/clean/chronotype_participant.csv` (`Chronotype` column,
"Evening"/"Morning").

**Hierarchy (PyMC).** Non-centred parameterisation. Put subject-level params on
unconstrained scales then transform:
- `alpha_gain`, `alpha_loss` via `sigmoid(mu_a + sigma_a * z_a)` → (0,1)
- `beta` via `softplus`/`exp` of a normal → (0, ∞), keep comparable to the MLE
  bound β ∈ [0,10]
- `bias` normal.
Group means `mu_*` get weakly-informative priors; include a **chronotype
group-level offset** on each parameter mean so you can read off the Evening−Morning
contrast as a posterior with a 94% HDI. The per-trial recurrence (sequential Q
update) is the awkward part in PyMC — implement it with `pytensor.scan` over
trials within subject, or precompute nothing (the update depends on sampled
α). If `scan` proves too slow, a NumPyro backend or a per-subject vectorised
scan is acceptable; document whatever you choose.

**Inference.** NUTS, 4 chains, seed 0; report `r_hat` (flag any > 1.01) and
divergences. Compare posterior subject means to the MLE values in
`reports/clean/rl_model/participant_rl_params.csv` (correlation + a shrinkage
plot: MLE vs posterior, showing extreme MLE values pulled in).

**Outputs:** `reports/clean/rl_hier/summary.json` with, per parameter, the
group-mean posterior, the Evening−Morning contrast (mean + 94% HDI + P(contrast
> 0)), max r_hat, n_divergences; plus `summary.md` narrating whether the
MLE-based conclusions (Evening higher α_gain, lower β) survive partial pooling;
and `shrinkage.png`. Header docstring: state that this is a mechanistic/estimation
model, group contrasts are **not** the predictive headline (that remains the
fusion), and that these contrasts are reported with HDIs, not as corrected NHST.

---

## Notes shared by both

- Run from repo root so the `scripts.dl...` import path resolves
  (`PYTHONPATH=.` or `python -m scripts.dl.p300_spec_curve`).
- `load_clean_epochs` imports `mne`; make sure env_bayes has it, or run G-E in
  `env_dl` (it needs no pymc) and G-D in env_bayes. Your call — just record which.
- Sanity-check counts against the manuscript: 39 labelled participants, 19
  Evening / 20 Morning, ~13,522 labelled epochs.
