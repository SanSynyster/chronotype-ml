# New-Session Brief — Computational Discovery on the Feedback-EEG Decision Dataset

*Paste this as the opening prompt for a fresh session. It is self-contained: assume
no memory of prior conversations. Read it fully before acting.*

---

## 0. Your mission

Study this project (code, docs, git history, and the **corrected** data), then design
and run a broad, rigorous program of computational analyses — machine learning, deep
learning, reinforcement-learning modelling, and multimodal fusion — to find a
**genuine, defensible result** in this dataset. "Find something" means *rigorously
discover whatever real signal exists and report it honestly, including nulls* — it
does **not** mean manufacture a finding. A prior line of work on this data collapsed
because of a data bug (below); the bar now is correctness and honesty first.

## 1. The dataset

EEG + behaviour from a **feedback-based risky decision-making task**. On each trial two
boxes show magnitudes {5, 25}; the sign (gain/loss) is hidden at choice and revealed as
feedback. Choosing the high-magnitude box = the "risky" choice. Feedback is crossed by
**valence (gain/loss) × correctness (correct/error)** → four conditions
(Gain-Correct, Gain-Error, Loss-Correct, Loss-Error). ~384 trials/participant, 16
blocks. 64-ch EEG (ANT Neuro, 10-10, 1 kHz). ~52 participants have EEG; 39 have a
decisive (non-intermediate) chronotype label; MEQ + MCTQ available.

## 2. What happened before (READ — do not repeat, do not chase ghosts)

The prior analysis reported a large chronotype effect (posterior-P300 loss-minus-gain
d ≈ 1.0; behaviour + fusion decoding chronotype at AUC ~0.8). **It was an artifact of a
data-linkage bug:** the behavioural files and the chronotype/MEQ metadata shared no
participant ID, so the pipeline matched them *statistically*, which **scrambled the
chronotype label for 14/39 participants and the MEQ for 31/39**. With the corrected
labels, **every chronotype result is null**:

| Result | Buggy | Corrected |
|---|---|---|
| Pz P300 loss−gain | d=−1.04, p=.003 | d≈0.08, p=.81 |
| Behaviour risky rate | d=+0.80, p=.02 | d=−0.31, p=.34 |
| Chronotype decoding (GRU/fusion) | AUC 0.71 / 0.80 | AUC ≈ 0.2, p≈.9 |
| Continuous MEQ ↔ P300 | r=0.29 | r≈−0.05, p=.75 |

**Conclusion: there is no detectable chronotype effect in this cohort.** Do not build
the new work around chronotype. Test it once, honestly, and move on.

## 3. Data — the single source of truth (never re-link statistically)

- **Identity/labels key:** `data/raw/meq mctq scores - Sheet1.csv` — authoritative for
  chronotype, MEQ, MCTQ, gender, age, EEG availability, keyed by the integer
  **`participant id`** (1001-1056). **Always join by this ID. Never statistically match.**
- **Behaviour:** `data/raw/all behavioral-2.xlsx` (~19,968 trial rows, keyed by
  `UserID`). Trust the trial columns; take chronotype/MEQ/age from the key, not this
  file's own columns.
- **ERP (condition-averaged window means, per participant):**
  `data/raw/frn_all_25-_350.xlsx` (FRN window) and `data/raw/p300_all_350_450.xlsx`
  (P300 window). Id column `ERPset` (strip leading whitespace). Columns
  `bin##_<risk>_<condition>__<channel>`; channels {Fz,FC1,FC2,Cz,Pz,POz,FCz}.
- **Single-trial EEG (for DL):** `data/raw/shifted_set/*.set` (EEGLAB epochs; loader
  pattern in `scripts/dl/load_clean_epochs.py`). Also `data/raw/_singletrial_means/`.
- **Questionnaire source:** `data/raw/MCTQ & MEQ (Responses) ...csv`.
- The corrected rebuild **scripts** and reports live on branch **`data-rebuild`**
  (`reports/clean/rebuild/old_vs_new.md`, `integrity.md`). Regenerate clean tables from
  raw with the corrected linkage; do not trust any stale derived file.

## 4. Non-negotiable rules

1. **Correct linkage only** — join everything by `participant id`; assert IDs unique and
   present on both sides; cross-check gender as an integrity signal; fail loudly on
   identity-critical disagreement (age ±1-2yr is a benign known warning).
2. **Leakage-safe** — participant-grouped CV; never split a participant across
   train/test; never use same-trial outcome/feedback to predict the same-trial choice.
3. **Permutation-clean** — model selection inside nested CV; permutation p-values with
   the full pipeline re-fit; report bootstrap CIs.
4. **Honest reporting** — report nulls as nulls. Pre-declare a primary vs exploratory
   split. Correct for multiple comparisons across the family of models/features. After a
   null primary, resist fishing; a genuine exploratory finding must be flagged as such
   and ideally cross-validated on held-out folds.
5. **Reproducible** — fixed seeds; scripts, not notebooks; write results to
   `reports/clean/<name>/summary.json`.

## 5. Where the real signal likely is (build here, not on chronotype)

Two effects were genuine even in the buggy era because they don't depend on the (null)
chronotype label:
- **Single-trial feedback decoding** — EEGNet decoded feedback **valence** cross-subject
  at ~AUC 0.64. Real.
- **Trial-level risky-choice prediction** — a causal GRU predicted risky choice from
  behavioural dynamics at ~AUC 0.65. Real, label-independent.

Candidate research directions (propose your own too), roughly strongest-first:

1. **Single-trial EEG decoding of decision/feedback variables** (valence, correctness,
   outcome magnitude, reward-prediction-error, risk level) — EEGNet / compact CNNs /
   time-frequency features; within- and cross-subject; establish what the EEG carries.
2. **Neural correlates of decision variables** — decode expected value, risk, chosen
   magnitude from pre-/post-choice EEG; representational/temporal-generalization analysis.
3. **Brain→behaviour coupling** — does single-trial feedback ERP (FRN/P300) or
   feedback-locked theta predict the *next* choice / RT / risk adjustment? A genuine
   mechanistic question, leakage-safe by construction (feedback precedes next choice).
4. **Computational RL modelling of choice** — fit (hierarchical Bayesian) RL models;
   characterise individual differences in gain/loss learning, exploration, risk
   sensitivity; test whether neural signals (ERP amplitude, theta) relate to fitted
   learning rates.
5. **Sequence models of choice dynamics** — GRU/Transformer on the trial sequence to
   predict choice/RT and to extract interpretable behavioural embeddings; what temporal
   structure drives risky choice.
6. **Multimodal fusion done for a real target** — combine EEG + behaviour to predict
   trial-level outcomes/choices (not the null chronotype).
7. **Individual differences, tested honestly** — can any stable trait (continuous MEQ,
   age, sex, overall risk propensity, choice consistency) be predicted from neural/
   behavioural features under strict CV? Report nulls (MEQ is likely null).
8. **Self-supervised / representation learning** on the full EEG set (incl. unlabeled
   subjects) for feedback processing, then probe.

The most publishable honest outcome may be a **feedback-processing / decision-dynamics**
paper (what the brain and behaviour reveal about risky-choice feedback evaluation),
with chronotype reported as a null side-analysis — *not* a chronotype paper.

## 6. Deliverables

1. A short **research plan** (`docs/discovery_plan.md`): the questions, why they're real,
   the primary/exploratory split, and the evaluation protocol.
2. A **corrected clean-data rebuild** you trust (from raw, via the `data-rebuild` scripts).
3. Iterative, honest **exploratory results** with permutation p-values and CIs, written
   to `reports/clean/` and summarised in a running findings doc.
4. A frank verdict: what real signal exists, what is null, and whether there is a
   defensible paper here.

## 7. First steps

1. Read `docs/` (esp. `methodology_dl.md`, `results_dl.md`, `discussion_dl.md`,
   `data_provenance.md`) and `scripts/`, `scripts/dl/` to learn the pipeline and prior
   methods — but treat all prior *chronotype results as invalid*.
2. Check out / diff the `data-rebuild` branch to get the corrected linkage code.
3. Rebuild the clean participant + trial tables from raw with correct linkage; verify
   integrity (gender-matched join, N per group).
4. Reproduce the two real baselines (feedback-valence decoding; risky-choice GRU) on the
   corrected data to anchor, then pursue §5.
```

*Old buggy data has been deleted from `data/`. Authoritative raw files only.*
