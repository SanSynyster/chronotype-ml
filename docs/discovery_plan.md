# Discovery Plan — Computational Analyses on the Feedback-EEG Decision Dataset

*Fresh start on the corrected data (see `docs/new_session_brief.md`). Chronotype is a
**null** side-variable; the goal is genuine signal in feedback processing and decision
dynamics, reported honestly.*

## Corrected foundation (done)
- `scripts/build_master.py` → `data/clean/participant_master.csv`. Authoritative key
  joined by `participant id`; **gender join conflicts = 0** (integrity verified).
- Cohort: 56 screened; **52 have EEG + behaviour**; 39 decisive Morning/Evening.
  Chronotype-agnostic work uses all 52 (more data than the old 39).

## Anchor signals (established on corrected data)
- **Risky choice is predictable above chance** from behavioural history: participant-
  grouped 5-fold CV, AUC 0.56 / balanced acc 0.54 (logistic + lag features; base rate
  0.53; majority 0.50). Sequence models expected to improve this.
- *(To reproduce next)* single-trial feedback-**valence** decoding from EEG (EEGNet
  gave ~AUC 0.64 in the buggy era; valence is label-independent so this should hold).

## Primary questions (pre-declared, real, label-independent)
1. **What does single-trial EEG carry about the outcome?** Decode feedback valence
   (gain/loss), correctness, and outcome magnitude cross- and within-subject
   (EEGNet / compact CNN / time-frequency). Establishes the neural signal ceiling.
2. **What predicts risky choice?** Sequence models (GRU/Transformer) + interpretable
   RL on the trial history; how far above the 0.56 anchor can we get, and what drives it.
3. **Brain → behaviour coupling.** Does single-trial feedback ERP (FRN/P300) or
   feedback-locked theta predict the *next* choice / RT / risk adjustment? Leakage-safe
   by construction (feedback precedes the next choice). This is the key mechanistic aim.

## Exploratory (flagged as such, corrected for multiplicity)
- Computational RL: hierarchical Bayesian fits; relate learning rates / exploration to
  neural signals.
- Neural decoding of decision variables (value, risk, chosen magnitude).
- Individual differences tested honestly (continuous MEQ — likely null; age; sex; risk
  propensity; choice consistency).
- Multimodal EEG+behaviour fusion for a *real* target (trial outcomes/choices).
- Self-supervised pretraining on all 52 EEG subjects, then probe.

## Evaluation protocol (non-negotiable)
Participant-grouped / leave-one-subject-out CV; no same-trial leakage; nested model
selection; permutation p-values with the full pipeline re-fit; bootstrap CIs; fixed
seeds; results to `reports/clean/<name>/summary.json`. Report nulls as nulls; resist
post-null fishing; genuine exploratory hits cross-validated on held-out folds.

## Roadmap
1. ✅ Corrected master + integrity.  2. ✅ Risky-choice anchor.
3. Reproduce EEG feedback-valence decoding on corrected `.set` epochs.
4. Sequence + RL models of choice (improve the anchor; extract mechanism).
5. Brain→behaviour single-trial coupling (the primary mechanistic result).
6. Consolidate into an honest findings doc; decide if a defensible paper exists
   (feedback-processing / decision-dynamics, not chronotype).
