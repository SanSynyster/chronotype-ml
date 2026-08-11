# Paper Architecture — Integrated Chronotype Manuscript (target: Psychophysiology)

Agreed structure for `docs/manuscript_draft.md`. Psychophysiology uses an
unstructured abstract (~250 words), full IMRaD, and endorses the SPR committee
ERP-reporting guidelines (Keil et al., 2014). Lead with the P300 (the journal's
core cognitive-neuroscience audience) and build to the super-additive fusion as
the payoff. Prose is owned by the Claude agent; tooling/figures/refs/supplementary
by the Codex agent.

## Title & abstract
- Title: *Chronotype in the Neural and Behavioural Evaluation of Decision Feedback:
  Converging ERP, Computational, and Fusion Evidence.*
- Abstract ~250 words, unstructured; keywords set.

## 1. Introduction (~700 words)
- Chronotype ↔ reward sensitivity, impulsivity, risky decision-making [CITATION].
- Feedback ERPs: FRN (frontocentral, ~250–300 ms, reward prediction error) and
  P300 (parietal/posterior, motivational salience / outcome evaluation) [CITATION].
- Gap: feedback-locked neural signatures distinguishing morning/evening types are
  underexplored; few studies pair ERP with rigorous, interpretable ML.
- Present study: single cohort; twofold mutually-validating approach (theory-driven
  group comparison + leakage-safe computational modelling of choice dynamics and
  brain–behaviour fusion).
- Hypotheses: H1 posterior-P300 loss-vs-gain differs by chronotype; H2 chronotype is
  decodable from risky-choice dynamics; H3 behaviour + ERP fuse convergently.

## 2. Methods
Participants · sample & power · task · EEG acquisition · preprocessing · chronotype
labels + linkage + MEQ validation · feature engineering · ERP/behaviour group stats
· computational models (§2.7 GRU, §2.8 asymmetric + hierarchical RL, §2.9 EEGNet,
§2.10 shared decoding evaluator + fusion + continuous-MEQ, §2.11 single-trial
coupling) · sensitivity analyses.

## 3. Results (narrative arc)
1. Posterior P300 distinguishes chronotypes (**Fig 1**)
2. Behavioural risk-taking (supporting)
3. Continuous-MEQ confirmation (**Fig 2**)
4. Robustness across exclusions + specification curve
5. Interpretable classifier converges on the P300
6. Chronotype decodable from choice dynamics (GRU)
7. **Super-additive behaviour + ERP fusion (Fig 3, HEADLINE)**
8. Continuous-MEQ prediction from the same features
9. Reinforcement-learning mechanism (**exploratory** — did not survive pooling)
10. EEGNet honest negative
11. Single-trial P300→choice coupling null (**Fig 4**, trait-level localisation)
12. Secondary: trial-level risky choice

## 4. Discussion
Principal findings · posterior-P300 interpretation · brain–behaviour convergence
(central strength) · tentative mechanism (with hierarchical caveat) · two honest
negatives as a methodological message for small-n individual-differences EEG ·
dimensional trait (continuous MEQ) · limitations · conclusion.

## Figures — consolidate 10 → ~5 main + supplementary
- **Main:** Fig 1 P300-by-chronotype (+ spec-curve panel) · Fig 2 continuous-MEQ ·
  **Fig 3 fusion (headline)** · Fig 4 single-trial coupling · Fig 5 ROC/pipeline.
- **Supplementary (S1..Sn):** sensitivity forest · confusion matrix · feature
  importance · risky-choice baselines · full specification curve · hierarchical-RL
  shrinkage · EEGNet decoding · Bayes factors · equivalence tests.

## Status of inputs
- Analyses: complete (all `[PENDING GPT]` slots closed).
- Blocking before submission: citations (sign-off on `citation_candidates.md`),
  team-only Methods/ethics (`coauthor_request.md`), positive-control AUC
  reconciliation, final statcheck.
