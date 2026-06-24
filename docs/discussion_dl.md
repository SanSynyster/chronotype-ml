# Discussion — Computational and Deep-Learning Analyses of Chronotype, Feedback Processing, and Risky Choice

*Draft for co-author review (Sahab → Mahsima). Companion to `methodology_dl.md` and
`results_dl.md`. Citations are flagged as [REF] for Mahsima to insert from the primary
manuscript's reference set.*

---

## 1. Principal findings

Across an internally-validated, leakage-safe, permutation-clean analysis of 39 participants, we
found that **chronotype is expressed in the dynamics of feedback-based risky choice and in the
feedback-locked P300/FRN, and that these behavioural and neural signals carry partly independent
information**. Specifically: (i) the temporal dynamics of risky choice predicted Morning vs
Evening (AUC 0.713) and the continuous MEQ (r = 0.31); (ii) the validated ERP feedback contrasts
predicted chronotype independently (AUC 0.668); (iii) the two combined super-additively
(AUC 0.797, p = 0.004; bootstrap CI [0.64, 0.92]); and (iv) a reinforcement-learning model
localised the behavioural effect to chronotype differences in **gain-driven learning** and
**choice consistency**. End-to-end deep learning on single-trial EEG did not recover chronotype
at this sample size.

---

## 2. A mechanistic account of chronotype and reward-based choice

The reinforcement-learning model moves the result beyond classification toward mechanism.
Evening types showed **higher gain learning rates** (α_gain) and **lower choice consistency**
(β, i.e. more exploratory/variable choosing), whereas Morning types showed a relative bias
toward **learning from losses** and more deterministic choosing. Read together with the
descriptive finding that Evening types take the high-magnitude ("risky") option more often, this
is consistent with a profile of **stronger approach/reward sensitivity and weaker
loss-avoidance** in Evening chronotypes [REF].

This dovetails with two literatures the primary manuscript already engages:

- **Chronotype, reward sensitivity and risk.** Evening-ness has been associated with greater
  reward sensitivity, impulsivity, and risk-taking, and with differences in approach motivation
  [REF]. A gain-weighted, more exploratory learning style is a natural computational expression
  of that profile.
- **Circadian modulation of dopaminergic reward processing.** Reward learning and its electro-
  physiological signatures are under circadian/dopaminergic influence [REF]; chronotype, as a
  stable individual difference in circadian phase preference, plausibly indexes tonic differences
  in how feedback is weighted. The asymmetry between gain- and loss-learning is exactly where such
  modulation would be expected to surface.

Importantly, the RL parameters **jointly predicted chronotype only weakly** (AUC 0.53). We
therefore interpret them as a mechanistic, convergent account of *why* the behavioural signal
exists, not as a competing predictor — the richer recurrent embedding (which is not constrained
to a four-parameter model) remains the stronger predictor. The division is deliberate: the GRU
answers "can we predict?", the RL model answers "what is differing?".

---

## 3. Neural feedback processing and brain–behaviour convergence

The feedback P300/FRN contrasts predicted chronotype on their own, replicating the primary ERP
group difference within a predictive framework. The posterior/parietal P300 to feedback is
commonly interpreted as indexing motivational salience and the evaluation of outcome
significance [REF], and the FRN as a reward-prediction-error signal [REF]; chronotype differences
in these contrasts therefore align with the behavioural evidence for differential outcome
weighting.

The most informative result is that **fusion was super-additive**: combining behaviour and the
ERP contrasts exceeded either alone and roughly doubled the correlation with continuous MEQ
(−0.42). If the two modalities indexed the same latent variable, fusion would have been
redundant; instead, behaviour (how choices evolve over many trials) and the P300 (the immediate
neural evaluation of each outcome) appear to capture **complementary facets** of the same
chronotype difference. This is the kind of cross-modal convergence that strengthens an
individual-differences claim well beyond a single-modality classifier.

---

## 4. Why deep learning on single-trial EEG did not work — and why that is informative

A compact convolutional network (EEGNet) decoded the feedback task itself across unseen
participants (AUC 0.64), confirming the cleaned epochs carry genuine single-trial signal, yet it
failed to recover chronotype from learned single-trial features (AUC ~0.4). The contrast with the
hand-crafted ERP features, which *did* predict chronotype, points to a clear interpretation:
**at N = 39 there are far too few participants for a network to learn the subtle, subject-level
chronotype difference**, whereas a small set of theory-driven, FDR-validated ERP contrasts encodes
it efficiently. Consistent with this, fusing the *learned* EEG embedding hurt performance
(0.65) while fusing the *validated* contrasts helped (0.80).

This is a useful, honest methodological message for the field: for small-sample individual-
differences EEG, validated low-dimensional features can outperform end-to-end representation
learning, and naive feature learning can actively dilute a real effect. We report the negative
result transparently rather than omitting it.

---

## 5. The trait is dimensional

Because 12 of 39 participants fall in the MEQ intermediate band, the binary Morning/Evening split
is necessarily soft. That the same features predicted the **continuous MEQ** score (fused
r = 0.34, p = 0.027) indicates the effect is not an artefact of dichotomisation and supports a
dimensional view of chronotype's relationship to reward-based choice. Future work should model
the continuous score as the primary outcome.

---

## 6. Limitations

- **Sample size and single cohort.** N = 39 from one cohort; all predictive findings are
  **internally validated only** and should be read as preliminary, hypothesis-strengthening
  evidence rather than established effects. No independent replication sample is available.
- **No external validation.** Robustness was established *within* the sample (bootstrap,
  exclusions, leave-one-subject-out) but not against a new dataset.
- **RL-parameter inference is exploratory.** The five parameter comparisons are uncorrected for
  multiple comparisons, and some participants' parameters approached the optimisation bounds
  (limited identifiability given ~270 free trials each); the RL results should be treated as a
  mechanistic interpretation, not confirmatory tests.
- **Influence of individual participants.** Although no single participant reduced the fused AUC
  to chance, removing the most influential participant lowered it from 0.80 to 0.65; the estimate
  carries real uncertainty (bootstrap CI [0.64, 0.92]).
- **Task-specificity.** The findings concern one feedback-based risky-choice paradigm with hidden
  signs; generalisation to other reward/decision tasks is untested.
- **Preprocessing provenance.** The EEG cleaning/epoching pipeline was performed upstream; the
  exact filter, ICA/artefact-rejection, and final-trial-count details should be specified by the
  collaborator and may bear on the single-trial analyses.

---

## 7. Future directions (feasible without new participants)

- **Continuous and hierarchical modelling.** Treat MEQ as the primary continuous outcome and fit
  the RL/choice models hierarchically (partial pooling) to stabilise per-subject estimates.
- **External validation on public data.** Since new participants cannot be collected, the
  strongest available check is to test the behavioural-dynamics signature on an independent
  public reward/feedback-EEG dataset.
- **Single-trial brain–behaviour coupling.** Test directly whether trial-by-trial feedback P300
  amplitude predicts subsequent risk adjustment, and whether that coupling differs by chronotype —
  a mechanistic link stronger than feature-level fusion.
- **Richer neural features.** Time-frequency (e.g. feedback-related theta) and peak-latency
  measures may carry chronotype information missed by window-mean amplitudes.
- **Self-supervised pretraining.** The 13 unlabeled EEG participants could pretrain the network,
  potentially making learned single-trial features competitive despite small labelled N.

---

## 8. Suggested publication framing

We recommend **two complementary papers**:

- **Paper A — the ERP group-difference paper (existing draft):** the focused cognitive-
  neuroscience finding that Morning and Evening chronotypes differ in feedback P300, unchanged.
- **Paper B — the computational paper (this work):** led by the behavioural-dynamics result and
  the super-additive brain–behaviour fusion (AUC 0.80), with the RL model providing mechanism,
  the continuous-MEQ analysis addressing dimensionality, and the EEG deep-learning negative
  reported as an honest limitation. Paper B reuses Paper A's validated P300 feature, so the two
  remain linked but distinct: A establishes the neural effect; B shows it is behaviourally
  convergent and jointly predictive.

Given the fixed sample size, Paper B is best targeted at a solid Q1 journal in chronobiology or
psychophysiology and framed as an individual-differences / computational cognitive-neuroscience
study — its strengths are methodological rigour, cross-modal convergence, and mechanism, not
scale.

*Mahsima — please adjust the theoretical framing in §2–3 to match the constructs and references
used in the primary manuscript, and add the citations marked [REF].*
