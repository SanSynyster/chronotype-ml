# Methodology — Computational and Deep-Learning Analyses of Chronotype, Feedback Processing, and Risky Choice

*Draft for co-author review (Sahab → Mahsima). This document covers the methods for the
computational/machine-learning strand only; the primary ERP group-difference analyses are
described separately.*

---

## 1. Overview and analytic philosophy

The goal of these analyses was to test whether **chronotype** (Morning vs Evening, and the
continuous Morningness–Eveningness Questionnaire [MEQ] score) can be predicted from, and
mechanistically explained by, (i) the **dynamics of risky choice**, (ii) **feedback-locked
EEG**, and (iii) their combination. Throughout, three principles were enforced because the
sample is small (N = 39):

1. **No information leakage.** Every predictive model is evaluated with **participant-grouped
   cross-validation**: all trials from a given participant are confined to either the training
   or the test partition, never split across both. Within-trial outcome/feedback information is
   never used to predict the same trial's choice.
2. **Permutation-clean inference with model selection inside the loop.** All hyper-parameters
   are selected inside a nested cross-validation, and the *entire* nested procedure is re-run
   under every label permutation, so reported p-values already account for model selection and
   require no further multiple-comparison correction for the hyper-parameter grid.
3. **Reproducibility.** All random seeds are fixed (seed = 0). Analyses run in an isolated
   Python environment; see §11.

---

## 2. Participants and chronotype labels

- **N = 39** participants with both behavioural and EEG data and a chronotype label
  (19 Evening, 20 Morning). A further 13 participants with EEG but no chronotype label were
  used only for the auxiliary EEG analyses, not for chronotype decoding.
- **Binary chronotype labels** were taken from the linked participant metadata and validated
  against the **continuous MEQ** score: MEQ separated the groups in the expected direction
  (Evening mean ≈ 37.3, Morning ≈ 57.7); all decisively-scored participants matched their
  binary label, and the two raw-behaviour label conflicts (participants 1027, 1036) were
  MEQ-confirmed.
- **12 of 39** participants fall in the MEQ intermediate band (42–58), where the binary split
  is inherently soft; this is addressed by (a) a continuous-MEQ analysis (§8) and (b) an
  exclusion in the robustness battery (§9).

---

## 3. Behavioural task

A feedback-based risky-choice task (16 blocks × 24 trials). On each trial two boxes were
shown, each displaying an absolute magnitude drawn from {5, 25}; the **sign was hidden during
choice** and revealed only as feedback. Participants chose the left or right box. After the
choice, the signed value of the chosen box was revealed with coloured feedback, and the score
updated by that signed value.

- **Trial types.** *Free* trials presented two different magnitudes (5 vs 25; N = 10,669) so
  the participant could choose between a low-magnitude "safe" option and a high-magnitude
  "risky" option. *Forced* trials presented equal magnitudes (N = 3,683) and carry no
  risk choice; they were excluded from choice modelling.
- **Risky choice** was defined as selecting the high-magnitude (25) box on a free trial.
- **Feedback conditions** (and EEG trigger codes): gain-correct (50), gain-error (60),
  loss-correct (70), loss-error (80), where gain/loss is the valence of the chosen box and
  correct/error is whether the higher signed value was chosen.

### 3.1 Behavioural features (leakage-safe)
Trial-level predictors used pre-choice context and *previous*-trial information only, never the
current trial's outcome: current-trial option values and their derived contrasts
(OptionDiff, ValueSum, value ratios, sign configuration, trial progress) and previous-trial
quantities (previous risky choice, previous feedback valence/correctness, previous
reaction-time and score change). Hand-engineered rolling-history summaries were deliberately
**withheld** from the sequence model so that the network had to learn temporal structure itself
(§5.1).

---

## 4. EEG acquisition and preprocessing

- **Recording.** 64-channel EEG (ANT Neuro), sampling rate 250 Hz, standard 10–20 layout.
- **Preprocessing.** Cleaning, artefact handling, filtering, epoching and baseline correction
  were performed by the collaborator in EEGLAB/ERPLAB; the present analyses used the resulting
  **cleaned, epoched single-trial datasets** (EEGLAB `.set`). No further preprocessing was
  applied beyond cropping and per-channel standardisation described below.
- **Epochs.** Feedback-locked epochs were cropped to **−0.2 to +0.8 s** (251 samples),
  yielding a tensor of shape (trials × 64 channels × 251 samples) per participant;
  **13,522 epochs** across the 39 labelled participants. Condition labels were recovered from
  the ERPLAB bin descriptors (feedback codes 50/60/70/80) and collapsed to valence
  (gain = 50/60, loss = 70/80) for the auxiliary decoding task.
- **Validated ERP features.** For the fusion analyses we used the project's pre-validated,
  window-mean ERP contrast features (FRN window 360–430 ms, P300 window 450–550 ms): six
  feedback contrasts including the parietal/posterior **Pz and POz P300 loss-minus-gain**
  contrasts that survived FDR correction in the primary ERP analysis.

---

## 5. Computational models of behaviour

### 5.1 Recurrent sequence model (GRU)
To capture how each participant adapts risk-taking over time, we trained a **causal
(unidirectional) gated recurrent unit (GRU)** to predict trial-level risky choice. Per
participant, trials were ordered chronologically; at each trial the network received the
current pre-choice context plus the previous trial's outcome (20 features; §3.1). The recurrence
is causal, so the prediction at trial *t* depends only on trials ≤ *t* (no future leakage).

- Architecture: single-layer GRU, hidden size 64 → linear read-out → per-trial logit.
- Training: binary cross-entropy over valid (non-padded) time steps; Adam (lr = 3 × 10⁻³);
  40 epochs; seed 0.
- Evaluation: **GroupKFold (5 folds) over participants**; features standardised using
  training-participant statistics only. Out-of-fold predictions are pooled to compute balanced
  accuracy and ROC AUC, directly comparable to the project's prior leakage-safe baseline.
- **Participant embedding.** For each held-out participant we averaged the GRU hidden state
  across that participant's valid trials, producing a 64-dimensional **behavioural embedding**.
  Because the embedding is out-of-fold and the network never sees chronotype, it is a
  leakage-safe, chronotype-agnostic summary of each participant's choice dynamics, used as input
  to the chronotype decoder (§7) and the fusion models (§8).

### 5.2 Asymmetric reinforcement-learning (RL) model
To obtain *interpretable* mechanistic parameters, we fit a reward-learning model to each
participant's free-trial choices. Because signs are hidden and random, there is no stable
stimulus value to learn; we therefore modelled outcome-driven updating of risk preference
rather than stimulus value. Treating "safe" (magnitude 5) and "risky" (magnitude 25) as two
options with action values Q:

- Choice rule: P(risky) = logistic( β·(Q_risky − Q_safe) + bias ).
- Update (chosen option only): Q ← Q + α·(r − Q), with **separate learning rates for positive
  and negative outcomes** (α = α_gain if r > 0 else α_loss); r is the signed value of the chosen
  box scaled to [−1, 1] (÷25).
- Free parameters per participant: **α_gain, α_loss, β (inverse temperature), bias (baseline
  risk propensity)**; a derived **learning-rate asymmetry** (α_loss − α_gain) was also computed.
- Fitting: maximum likelihood by L-BFGS-B with bounds (α ∈ [0,1], β ∈ [0,10], bias ∈ [−5,5]),
  8 random restarts per participant to mitigate local minima.

---

## 6. EEG deep learning (EEGNet)

We used **EEGNet** (Lawhern et al., 2018), a compact convolutional network, on the cleaned
single-trial epochs (input 64 × 251). The channel axis is treated as the spatial dimension of a
depthwise convolution; defaults F1 = 8, D = 2, F2 = 16, temporal kernel = 125 (0.5 s at 250 Hz),
dropout 0.5.

- **Auxiliary decoding task.** The network was trained **cross-subject** (GroupKFold, 5 folds
  over participants) to classify single-trial feedback **valence (loss vs gain)** — a
  chronotype-agnostic task. Per-channel z-scoring used training statistics; optimisation by Adam
  (lr = 10⁻³, weight decay 10⁻³), 25 epochs, batch size 128, with per-mini-batch normalisation
  for memory efficiency. This establishes that the cleaned epochs carry decodable single-trial
  signal.
- **EEG participant embeddings.** From the trained (out-of-fold) networks we extracted the
  penultimate-layer features per trial and aggregated them per participant in two ways:
  (i) **mean** over all trials, and (ii) a **contrast** embedding = mean(loss trials) −
  mean(gain trials), the latter chosen to match the validated P300 loss-minus-gain effect.
  Both were passed to the chronotype decoder (§7).

---

## 7. Chronotype decoding framework (shared evaluator)

All embeddings (behavioural, EEG, RL, fused) were evaluated for chronotype prediction with a
single, shared, permutation-clean procedure:

- **Estimator.** Pipeline of StandardScaler → PCA → L2-regularised logistic regression.
- **Nested cross-validation.** Inner: 4-fold stratified CV grid search over PCA components
  {5, 10, 20} (capped to remain valid given inner-fold sample size) and logistic C
  {0.1, 0.5, 1.0}, scored by ROC AUC. Outer: **leave-one-participant-out**, producing one
  out-of-fold decision score per participant.
- **Primary metric.** ROC AUC (and balanced accuracy) of the out-of-fold scores.
- **Inference.** A **label-permutation test** (1000 permutations): chronotype labels are
  shuffled and the *entire* nested procedure re-run; p = (1 + #{null ≥ observed}) / (1 + 1000).
- **Convergent validation.** The out-of-fold decision scores were correlated (Pearson) with the
  continuous MEQ score as an independent check.

---

## 8. Multimodal fusion and continuous-MEQ prediction

- **Fusion (behavioural + neural).** The behavioural GRU embedding was concatenated with the six
  validated ERP contrast features and evaluated with the §7 framework; the behavioural-only and
  ERP-only feature sets were evaluated identically for comparison. A parallel fusion of the GRU
  embedding with the *learned* EEG embedding (§6) was also evaluated.
- **Continuous-MEQ regression.** To avoid dichotomising a continuous trait, we predicted the
  actual MEQ score with **nested leave-one-out Ridge regression** (inner 5-fold grid search over
  ridge α ∈ {0.1, 1, 10, 100}), scored by the Pearson correlation between predicted and observed
  MEQ, with a 1000-permutation p-value. The behavioural, ERP, RL-parameter and fused feature
  sets were each evaluated.

---

## 9. Statistical analysis of RL parameters and robustness

- **Group differences.** For each RL parameter, Morning vs Evening were compared with a
  Mann–Whitney U test and Cohen's d with a 5000-sample bootstrap 95% CI; each parameter was also
  correlated with the continuous MEQ (Pearson and Spearman).
- **Robustness battery (fused model).**
  - **Bootstrap CI:** 2000 participant-level resamples of the out-of-fold scores give a 95% CI
    on the AUC.
  - **Exclusion sensitivity:** the analysis was repeated after removing (i) the flagged
    participant 1013, (ii) the label-conflict participants 1027/1036, (iii) all flagged, and
    (iv) the 12 MEQ-intermediate participants.
  - **Leave-one-subject-out influence:** each participant was removed in turn and the nested
    AUC recomputed, reporting the AUC range and the most influential participant.

---

## 10. Software and reproducibility

Analyses were implemented in Python in an isolated virtual environment, kept separate from the
primary ERP analysis environment. Key libraries: PyTorch 2.12 (GRU, EEGNet), MNE-Python 1.12
(reading EEGLAB `.set`), scikit-learn 1.9 (cross-validation, logistic/ridge models), SciPy
(RL fitting, statistics), NumPy/pandas, joblib (parallel permutations). The raw ANT-Neuro `.cnt`
recordings — which are not Neuroscan format and cannot be read by the standard MNE CNT reader —
were handled via the `antio` library, although the analyses above use the collaborator's cleaned
`.set` epochs. All scripts, fixed seeds, and exact hyper-parameters are version-controlled.

---

## 11. Notes / limitations carried into the methods

- N = 39 from a single cohort; all predictive results are internally validated only and are
  framed as such.
- The deep EEG model is reported with an **honest negative**: it learns the feedback task but
  does not, at this sample size, recover chronotype from learned single-trial features.
- RL parameter group comparisons are **uncorrected across the five parameters** and are reported
  as mechanistic/convergent rather than confirmatory.

*Please flag anything that should be reworded to match how the data were actually collected
(e.g., exact EEG amplifier/montage, filter settings, ICA/artefact-rejection details, and the
final epoch count after rejection), since those preprocessing specifics live on your side.*
