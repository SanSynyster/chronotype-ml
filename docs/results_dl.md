# Results — Computational and Deep-Learning Analyses of Chronotype, Feedback Processing, and Risky Choice

*Draft for co-author review (Sahab → Mahsima). Companion to `methodology_dl.md`. All models
use participant-grouped, permutation-clean evaluation on N = 39 (19 Evening, 20 Morning);
continuous-MEQ analyses use the 38 participants with a recorded MEQ score.*

---

## 1. Summary of headline findings

1. **Risky choice is predictable from learned temporal dynamics** — a causal GRU exceeds the
   project's leakage-safe baseline while discarding hand-engineered history features.
2. **Chronotype is decodable from behavioural choice dynamics alone** (ROC AUC = 0.713,
   p = 0.027), independent of EEG.
3. **The feedback P300/FRN independently predicts chronotype** (AUC = 0.668, p = 0.032).
4. **Behaviour and neural feedback signals combine super-additively** — fusing the behavioural
   embedding with the validated ERP contrasts reaches **AUC = 0.797, p = 0.004**, exceeding
   either modality alone.
5. **The continuous MEQ score is predictable** from the same features (fused r = 0.344,
   p = 0.027), so the result does not depend on the binary split.
6. **A reinforcement-learning model gives an interpretable mechanism** — chronotype is
   associated with differences in gain/loss learning and choice consistency.
7. **Honest negative** — a deep network (EEGNet) decodes the feedback task from single trials
   but does *not* recover chronotype from learned single-trial features at this sample size.
8. **The fused result is robust** to participant exclusions, bootstrap resampling, and
   leave-one-subject-out influence.

---

## 2. Risky-choice sequence model (GRU)

Under participant-grouped 5-fold CV, the causal GRU predicted trial-level risky choice with
**balanced accuracy 0.603** and **ROC AUC 0.647** (10,669 free trials, 39 participants),
exceeding the project's prior leakage-safe baseline (balanced accuracy 0.587, AUC 0.62) — and
doing so **without** the hand-engineered rolling-history features, i.e. the network learned the
relevant temporal structure itself.

---

## 3. Chronotype from behavioural dynamics

The 64-dimensional out-of-fold GRU embedding (a chronotype-agnostic summary of each
participant's choice dynamics) predicted Morning vs Evening under nested leave-one-out CV:

| Metric | Value |
|---|---|
| ROC AUC | **0.713** |
| Balanced accuracy | 0.691 |
| Permutation p (1000) | **0.027** |
| Null AUC mean / 95th pct | 0.457 / 0.671 |
| Correlation with continuous MEQ | r = −0.31, p = 0.057 |

Chronotype is therefore decodable from risky-choice dynamics alone, with no EEG, and the
out-of-fold scores track the continuous MEQ in the expected direction.

---

## 4. Chronotype from feedback ERPs

Using only the six validated FRN/P300 contrast features (including the FDR-surviving Pz and
POz P300 loss-minus-gain contrasts), chronotype was predicted at **AUC 0.668, p = 0.032**
(balanced accuracy 0.667). This confirms the neural group difference within the same
permutation-clean predictive framework used for the behavioural and fused models.

---

## 5. Multimodal fusion (behaviour + neural)

| Model | n features | ROC AUC | Balanced acc | Perm p | MEQ r |
|---|---|---|---|---|---|
| Behavioural (GRU embedding) | 64 | 0.713 | 0.691 | 0.027 | −0.31 |
| Neural (validated ERP P300/FRN) | 6 | 0.668 | 0.667 | 0.032 | −0.10 |
| **Fused (behaviour + ERP)** | 70 | **0.797** | **0.742** | **0.004** | **−0.42** |

The fused model **exceeds either modality alone** on every metric: AUC rises from 0.71/0.67 to
0.80, the permutation p strengthens to 0.004, and the correlation with continuous MEQ nearly
doubles to −0.42 (p = 0.009). Because the combination is greater than each part, the
behavioural and neural signals carry **partly independent** chronotype information.

### 5.1 Fusion with *learned* EEG features (control)
Fusing the GRU embedding with the deep EEGNet embedding instead of the validated ERP features
**reduced** performance (AUC 0.65, p = 0.072) relative to behaviour alone — the high-dimensional
learned features added noise rather than signal (see §7). This contrast motivates using the
low-dimensional, validated ERP features for fusion.

---

## 6. Predicting the continuous MEQ score

To avoid dichotomising a continuous trait, the actual MEQ score was predicted by nested
leave-one-out Ridge regression (predicted-vs-observed Pearson r; n = 38):

| Feature set | r (pred vs MEQ) | Perm p |
|---|---|---|
| Behavioural (GRU) | 0.310 | 0.039 |
| Neural (ERP P300/FRN) | 0.145 | 0.075 |
| RL parameters | −0.084 | 0.21 |
| **Fused (behaviour + ERP)** | **0.344** | **0.027** |

The pattern mirrors the binary classification: behaviour predicts MEQ, the fusion is best, and
the result does not rely on the binary Morning/Evening split.

---

## 7. EEG deep learning (EEGNet)

- **Auxiliary task (positive control).** Trained cross-subject to classify single-trial feedback
  valence (loss vs gain), EEGNet reached **AUC 0.641** on held-out participants — the cleaned
  epochs clearly carry decodable single-trial signal.
- **Chronotype (negative result).** Neither learned per-subject EEG embedding predicted
  chronotype: the mean-pooled embedding gave AUC 0.426 (p = 0.61) and the loss-minus-gain
  contrast embedding gave AUC 0.389 (p = 0.71). The contrast embedding did weakly track the
  continuous MEQ (r = 0.30, p = 0.068).

Interpretation: chronotype information is present but faint in the EEG; at N = 39 a network
cannot *learn* it from single trials, even though the hand-measured P300 contrast captures it
(§4). The deep model is therefore reported as an honest negative, and the validated ERP features
— not learned features — are used for fusion.

---

## 8. Reinforcement-learning model: mechanism

Per-participant fits of the asymmetric reward-learning model yielded interpretable parameter
differences by chronotype (Mann–Whitney U; Cohen's d Evening−Morning with bootstrap 95% CI;
Pearson correlation with MEQ):

| Parameter | Morning | Evening | d (95% CI) | group p | MEQ r (p) |
|---|---|---|---|---|---|
| α_gain (learning from gains) | 0.05 | 0.23 | 0.59 [−0.03, 1.22] | **0.040** | −0.32 (0.053) |
| α_loss (learning from losses) | 0.26 | 0.20 | −0.16 [−0.80, 0.49] | 0.28 | −0.09 (0.59) |
| Learning asymmetry (α_loss − α_gain) | 0.20 | −0.03 | −0.53 [−1.22, 0.07] | 0.072 | 0.16 (0.33) |
| β (choice consistency) | 6.62 | 4.42 | −0.57 [−1.39, 0.07] | 0.165 | **0.36 (0.027)** |
| bias (baseline risk) | 1.62 | 0.82 | −0.37 [−1.06, 0.25] | 0.81 | −0.03 (0.86) |

**Mechanistic reading:** Evening types learned more strongly from gains (higher α_gain,
p = 0.040; MEQ r = −0.32) and chose less consistently / more exploratorily (lower β; MEQ
r = 0.36, p = 0.027), whereas Morning types weighted losses more (positive learning asymmetry,
trend p = 0.072) and chose more consistently.

The RL parameters **jointly classify chronotype only weakly** (AUC 0.532, p = 0.158), so they
are interpreted as a mechanistic, convergent account rather than a predictive model — the
predictive headline remains the fusion (§5). *These five parameter comparisons are uncorrected
and are reported as exploratory/mechanistic.*

---

## 9. Robustness of the fused result

| Check | Result |
|---|---|
| Full model (N = 39) | AUC 0.797, permutation p = 0.004 |
| Bootstrap 95% CI on AUC (2000 resamples) | **[0.639, 0.924]** (excludes chance) |
| Exclude flagged participant 1013 (n = 38) | AUC 0.762 |
| Exclude label conflicts 1027/1036 (n = 37) | AUC 0.795 |
| Exclude all flagged (n = 36) | AUC 0.775 |
| Exclude MEQ-intermediate band (n = 27) | AUC 0.698 |
| Leave-one-subject-out AUC range | 0.653 – 0.853 |
| Most influential participant | 1001 (removal lowers AUC to 0.653, still above chance) |

The effect survives every pre-defined exclusion, the bootstrap CI lies entirely above chance,
and no single participant drives it (worst case AUC 0.653 with participant 1001 removed). The
influence of participant 1001 is reported transparently.

---

## 10. One-paragraph synthesis

Across an internally-validated, leakage-safe, permutation-clean analysis of 39 participants,
chronotype was predicted from the temporal dynamics of risky choice (AUC 0.713) and,
independently, from the feedback-locked P300/FRN (AUC 0.668); the two signals combined
super-additively to AUC 0.797 (p = 0.004; bootstrap 95% CI [0.64, 0.92]) and predicted the
continuous MEQ score (r = 0.34, p = 0.027). A reinforcement-learning model localised the
behavioural effect to chronotype differences in gain-driven learning and choice consistency.
End-to-end deep learning on single-trial EEG did not recover chronotype at this sample size,
indicating the neural contribution is best captured by the validated ERP contrasts. The fused
result was robust to participant exclusions and to the removal of any single participant.

*Mahsima — please check the framing against the primary ERP results and flag any numbers you
would like reported differently (e.g., adding exact effect-size CIs from the ERP analysis, or
aligning terminology with the main manuscript).*
