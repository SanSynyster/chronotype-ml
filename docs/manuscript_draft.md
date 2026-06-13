# Chronotype and Feedback Processing: Posterior P300 Differences Between Morning and Evening Types

Authors: Sahab Taali, Mahsima Hajiaboo, Dr Sommayye Heysiattalab.

> **Cover note for co-authors (delete before submission).** This is the first
> complete draft, for your revision and comments. All statistics, effect sizes,
> and figures are final and reproducible from the repository. Two kinds of gaps
> remain, both marked inline in **bold brackets**: `[CITATION]` where a domain
> reference is needed, and `[AUTHOR INPUT]` where only the team has the
> information (recruitment, EEG acquisition, ethics, journal choice). A
> consolidated checklist is at the end. To comment: open `paper.html` in Word
> (File > Open) or import it into Google Docs, then use tracked changes.

---

## Abstract

Chronotype, the stable individual preference for morning versus evening activity,
has been linked to differences in reward sensitivity and decision-making, but its
neural correlates during feedback processing are not well characterized. We
recorded EEG while 39 participants (19 Evening, 20 Morning) performed a risky
decision-making task with trial-by-trial gain/loss feedback, and examined
feedback-locked event-related potentials together with behavioural adaptation.
Morning and Evening chronotypes differed robustly in the posterior P300 response
to feedback valence: the parietal/posterior P300 loss-minus-gain contrast
separated the groups with large effect sizes (Pz Cohen's d = -1.04, FDR
p = 0.034; POz d = -0.92, FDR p = 0.045), and this difference was invariant
across participant-exclusion sensitivity analyses (d approximately -1.0,
p < 0.011 in every scenario). The association was directionally confirmed against
the continuous MEQ score (Pz r = 0.29), and the binary labels were themselves
validated against MEQ. Behaviourally, Evening types showed higher
risky-choice rates (medium-to-large uncorrected effects). Using an interpretable,
leakage-aware machine-learning framework with nested cross-validation, a tuned
logistic-regression model classified chronotype from combined behavioural and ERP
features with balanced accuracy 0.72, ROC AUC 0.75-0.79, and a label-permutation
p = 0.02; its strongest predictor was the same posterior P300 contrast,
demonstrating convergence between the multivariate model and the univariate
neural effect. Classifier performance was modest and, across a family of feature
sets, did not survive multiple-comparison correction, so it is interpreted as a
multivariate complement to the neural finding rather than a deployable
diagnostic. The results
provide preliminary single-cohort evidence that chronotype modulates the neural
evaluation of decision feedback, concentrated in the posterior P300. [AUTHOR
INPUT: confirm task name and any preregistration status.]

**Keywords:** chronotype, P300, feedback-related negativity, risky decision-making, EEG, individual differences.

---

## 1. Introduction

Chronotype describes stable inter-individual variation in the timing of sleep,
alertness, and activity, commonly assessed with the Morningness-Eveningness
Questionnaire (MEQ) and the Munich ChronoType Questionnaire (MCTQ) [CITATION:
Horne & Ostberg 1976; Roenneberg et al.]. Beyond sleep timing, chronotype has
been associated with differences in reward processing, impulsivity, and risky
decision-making, with evening types frequently reported to show greater reward
sensitivity and risk-taking [CITATION].

Feedback during value-based decisions elicits well-characterized event-related
potentials. The feedback-related negativity (FRN), a frontocentral component
peaking around 250-300 ms, is sensitive to outcome valence and reward prediction
error [CITATION: Miltner et al.; Holroyd & Coles]. The P300, a later
parietal/posterior positivity, indexes motivational salience and the allocation
of attention to outcomes [CITATION]. If chronotype reflects differences in how
gains and losses are neurally evaluated, those differences should appear in the
valence sensitivity of these feedback-locked components.

Despite behavioural links between chronotype and risk, the feedback-locked
neural signatures that distinguish morning and evening types during decision-
making remain underexplored, and few studies combine event-related potentials
with rigorous, interpretable machine learning to characterize chronotype. Here we
test, in a single cohort, whether morning and evening chronotypes differ in
feedback-related FRN and P300 responses and in behavioural adaptation to feedback
during a risky-choice task. Our analytic approach is twofold and mutually
validating: (i) an interpretable, leakage-aware machine-learning framework with
nested cross-validation that classifies chronotype from combined behavioural and
ERP features and exposes which features drive the prediction, and (ii) a classical
group comparison of the same theory-driven features with the posterior P300 as the
pre-specified neural hypothesis. The contribution is methodological as much as
empirical: a transparent ML pipeline whose feature importance is interpreted
against, and converges with, the univariate neural statistics. As a secondary
question, we ask whether trial-level risky choice can be predicted from pre-choice
value and choice history under a leakage-safe, participant-generalizing evaluation.

---

## 2. Methods

### 2.1 Participants

Thirty-nine participants were analysed (19 Evening, 20 Morning chronotype).
[AUTHOR INPUT: recruitment, age/sex distribution, inclusion/exclusion criteria,
ethics/IRB approval and consent statement.]

### 2.2 Sample and statistical power

For a two-group comparison with this allocation (19 vs 20), the minimum effect
detectable with 80% power at a two-sided alpha of 0.05 is Cohen's d of
approximately 0.90 (normal approximation). Estimated power is about 0.90 for the
observed posterior P300 effect (d approximately 1.0) but only about 0.35 for a
medium effect (d = 0.5). The study is therefore powered to detect only large
between-group effects; weak or null results for medium-sized contrasts are
inconclusive rather than evidence of absence.

### 2.3 Task and EEG acquisition

Participants performed a risky decision-making task with trial-by-trial feedback
classified by valence (gain/loss) and correctness, yielding Gain-Correct,
Gain-Error, Loss-Correct, and Loss-Error conditions. [AUTHOR INPUT: task design
details, number of trials/blocks, stimulus timing, response mapping, free vs
forced trials.] EEG was recorded and feedback-locked single-trial means were
exported per channel and time window. [AUTHOR INPUT: EEG system, electrode
montage, reference, sampling rate, filtering, artifact rejection, ERP windowing.]

### 2.4 Chronotype labels and data linkage

Primary chronotype labels (Morning vs Evening) were taken from study metadata
(`all final data.xlsx`). Because the metadata workbooks lacked an explicit
participant identifier, participant-summary rows were linked to behavioural
`UserID`s using an optimal one-to-one (Hungarian) assignment over standardized
previous-feedback behavioural aggregates recomputed from the raw trial data
(`scripts/link_raw_metadata.py`). The assignment is a guaranteed bijection; the
smallest assignment margin (0.157) was large relative to typical match distances
(approximately 0.013), and all 39 links were unambiguous.

Two independent metadata sources (`participant_summary.xlsx` and
`all final data.xlsx`) agreed on the chronotype label for every participant, and
the raw behavioural-trials chronotype column disagreed for only two participants
(1027, 1036). The binary labels were validated against the continuous MEQ score
(`scripts/validate_meq_labels.py`): the MEQ score, attached by participant name,
was strongly separated by label (Evening mean 37.3; Morning mean 57.7) in the
standard Horne-Ostberg direction, and all 26 participants with a decisive MEQ
score (outside the 42-58 intermediate band) were consistent with their binary
label. Both raw-behaviour conflict cases had decisive MEQ scores confirming the
primary label (1027 MEQ = 61, Morning; 1036 MEQ = 27, Evening), indicating the
raw-behaviour column was simply in error for these two. Twelve participants fell
in the MEQ intermediate band, where binary assignment is inherently softer.
[AUTHOR INPUT: confirm the binary cutoff/median-split rule.]

### 2.5 Feature engineering

Behaviour was trimmed to match the EEG single-trial exports, and feedback-locked
EEG single-trial means were merged to behaviour by participant and trial index
(`scripts/build_ml_ready_from_raw.py`). Participant-level features were
aggregated to one row per participant (`scripts/build_clean_chronotype.py`) and
included behavioural adaptation measures (post-error slowing, RT slope across the
task, risky-choice late-minus-early change, condition-specific risky-choice
rates) and feedback-locked ERP contrasts (frontocentral FRN error-vs-correct and
loss-vs-gain contrasts; parietal/posterior P300 loss-vs-gain and error-vs-correct
contrasts). Behavioural adaptation features used previous-trial feedback; ERP
condition contrasts used the current trial's feedback label. The theory-driven
compact feature set comprised 12 predictors.

For the secondary risky-choice analysis, trial-level predictors were restricted
to same-trial pre-choice values plus previous-trial and rolling history. Same-
trial outcome, correctness, score, feedback, and feedback-locked EEG were
excluded as predictors to prevent leakage (`scripts/build_clean_risky_choice.py`).

### 2.6 Statistical analysis

**Primary (neural/behavioural group comparison).** For each theory-driven
feature, Evening and Morning groups were compared with Welch's t-test and the
nonparametric Mann-Whitney U test. Effect sizes are reported as Cohen's d with
percentile-bootstrap 95% confidence intervals (10,000 resamples) and as the
small-sample bias-corrected Hedges g. P-values were corrected across the
theory-driven feature family with the Benjamini-Hochberg FDR procedure
(`scripts/group_stats_chronotype.py`).

**Machine-learning classification (Figure 6).** Chronotype was classified at the
participant level from the theory-driven 12-feature set. All preprocessing
(median imputation and standardization of numeric features; most-frequent
imputation and one-hot encoding of any categorical features) was encapsulated in a
scikit-learn pipeline fit only on training folds, eliminating preprocessing
leakage; class imbalance was handled by class weighting with balanced accuracy as
the primary metric. Five classifiers were compared on identical folds: L2- and
L1-regularized logistic regression, random forest, an RBF-kernel support vector
machine, and histogram gradient boosting, with the L2 logistic regression
pre-specified as the primary model. Generalization was estimated with nested
cross-validation: an outer repeated stratified 5-fold loop (10 repeats) estimated
out-of-fold performance while an inner stratified 3-fold grid search tuned each
model's hyperparameters using balanced accuracy, so no test fold informed model
selection. Performance is reported as balanced accuracy, accuracy, ROC AUC,
sensitivity, specificity, and macro F1 (mean and SD across outer folds), with
out-of-fold predicted probabilities pooled for the ROC curve and confusion matrix.
The tuned primary model was tested against a label-permutation null with the full
nested pipeline re-fit on each of 200 permutations. Model interpretability was
assessed from the standardized coefficients of the primary model refit on the full
sample and from cross-validated held-out permutation importance, to test
convergence with the univariate group statistics. As a guard against optimism from
comparing feature sets, the five pre-specified literature feature packs were each
permutation-tested and FDR-corrected as a family, and higher-dimensional models
(47-171 features) were retained only as exploratory comparisons given the
feature-to-sample ratio. Analyses used scikit-learn 1.4 (Python 3.11);
`scripts/ml_chronotype_full.py`.

**Secondary (risky choice).** Trial-level risky choice was evaluated with
participant-grouped cross-validation so that performance reflects generalization
to held-out participants. Results are contextualized against majority-class,
previous-choice-persistence, and participant-mean-oracle baselines
(`scripts/risky_choice_baseline.py`).

**Sensitivity analyses.** All primary and secondary chronotype analyses were
repeated excluding (a) participant 1013 (an EEG/trigger quality-control case),
(b) the two label-conflict participants (1027, 1036), and (c) all three together
(`scripts/sensitivity_matrix.py`).

---

## 3. Results

### 3.1 Primary: posterior P300 distinguishes chronotypes (Figure 1)

Morning and Evening chronotypes differed in the feedback-locked posterior P300
loss-minus-gain contrast. At Pz, Evening types showed a negative loss-minus-gain
contrast and Morning types a positive one (Evening mean -0.96, Morning mean
0.31; Cohen's d = -1.04, 95% CI [-1.63, -0.59]; Welch p = 0.0028; Mann-Whitney
p = 0.005). The POz contrast showed the same pattern (d = -0.92, 95% CI
[-1.55, -0.39]; Welch p = 0.0076; Mann-Whitney p = 0.002). These two contrasts
were the only features that survived Benjamini-Hochberg FDR correction across the
theory-driven feature family (FDR p = 0.034 and 0.045, respectively).

### 3.2 Supporting: behavioural risk-taking

Evening types showed higher risky-choice rates than Morning types, with
medium-to-large uncorrected effects (loss-error risky rate d = 0.81; free risky
rate d = 0.80; gain-correct risky rate d = 0.77; all Welch p < 0.025). These
effects did not survive FDR correction, and their effect-size CIs included small
effects, consistent with the study being underpowered for medium effects.
Frontocentral FRN contrasts did not differ significantly between groups
(e.g. Fz FRN error-minus-correct d = -0.60, FDR p = 0.14).

### 3.3 Continuous MEQ association (Figure 5)

To confirm the effect was not an artifact of dichotomizing chronotype, the
posterior P300 loss-minus-gain contrasts were related to the continuous MEQ score
(n = 38). Both electrodes showed a positive association in the predicted
direction (Pz Pearson r = 0.29, 95% CI [0.06, 0.49], Spearman rho = 0.32; POz
r = 0.24, 95% CI [-0.01, 0.46], Spearman rho = 0.30). The correlations were
modest and only marginally significant, consistent with limited power for
effects of this size (minimum detectable r approximately 0.44 at 80% power for
n = 38) and with the 12 intermediate-band participants adding scatter. The
continuous and group-level analyses thus agree on a graded posterior-P300
association with morningness that the present sample resolves precisely only at
the group-contrast level.

### 3.4 Robustness across participant exclusions (Figure 2)

The posterior P300 group difference was essentially invariant to participant
exclusions. Across the full sample and all three exclusion scenarios, the Pz
loss-minus-gain effect remained near d = -1.0 (range -1.00 to -1.07) with
Welch p < 0.011 and a 95% CI that excluded zero in every case; POz behaved
similarly. In contrast, the classifier (Section 3.5) was more sensitive to the
sample: its permutation p rose from 0.034 in the full sample to 0.38 when the two
label-conflict participants were removed. The dissociation between the invariant
neural effect and the exclusion-sensitive classifier indicates the univariate
group difference is the more robust of the two mutually-validating analyses.

### 3.5 Machine-learning classification of chronotype (Figures 3, 6-8)

Under nested cross-validation with hyperparameter tuning (Figure 6), the five
classifiers were compared on identical folds of the 12-feature set. The
pre-specified L2 logistic regression performed best on balanced accuracy
(0.717 +/- 0.14), with accuracy 0.715, ROC AUC 0.750, sensitivity (Morning) 0.695,
and specificity (Evening) 0.738; the random forest reached the highest AUC
(0.772), L1 logistic regression 0.651 balanced accuracy, the RBF SVM 0.609, and
gradient boosting degenerated to a single-class predictor at this sample size
(Table 1). The tuned primary model selected strong regularization (C = 0.01),
consistent with the small sample.

Pooling out-of-fold predictions for the primary model, the classifier correctly
labelled 28 of 39 participants (accuracy 0.718; sensitivity 0.75, specificity
0.68; confusion matrix Figure 8) with a pooled ROC AUC of 0.79 (Figure 7).
Against a label-permutation null with the full nested pipeline re-fit on each
permutation, the model was significantly above chance (observed balanced accuracy
0.717, null mean 0.509, p = 0.020).

Interpretability links the model directly to the neural finding: the strongest
standardized coefficient was `Pz_P300_loss_minus_gain` (+0.34, the top predictor
of Morning), followed by `loss_error_risky_rate` (-0.31), `gain_correct_risky_rate`
(-0.24), and `POz_P300_loss_minus_gain` (+0.24); held-out permutation importance
showed the same ordering (Figure 3). The multivariate classifier thus relies on
the same posterior P300 contrast identified by the univariate group comparison,
so the two analyses are mutually validating.

Two caveats temper the predictive claim. First, across the family of five
pre-specified feature packs no pack survived FDR correction (best raw permutation
p = 0.051, FDR p = 0.175), so the single-model significance should not be read as
robust to feature-set selection. Second, classifier significance was sensitive to
removing the two label-conflict participants (Section 3.4), whereas the neural
group difference was not. Higher-dimensional models (47-171 features) scored
numerically higher but are reported only as exploratory given the
feature-to-sample ratio. We therefore present the classifier as an interpretable
multivariate complement to the neural effect rather than a deployable diagnostic.

### 3.6 Secondary: trial-level risky choice (Figure 4)

Under participant-grouped cross-validation, the best leakage-safe models reached
a balanced accuracy of approximately 0.587 (ROC AUC approximately 0.62) on
10,669 free-choice trials. This exceeded the majority-class (0.50) and previous-
choice-persistence (0.554) baselines and approached the participant-mean oracle
ceiling (0.604) without ever observing the held-out participant's own data.
Previous-trial and rolling history features carried most of the signal; adding
previous-trial EEG did not materially improve performance over history and value
features in the current representation.

---

## 4. Discussion

In a single cohort, morning and evening chronotypes differed robustly in the
feedback-locked posterior P300, with evening types showing a more negative
loss-minus-gain contrast at parietal/posterior sites. The effect was large
(d approximately 1.0), survived correction for multiple comparisons, and was
stable across participant-exclusion sensitivity analyses. Because the P300 is
associated with the motivational salience of outcomes [CITATION], this pattern is
consistent with chronotype-related differences in how the relative salience of
losses and gains is neurally represented during decision feedback. [AUTHOR INPUT:
connect direction of effect to prior chronotype/reward literature and to the
behavioural risk-taking differences.]

Behaviourally, evening types took more risks, echoing prior reports of greater
reward sensitivity in evening types [CITATION], although these behavioural
effects did not survive correction and the study was underpowered for medium
effects. The convergence of the dominant classifier feature (Pz P300) with the
group comparison strengthens the interpretation that the posterior P300 is the
most informative chronotype-related signal in this dataset.

Methodologically, the study contributes a transparent, leakage-aware,
interpretable machine-learning analysis of chronotype from EEG/ERP and behaviour.
Under nested cross-validation the tuned model classified chronotype with balanced
accuracy 0.72 and AUC 0.75-0.79 (permutation p = 0.02), and -- importantly -- its
most influential feature was the same posterior P300 contrast that drove the
univariate effect. This convergence of multivariate prediction and univariate
statistics is the central evidentiary strength: two methodologically distinct
analyses implicate the same neural signal. At the same time we are deliberately
conservative about the predictive claim: classifier performance was modest, did
not survive correction across feature-set choices, and was sensitive to two
participants (whose labels the MEQ score nonetheless confirms). The model is best
read as an interpretable multivariate complement to the neural finding, not a
deployable diagnostic, and these limits are intrinsic to the sample size rather
than the method. The leakage-safe risky-choice analysis shows that trial-level
choice is weakly but genuinely predictable from choice history in a way that
generalizes across participants.

---

## 5. Limitations

- The sample is a single cohort of 39 participants and is powered only for large
  effects; medium and small effects are inconclusive. Findings require
  independent replication.
- Chronotype is analysed as a binary Morning/Evening label. The labels are
  MEQ-derived and were validated against the continuous MEQ score (all 26
  decisively-scored participants consistent; both raw-behaviour conflict cases
  MEQ-confirmed), but 12 of 39 participants fall in the MEQ intermediate band
  where the dichotomy is inherently soft; modelling the continuous score is left
  to future work.
- ERP features are window-level single-trial means and may miss peak-latency,
  time-frequency, or trial-quality effects.
- Participant 1013 has a known EEG/trigger agreement issue after block 10.
- Although the pre-specified classifier is significant under nested
  cross-validation (permutation p = 0.02), it does not survive FDR correction
  across the family of feature sets and is sensitive to two participants; it is
  therefore interpreted as an interpretable complement to, not independent
  confirmation of, the neural effect. Larger samples are needed for a robust
  predictive model.
- There is no external validation cohort.

---

## 6. Reproducibility and data availability

Analysis code is available in this repository. The environment is pinned in
`requirements.txt` (Python 3.11) with a full freeze in `requirements.lock.txt`.
Raw data are held locally and not committed. [AUTHOR INPUT: decide what to share
- the derived participant-level table (39 rows) is a candidate for public
release; specify a repository/DOI and a contact for raw data requests.] The
full pipeline rebuilds from raw with `python scripts/rebuild_from_raw.py
--execute`; figures are regenerated with `python scripts/make_figures.py`.

---

## Author-input checklist (remove before submission)

- [ ] Participants: recruitment, demographics, ethics/IRB, consent.
- [ ] Task design and full EEG acquisition/preprocessing details.
- [ ] Domain citations for chronotype-reward, FRN, and P300 literature.
- [ ] Confirm the binary chronotype cutoff/median-split rule (12 participants in
      the MEQ 42-58 intermediate band).
- [ ] Direction-of-effect interpretation vs prior literature.
- [ ] Target journal, formatting, CRediT author contributions, funding/conflicts.
- [ ] Preregistration status (the primary vs exploratory split is written to be
      preregistration-friendly).
- [ ] Data-availability statement and any repository DOI.
