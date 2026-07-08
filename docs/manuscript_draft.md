# Chronotype in the Neural and Behavioural Evaluation of Decision Feedback: Converging ERP, Computational, and Fusion Evidence

Authors: Sahab Taali, Mahsima Hajiaboo, Dr Sommayye Heysiattalab.

Target journal: **Psychophysiology**.

> **Cover note for co-authors (delete before submission).** This is the
> **integrated** draft merging the ERP group-difference work and the
> computational/deep-learning strand into one paper (superseding the earlier
> two-paper plan). All analyses are complete and integrated (Intro, Methods
> §2.1–2.11, Results §3.1–3.12, Discussion §4.1–4.7, References); all statistics
> match their source artifacts (`reports/clean/statcheck/report.md`, 30/30). Nine
> DOI-verified citations are inserted; the only remaining inline markers are
> `[AUTHOR INPUT]` — information only the team holds (ethics/IRB, recruitment,
> EEG acquisition + preprocessing, task details, MEQ cutoff, preregistration,
> CRediT/funding, data availability), all itemized in `docs/coauthor_request.md`,
> plus one request for team-preferred chronotype-reward empirical citations. To
> comment: open `docs/paper.docx` in Word (tracked changes) — it is regenerated
> from this file with `python scripts/build_paper.py`. Supplementary material is
> in `docs/supplementary.md`; consolidated main figures are
> `docs/figures/fig_main_*`.

---

## Abstract

Chronotype, the stable individual preference for morning versus evening activity,
has been linked to differences in reward sensitivity and decision-making, but its
neural correlates during feedback processing are not well characterized. We
recorded EEG while 39 participants (19 Evening, 20 Morning) performed a risky
decision-making task with trial-by-trial gain/loss feedback, and characterized
chronotype through converging event-related potential (ERP), computational, and
multimodal-fusion analyses under a uniformly leakage-safe, permutation-clean
evaluation. First, Morning and Evening chronotypes differed robustly in the
feedback-locked **posterior P300**: the parietal/posterior loss-minus-gain
contrast separated the groups with large effect sizes (Pz Cohen's d = -1.04, FDR
p = 0.034; POz d = -0.92, FDR p = 0.045), invariant across participant-exclusion
sensitivity analyses (d ≈ -1.0, p < 0.011 throughout) and directionally confirmed
against the continuous MEQ score. Second, a causal recurrent (GRU) model of
risky-choice **dynamics** decoded chronotype from behaviour alone (out-of-fold
ROC AUC 0.713, permutation p = 0.027), independent of EEG. Third, and centrally,
**fusing** the behavioural dynamics with the validated ERP contrasts was
**super-additive** — chronotype decoding rose to AUC 0.797 (p = 0.004; bootstrap
95% CI [0.64, 0.92]), exceeding either modality alone and predicting the
continuous MEQ score (r = 0.34, p = 0.027) — indicating partly independent
behavioural and neural information. An exploratory asymmetric reinforcement-learning
analysis suggested chronotype differences in outcome-specific learning and choice
consistency, but these were weakly identified and did not survive a hierarchical
partial-pooling refit, so the mechanism is reported as tentative. Two honest
negatives bound the claim: an end-to-end network (EEGNet) decoded single-trial
feedback valence cross-subject (AUC 0.64) but did **not** recover chronotype from
learned features at this sample size, and the chronotype effect was **not**
expressed as a within-subject single-trial P300→next-choice coupling (d = -0.36,
p = 0.22), localising it to the between-subject/trait level. The results provide
preliminary single-cohort evidence that chronotype shapes the neural evaluation
of decision feedback — concentrated in the posterior P300 — and that this neural
signature converges super-additively with the behavioural dynamics of risky
choice. [AUTHOR INPUT: confirm task name and any preregistration status.]

**Keywords:** chronotype, P300, feedback processing, reinforcement learning,
risky decision-making, EEG, multimodal fusion, individual differences.

---

## 1. Introduction

Chronotype describes stable inter-individual variation in the diurnal timing of
sleep, alertness, and performance, commonly indexed by the Morningness-Eveningness
Questionnaire (MEQ) and the Munich ChronoType Questionnaire (MCTQ; Horne &
Ostberg, 1976; Roenneberg et al., 2003). It is more than a sleep-timing
preference: chronotype is embedded in circadian physiology and has been linked to
broad differences in affect, cognition, and self-regulation (Adan et al., 2012). Of
particular relevance to decision-making, evening-type individuals are frequently
reported to show greater sensation seeking and reward sensitivity (Muro et al.,
2012), higher impulsivity (Adan et al., 2010; McGowan & Coogan, 2018), stronger
approach-related (BIS/BAS) tendencies (Randler et al., 2014), and more risk-taking
(Killgore, 2007) than morning types, a profile that plausibly reflects differences
in how the outcomes of choices — gains and losses — are evaluated. Neuroimaging has
begun to link evening-ness to altered reward-related brain function (Hasler et al.,
2013; Hasler & Clark, 2013). Yet most evidence for this link is behavioural or
trait-questionnaire based, and the *neural* processing of decision feedback in
morning versus evening types — its feedback-locked electrophysiology — remains
poorly characterized. [AUTHOR INPUT: the team may add or substitute preferred
chronotype-reward empirical citations here.]

Feedback during value-based decisions elicits well-characterized event-related
potentials (ERPs) that dissociate distinct evaluative processes. The
feedback-related negativity (FRN), a frontocentral deflection peaking ~250-300 ms
after outcome onset, is sensitive to outcome valence and has been interpreted as a
reward-prediction-error signal arising from midbrain-to-medial-frontal projections
(Gehring & Willoughby, 2002; Hajcak et al., 2007; Holroyd & Coles, 2002; Miltner
et al., 1997; Sambrook & Goslin, 2015), and its reward-sensitive positive
counterpart is the reward positivity (Proudfit, 2015). The feedback P300, a later parietal/posterior
positivity, scales with the motivational salience and subjective significance of
outcomes and the allocation of attention to them (Polich, 2007; San Martín, 2012;
Wu & Zhou, 2009; Yeung & Sanfey, 2004). Because these components separate the rapid
valence-coding and the later salience-weighting of feedback (Walsh & Anderson,
2012), they provide a principled window onto *how* individuals neurally evaluate
gains and losses. If chronotype reflects differences in outcome
evaluation, those differences should surface in the valence sensitivity of these
feedback-locked components — most plausibly in the salience-weighting P300.

Despite the behavioural links between chronotype and risk, the feedback-locked
neural signatures that distinguish morning and evening types during decision-making
are largely unknown, and existing individual-differences work rarely combines ERPs
with rigorous, leakage-aware modelling of the *behaviour* those ERPs accompany.
Characterizing a stable trait such as chronotype from neural and behavioural data
is methodologically demanding in the modest samples typical of ERP studies:
naive multivariate models readily overfit and leak information across the
train/test boundary, inflating apparent effects (Kriegeskorte et al., 2009;
Varoquaux, 2018). A convincing account therefore
requires (i) theory-driven neural hypotheses tested with classical statistics,
(ii) predictive models evaluated under strict, participant-generalizing,
permutation-clean protocols, and (iii) an explicit test of whether behavioural and
neural signals carry *convergent* or *redundant* information about the trait.

Here we characterize chronotype during a feedback-based risky-choice task in a
single cohort, integrating ERP, computational, and multimodal-fusion analyses under
a uniformly leakage-safe, permutation-clean evaluation. We test three hypotheses.
**H1 (neural):** morning and evening chronotypes differ in the feedback-locked
posterior P300 response to outcome valence (loss versus gain), our pre-specified
neural hypothesis, tested by classical group comparison with multiple-comparison
control and corroborated against the continuous MEQ score. **H2 (behavioural):**
chronotype is decodable from the *dynamics* of risky choice — how individuals adapt
choices to feedback over time — captured by a causal recurrent model that learns
temporal structure without hand-engineered history features. **H3 (convergence):**
combining the behavioural dynamics with the validated ERP contrasts predicts
chronotype *better than either modality alone*, which would indicate that the neural
and behavioural expressions of chronotype carry partly independent information. To
sharpen interpretation we further ask (a) whether an interpretable reinforcement-
learning model can localise the behavioural difference to specific learning or
choice-consistency parameters, (b) whether end-to-end deep learning recovers
chronotype from single-trial EEG, and (c) whether the neural effect operates as a
within-subject, single-trial coupling between feedback P300 and subsequent choice
or as a between-subject trait. As a secondary question, we assess how far
trial-level risky choice itself is predictable from pre-choice value and choice
history under participant-generalizing evaluation.

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
exported per channel and time window. Acquisition, preprocessing, and ERP
measurement are reported in line with the Society for Psychophysiological Research
committee guidelines (Keil et al., 2014). [AUTHOR INPUT: EEG system, electrode
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
standard Horne-Ostberg direction (Horne & Ostberg, 1976; see also Roenneberg et
al., 2007, and Wittmann et al., 2006, on chronotype/MCTQ assessment), and all 26
participants with a decisive MEQ
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
(Benjamini & Hochberg, 1995; `scripts/group_stats_chronotype.py`). Where the
evidence for a null was of interest, we complemented significance tests with
JZS Bayes factors (Rouder et al., 2009) and two one-sided equivalence tests
(TOST; Lakens, 2017).

**Machine-learning classification (Figure 6).** Chronotype was classified at the
participant level from the theory-driven 12-feature set. All preprocessing
(median imputation and standardization of numeric features; most-frequent
imputation and one-hot encoding of any categorical features) was encapsulated in a
scikit-learn pipeline fit only on training folds, eliminating preprocessing
leakage (Kriegeskorte et al., 2009; Varoquaux, 2018); class imbalance was handled
by class weighting with balanced accuracy as the primary metric. Five classifiers were compared on identical folds: L2- and
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
nested pipeline re-fit on each of 200 permutations (Combrisson & Jerbi, 2015;
Poldrack et al., 2020). Model interpretability was
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

### 2.7 Recurrent sequence model of choice dynamics (GRU)

To capture how participants adapt risk-taking over time, a causal (unidirectional)
single-layer gated recurrent unit (GRU; Cho et al., 2014; hidden size 64) was trained to predict
trial-level risky choice from the current pre-choice context plus the previous
trial's outcome (20 features), with no hand-engineered rolling-history features so
the network had to learn temporal structure itself. The recurrence is causal, so
the prediction at trial *t* depends only on trials ≤ *t*. Training used
binary cross-entropy over valid time steps (Adam, lr = 3×10⁻³, 40 epochs, seed 0)
and was evaluated with participant-grouped 5-fold CV, standardizing features on
training participants only. For each held-out participant, the GRU hidden state
was averaged across trials to yield a 64-dimensional, out-of-fold, chronotype-
agnostic **behavioural embedding** used downstream (`scripts/dl/risky_choice_seq.py`).

### 2.8 Asymmetric reinforcement-learning model

To obtain interpretable mechanistic parameters, a reward-learning model was fit to
each participant's free-trial choices. Because the sign of each box was hidden and
random, we modelled outcome-driven updating of risk preference rather than stimulus
value: two options (safe = magnitude 5, risky = magnitude 25) with action values Q,
choice rule P(risky) = logistic(β·(Q_risky − Q_safe) + bias), and chosen-option
update Q ← Q + α·(r − Q) using separate learning rates for gains and losses,
motivated by evidence for valence-asymmetric reinforcement learning (Frank et al.,
2007; Gershman, 2015; Lefebvre et al., 2017); r is the signed chosen value scaled
to [−1, 1] (Sutton & Barto, 2018). Parameters (α_gain, α_loss, β, bias, and derived
asymmetry α_loss − α_gain) were fit by maximum likelihood (L-BFGS-B, 8 restarts;
Daw, 2011; `scripts/dl/rl_model.py`) and, as a robustness check, refit in a Bayesian
**hierarchical partial-pooling** model with a chronotype group-level offset on each
parameter, sampled with NUTS (Ahn et al., 2017; `scripts/dl/rl_hierarchical.py`).

### 2.9 EEG deep learning (EEGNet)

EEGNet (Lawhern et al., 2018), a compact convolutional network widely used for
end-to-end EEG decoding (Roy et al., 2019; Schirrmeister et al., 2017), was
applied to the cleaned single-trial epochs (64 × 251). It was trained cross-subject
(GroupKFold over participants) to decode single-trial feedback **valence**
(loss vs gain) — a chronotype-agnostic positive control — with per-channel z-scoring
on training statistics (Adam, lr = 10⁻³, weight decay 10⁻³, 25 epochs). Penultimate-
layer features were aggregated per participant (mean, and a loss-minus-gain contrast)
to form learned EEG embeddings (`scripts/dl/eegnet.py`, `eeg_chronotype.py`).

### 2.10 Chronotype-decoding evaluator, fusion, and continuous-MEQ prediction

All embeddings (behavioural, EEG, RL, fused) were evaluated for chronotype
prediction with one shared, permutation-clean procedure: a StandardScaler → PCA →
L2-logistic pipeline under nested CV (inner 4-fold grid search over PCA components
and logistic C; outer leave-one-participant-out), yielding one out-of-fold score per
participant. Inference used a 1000-iteration label-permutation test with the *entire*
nested procedure re-run per permutation; out-of-fold scores were also correlated with
the continuous MEQ. Multimodal **fusion** concatenated the GRU behavioural embedding
with the six validated ERP contrast features (behaviour-only and ERP-only evaluated
identically). To avoid dichotomizing the trait, the continuous MEQ score was
additionally predicted by nested leave-one-out Ridge regression (predicted-vs-observed
Pearson r, 1000-permutation p). Robustness of the fused result was assessed by
participant-level bootstrap CI, pre-defined exclusion scenarios, and leave-one-subject-
out influence (`scripts/dl/multimodal_chronotype.py`, `continuous_meq.py`,
`robustness.py`).

### 2.11 Single-trial P300 → next-choice coupling

To test whether the neural effect operates within-subject at the single-trial level,
each trial *t* was paired to the following trial *t*+1 (leakage-safe: the feedback-
locked P300 at *t* follows the *t* choice and precedes the *t*+1 choice). A
per-participant logistic slope of next-trial risky choice on the within-participant
z-scored trial-*t* P300 (overall, and a valence-resolved P300×loss interaction) was
compared between chronotypes (two-stage Mann-Whitney + bootstrap d), with a
confirmatory Binomial mixed model carrying a P300×chronotype interaction and random
P300 slopes (`scripts/dl/p300_risk_coupling.py`).

All deep-learning/computational analyses ran in a separate environment (PyTorch,
MNE-Python, scikit-learn, PyMC), with fixed seed 0; see `docs/methodology_dl.md` for
full detail.

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
(e.g. Fz FRN error-minus-correct d = -0.60, FDR p = 0.14). Bayes factors were
consistent with the absence of an FRN group difference at central sites (FCz
BF01 = 2.46, Cz BF01 = 3.17 — moderate evidence for the null) while remaining
inconclusive frontally (Fz BF01 = 0.82). Equivalence tests (TOST, bounds ±0.5 SD)
on the loss-minus-gain FRN contrasts could not formally establish equivalence
(e.g. Fz TOST p = 0.43), consistent with the study being underpowered to bound
medium effects rather than with a demonstrated null.

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

The P300 effect was also robust to the choice of analysis window. Across a
72-cell specification curve (Pz/POz × window centre 400–600 ms × width 50/100 ms
× mean/peak summary), recomputed directly from the single-trial epochs, the
loss-minus-gain group difference kept the same sign (Evening more negative) in
64/72 cells and was large and significant (d < −0.8, p < 0.05) in 19/72, with the
largest and most significant cells clustering around the pre-specified 450–550 ms
P300 window (anchor cell Pz d = −0.84, POz d = −0.66; slightly attenuated relative
to the validated-feature estimates because this curve is recomputed from raw epoch
amplitudes via an independent pipeline). The primary effect is therefore not an
artefact of window selection.

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

### 3.7 Chronotype is decodable from risky-choice dynamics alone (Figure 9)

A causal (unidirectional) GRU trained to predict trial-level risky choice from
pre-choice context and the previous trial's outcome exceeded the leakage-safe
baseline (balanced accuracy 0.603, AUC 0.647) **without** hand-engineered history
features, learning the temporal structure itself. Its out-of-fold 64-dimensional
behavioural embedding — a chronotype-agnostic summary of each participant's choice
dynamics — predicted Morning vs Evening under nested leave-one-participant-out CV
at **ROC AUC 0.713** (balanced accuracy 0.691; label-permutation p = 0.027; null
AUC mean 0.457), and the out-of-fold scores tracked the continuous MEQ (r = -0.31).
Chronotype is therefore expressed in the dynamics of risky choice, independent of
any EEG.

### 3.8 Behaviour and neural feedback signals fuse super-additively (Figure 9; headline)

Using only the six validated FRN/P300 contrast features, chronotype was predicted
at AUC 0.668 (p = 0.032), confirming the neural group difference within the same
permutation-clean predictive framework. **Fusing** the behavioural GRU embedding
with these validated ERP contrasts was **super-additive**: chronotype decoding
rose to **AUC 0.797** (balanced accuracy 0.742, permutation p = 0.004), exceeding
either modality alone, and the correlation with the continuous MEQ nearly doubled
to r = -0.42 (p = 0.009).

| Model | n features | ROC AUC | Balanced acc | Perm p | MEQ r |
|---|---|---|---|---|---|
| Behavioural (GRU embedding) | 64 | 0.713 | 0.691 | 0.027 | -0.31 |
| Neural (validated ERP P300/FRN) | 6 | 0.668 | 0.667 | 0.032 | -0.10 |
| **Fused (behaviour + ERP)** | 70 | **0.797** | **0.742** | **0.004** | **-0.42** |

Because the combination exceeds each part, the behavioural and neural signals
carry **partly independent** chronotype information. As a control, fusing the GRU
embedding with the *learned* EEGNet embedding instead reduced performance
(AUC 0.65), motivating the use of low-dimensional validated ERP features (§3.11).
The fused result was robust across all pre-defined participant exclusions
(AUC 0.70-0.80), had a bootstrap 95% CI of [0.639, 0.924] that excluded chance,
and was not driven by any single participant (leave-one-subject-out AUC range
0.653-0.853; most influential participant 1001 reported transparently).

### 3.9 The continuous MEQ score is predictable (Figure 9)

To avoid dichotomizing a continuous trait, the actual MEQ score was predicted by
nested leave-one-out Ridge regression (n = 38). Behaviour predicted MEQ
(r = 0.310, p = 0.039), the neural features less so (r = 0.145), and the fused
feature set best (**r = 0.344, p = 0.027**), mirroring the binary classification
and confirming the effect does not depend on the Morning/Evening split.

### 3.10 A reinforcement-learning model localises the mechanism (Figure 9)

Per-participant maximum-likelihood fits of an asymmetric reward-learning model
gave exploratory point estimates suggesting Evening types learned more strongly
from gains (α_gain Evening 0.23 vs Morning 0.05; group p = 0.040; MEQ r = -0.32)
and chose less consistently (β; MEQ r = 0.36, p = 0.027), with Morning types
showing a relative bias toward learning from losses (learning-asymmetry trend
p = 0.072). The RL parameters jointly classified chronotype only weakly
(AUC 0.532).

However, these point-estimate contrasts **did not survive a Bayesian hierarchical
partial-pooling refit** (NUTS, chronotype group-level offset per parameter; max
R-hat 1.02, no divergences). Under partial pooling the α_gain contrast collapsed to
essentially zero (Evening − Morning ≈ 0.00, 94% HDI [-0.010, 0.011],
P(contrast>0) = 0.46) and the β contrast reversed sign and remained uncertain
(+0.38, HDI [-0.53, 1.28]); the only tendency was weaker learning from losses in
Evening types (α_loss contrast -0.012, HDI [-0.032, 0.002]; learning asymmetry
P(contrast>0) = 0.06). Crucially, the subject-level MLE and hierarchical estimates
were weakly correlated (α_loss r = 0.02, β r = 0.13, α_gain r = 0.45), showing the
per-participant MLE parameters were poorly identified given ~270 free trials with
hidden, random signs. We therefore treat the RL analysis as **weakly identified and
exploratory**: it does not provide a confirmed mechanism, and — importantly — the
predictive headline (the behaviour+ERP fusion, §3.8) does not depend on it. The
five MLE comparisons are uncorrected; the hierarchical contrasts are reported with
HDIs rather than as tests.

### 3.11 Honest negative: deep learning on single-trial EEG (Figure 9)

Trained cross-subject, EEGNet decoded single-trial feedback **valence** (loss vs
gain) at AUC 0.641 on held-out participants — the cleaned epochs carry genuine
decodable single-trial signal. However, neither learned per-subject EEG embedding
predicted chronotype (mean-pooled AUC 0.426; loss-minus-gain contrast embedding
AUC 0.389). At N = 39 a network cannot *learn* the subtle subject-level chronotype
difference from single trials, even though a small set of theory-driven, FDR-
validated ERP contrasts captures it (§3.8). We report this transparently: for
small-sample individual-differences EEG, validated low-dimensional features can
outperform end-to-end representation learning. Both learned embeddings decoded
chronotype at chance under permutation (mean AUC 0.426, p = 0.61; contrast AUC
0.389, p = 0.71); a permutation-null density-ratio Bayes factor was only weakly
informative (BF01 ≈ 1.2), because the observed AUCs fell slightly below chance,
so the non-significant permutation test is the clearer statement of the null.
(The positive-control valence AUC of 0.64 is from the canonical EEGNet run
[30 training epochs, 5-fold participant-grouped CV] that generated the embeddings
used here; a separate 2-epoch/2-fold sanity run gave 0.59 and is not used.)

### 3.12 The chronotype effect is trait-level, not a single-trial coupling (Figure 10)

To test whether the neural effect operates as a within-subject dynamic — i.e.
whether the feedback P300 on trial *t* drives the participant's risk adjustment on
trial *t*+1 — we fit a leakage-safe coupling analysis (the P300 follows the *t*
choice and precedes the *t*+1 choice). Across 10,630 consecutive trial pairs, the
per-participant coupling slope did **not** differ by chronotype, either overall
(Evening -0.033 vs Morning 0.025; d = -0.36, 95% CI [-1.12, 0.25]; Mann-Whitney
p = 0.22) or in the theory-matched valence-resolved form (P300×loss interaction
d = -0.08, p = 0.62); a confirmatory Binomial mixed model gave a P300×chronotype
interaction of p = 0.12. Thus the chronotype difference is expressed at the
**between-subject/trait level** — in aggregate feedback evaluation and in choice
dynamics that fuse super-additively (§3.8) — rather than as a strong single-trial
brain→behaviour coupling. This bounds the mechanistic interpretation and indicates
the fusion is not reducible to a moment-to-moment neural drive of choice.

---

## 4. Discussion

### 4.1 Overview

In a single cohort of 39 participants, chronotype was expressed in the neural and
behavioural evaluation of decision feedback, and the two modalities converged. Four
findings anchor the account: (i) morning and evening chronotypes differed robustly
in the feedback-locked posterior P300 to outcome valence (H1); (ii) chronotype was
decodable from the dynamics of risky choice alone (H2); (iii) fusing the behavioural
dynamics with the validated ERP contrasts predicted chronotype better than either
modality alone (H3), and predicted the continuous MEQ score; and (iv) two
pre-planned negative tests — end-to-end EEG decoding and a within-subject
single-trial coupling — localised the effect to a between-subject trait rather than
a moment-to-moment neural drive of choice. Because every predictive result was
obtained under participant-generalizing, permutation-clean evaluation, the
convergence across methodologically distinct analyses, rather than any single
model's accuracy, is the study's central evidentiary strength.

### 4.2 The posterior P300 and the neural evaluation of feedback

The pre-specified neural hypothesis was confirmed: evening types showed a more
negative loss-minus-gain P300 contrast at parietal/posterior sites, a large effect
(d ≈ 1.0) that survived FDR correction, was stable across participant-exclusion
sensitivity analyses and across a 72-cell window specification curve, and was
directionally corroborated against the continuous MEQ. Because the posterior P300
scales with the motivational salience and subjective significance of outcomes
(Polich, 2007; Yeung & Sanfey, 2004), this pattern is consistent with
chronotype-related differences in how the relative salience of losses and gains is
neurally weighted during feedback. The direction is coherent with the wider
literature: evening types took more risks here and are reported to be more
reward-sensitive and impulsive (Adan et al., 2010; Killgore, 2007; Muro et al.,
2012), and evening-ness has been associated with altered reward-related brain
function (Hasler et al., 2013; Hasler & Clark, 2013) and with a circadian system
that up-regulates reward motivation at certain phases (Murray et al., 2009). A
relatively smaller posterior-P300 response to losses than gains in evening types is
thus the electrophysiological complement of a behavioural profile that
comparatively down-weights losses (cf. loss aversion; Kahneman & Tversky, 1979).
[AUTHOR INPUT: the team may sharpen this direction-of-effect link to the specific
prior findings it wishes to foreground.] That the frontocentral FRN did not
distinguish the groups — with Bayes factors giving moderate evidence for the null
at central sites — further localises the chronotype difference to the later,
salience-weighting stage of feedback processing rather than the earlier
valence/prediction-error stage indexed by the FRN (Holroyd & Coles, 2002;
Sambrook & Goslin, 2015).

### 4.3 Brain–behaviour convergence: the super-additive fusion

The most informative result is that behaviour and neural feedback signals combined
**super-additively**. Chronotype was decodable from risky-choice dynamics alone
(GRU AUC 0.713) and from the validated ERP contrasts alone (AUC 0.668), but fusing
them raised decoding to AUC 0.797 and roughly doubled the correlation with the
continuous MEQ (to −0.42). Had the two modalities indexed the same latent variable,
fusion would have been redundant; instead, how choices evolve over many trials and
the immediate neural evaluation of each outcome appear to capture partly independent,
complementary facets of the same chronotype difference. This cross-modal
convergence is precisely the evidence that strengthens an individual-differences
claim beyond a single-modality classifier, and it is consistent with the
interpretable group-comparison result: the multivariate classifier's most
influential feature was the same posterior-P300 contrast that drove the univariate
effect. We are nonetheless deliberately conservative about deployment. The
single-model classifier was modest, did not survive correction across the family of
feature sets, and was sensitive to two participants (whose labels the MEQ
nonetheless confirms), and the fused estimate carries real uncertainty (bootstrap CI
[0.64, 0.92]). The models are best read as evidence of a convergent trait
signature, not as a deployable diagnostic — a limit intrinsic to the sample size
rather than the method.

### 4.4 A tentative computational mechanism

To move from prediction toward mechanism, an asymmetric reinforcement-learning model
(cf. Frank et al., 2007; Lefebvre et al., 2017) offered a candidate account —
greater gain-driven learning and lower choice consistency in evening types — that
would fit a profile of stronger approach/reward sensitivity and weaker
loss-avoidance, and a risk-sensitive learning process (Niv et al., 2012), plausibly
under circadian/dopaminergic modulation of reward function (McClung, 2007; Murray
et al., 2009; Webb et al., 2009). However, we treat this account as **exploratory
and weakly identified**: the maximum-likelihood parameter contrasts did not survive
a hierarchical partial-pooling refit, and the per-participant estimates were poorly
identified in this hidden-sign paradigm (weak MLE-to-hierarchical agreement). The
mechanism is therefore a hypothesis for a trial-richer or hierarchically-designed
follow-up rather than a confirmed finding, and — importantly — the predictive and
neural results stand independently of it [AUTHOR INPUT: align this tentative
interpretation with the constructs/references in the team's prior chronotype work].

### 4.5 Two informative negatives and a methodological message

Two pre-planned negative tests sharpen rather than weaken the account. First,
end-to-end deep learning on single-trial EEG (EEGNet) decoded the feedback task
across unseen participants but did not recover chronotype from learned single-trial
features, whereas a small set of theory-driven, FDR-validated ERP contrasts did.
At the sample sizes typical of ERP research, validated low-dimensional features can
thus outperform representation learning for subtle individual-difference signals,
and naive feature learning can dilute a real effect — a practically useful message
for small-sample individual-differences EEG, where deep models are data-hungry
(Roy et al., 2019) and small samples inflate cross-validated error (Varoquaux,
2018). Second, the chronotype effect was
**not** expressed as a within-subject, single-trial coupling between feedback P300
and the subsequent choice; it is a between-subject trait. The super-additive fusion
is therefore a convergence of two trait-level signatures — a stable difference in
feedback salience-weighting and a stable difference in choice dynamics — not a
reducible moment-to-moment neural control of behaviour. Reporting these negatives
transparently both bounds the mechanistic claim and pre-empts the concern that the
fusion merely re-describes a single between-subject correlation.

### 4.6 Chronotype as a dimensional trait

Although we analysed a binary morning/evening label, the same features predicted the
**continuous MEQ** score (fused r = 0.34), and 12 of 39 participants fell in the MEQ
intermediate band where the dichotomy is inherently soft. The dimensional result
indicates the effect is not an artefact of dichotomization and supports treating
morningness–eveningness as a graded individual difference in the reward-based
evaluation of decision feedback. Future work should model the continuous score as
the primary outcome.

### 4.7 Conclusion

Using converging ERP, computational, and fusion analyses under uniformly
leakage-safe evaluation, we provide preliminary single-cohort evidence that
chronotype shapes the neural evaluation of decision feedback — concentrated in the
posterior P300 — and that this neural signature converges super-additively with the
behavioural dynamics of risky choice. The findings are internally validated and
await independent replication, but the cross-modal, multi-method convergence offers
a rigorous template for characterizing stable traits from modest neural and
behavioural samples.

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
- All predictive results (GRU decoding, fusion, continuous-MEQ) are **internally
  validated only** on one cohort of 39; robustness was established within-sample
  (bootstrap, exclusions, leave-one-subject-out) but not against a new dataset.
  Removing the most influential participant lowered the fused AUC from 0.80 to
  0.65, so the estimate carries real uncertainty (CI [0.64, 0.92]).
- The reinforcement-learning parameters were weakly identified (limited to ~270
  free trials/participant with hidden random signs): the MLE contrasts did not
  survive a hierarchical partial-pooling refit and MLE-vs-hierarchical subject
  estimates were weakly correlated. The RL account is therefore exploratory and
  tentative, not a confirmed mechanism; the predictive and neural findings do not
  depend on it.
- Findings concern one feedback-based risky-choice paradigm with hidden signs;
  generalization to other reward/decision tasks is untested.
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

## References

*APA-7 list, DOI-verified (source `references.bib` / `docs/references_apa7.md`;
regenerated into `docs/paper.docx` by `scripts/build_paper.py`). One remaining
`[AUTHOR INPUT]` marker in the text invites team-preferred chronotype-reward
citations; Byrne & Murray (2017) was dropped as unverifiable.*

Adan, A., Archer, S. N., Hidalgo, M. P., Di Milia, L., Natale, V., & Randler, C. (2012). Circadian typology: A comprehensive review. *Chronobiology International, 29*(9), 1153–1175. https://doi.org/10.3109/07420528.2012.719971

Adan, A., Natale, V., Caci, H., & Prat, G. (2010). Relationship between circadian typology and functional and dysfunctional impulsivity. *Chronobiology International, 27*(3), 606–619. https://doi.org/10.3109/07420521003663827

Ahn, W.-Y., Haines, N., & Zhang, L. (2017). Revealing neurocomputational mechanisms of reinforcement learning and decision-making with the hBayesDM package. *Computational Psychiatry, 1*, 24–57. https://doi.org/10.1162/CPSY_a_00002

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological), 57*(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1724–1734). https://doi.org/10.3115/v1/D14-1179

Combrisson, E., & Jerbi, K. (2015). Exceeding chance level by chance: The caveat of theoretical chance levels in brain signal classification and statistical assessment of decoding accuracy. *Journal of Neuroscience Methods, 250*, 126–136. https://doi.org/10.1016/j.jneumeth.2015.01.010

Daw, N. D. (2011). Trial-by-trial data analysis using computational models. In *Decision making, affect, and learning: Attention and Performance XXIII* (pp. 3–38). Oxford University Press. https://doi.org/10.1093/acprof:oso/9780199600434.003.0001

Frank, M. J., Moustafa, A. A., Haughey, H. M., Curran, T., & Hutchison, K. E. (2007). Genetic triple dissociation reveals multiple roles for dopamine in reinforcement learning. *Proceedings of the National Academy of Sciences, 104*(41), 16311–16316. https://doi.org/10.1073/pnas.0706111104

Gehring, W. J., & Willoughby, A. R. (2002). The medial frontal cortex and the rapid processing of monetary gains and losses. *Science, 295*(5563), 2279–2282. https://doi.org/10.1126/science.1066893

Gershman, S. J. (2015). Do learning rates adapt to the distribution of rewards? *Psychonomic Bulletin & Review, 22*(5), 1320–1327. https://doi.org/10.3758/s13423-014-0790-3

Hajcak, G., Moser, J. S., Holroyd, C. B., & Simons, R. F. (2007). It's worse than you thought: The feedback negativity and violations of reward prediction in gambling tasks. *Psychophysiology, 44*(6), 905–912. https://doi.org/10.1111/j.1469-8986.2007.00567.x

Hasler, B. P., & Clark, D. B. (2013). Circadian misalignment, reward-related brain function, and adolescent alcohol involvement. *Alcoholism: Clinical and Experimental Research, 37*(4), 558–565. https://doi.org/10.1111/acer.12003

Hasler, B. P., Sitnick, S. L., Shaw, D. S., & Forbes, E. E. (2013). An altered neural response to reward may contribute to alcohol problems among late adolescents with an evening chronotype. *Psychiatry Research: Neuroimaging, 214*(3), 357–364. https://doi.org/10.1016/j.pscychresns.2013.08.005

Holroyd, C. B., & Coles, M. G. H. (2002). The neural basis of human error processing: Reinforcement learning, dopamine, and the error-related negativity. *Psychological Review, 109*(4), 679–709. https://doi.org/10.1037/0033-295X.109.4.679

Horne, J. A., & Ostberg, O. (1976). A self-assessment questionnaire to determine morningness-eveningness in human circadian rhythms. *International Journal of Chronobiology, 4*, 97–110.

Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica, 47*(2), 263–291. https://doi.org/10.2307/1914185

Keil, A., Debener, S., Gratton, G., Junghöfer, M., Kappenman, E. S., Luck, S. J., Luu, P., Miller, G. A., & Yee, C. M. (2014). Committee report: Publication guidelines and recommendations for studies using electroencephalography and magnetoencephalography. *Psychophysiology, 51*(1), 1–21. https://doi.org/10.1111/psyp.12147

Killgore, W. D. S. (2007). Effects of sleep deprivation and morningness-eveningness traits on risk-taking. *Psychological Reports, 100*(2), 613–626. https://doi.org/10.2466/pr0.100.2.613-626

Kriegeskorte, N., Simmons, W. K., Bellgowan, P. S. F., & Baker, C. I. (2009). Circular analysis in systems neuroscience: The dangers of double dipping. *Nature Neuroscience, 12*(5), 535–540. https://doi.org/10.1038/nn.2303

Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science, 8*(4), 355–362. https://doi.org/10.1177/1948550617697177

Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering, 15*(5), 056013. https://doi.org/10.1088/1741-2552/aace8c

Lefebvre, G., Lebreton, M., Meyniel, F., Bourgeois-Gironde, S., & Palminteri, S. (2017). Behavioural and neural characterization of optimistic reinforcement learning. *Nature Human Behaviour, 1*(4), Article 0067. https://doi.org/10.1038/s41562-017-0067

McClung, C. A. (2007). Circadian genes, rhythms and the biology of mood disorders. *Pharmacology & Therapeutics, 114*(2), 222–232. https://doi.org/10.1016/j.pharmthera.2007.02.003

McGowan, N. M., & Coogan, A. N. (2018). Sleep and circadian rhythm function and trait impulsivity: An actigraphy study. *Psychiatry Research, 268*, 251–256. https://doi.org/10.1016/j.psychres.2018.07.030

Miltner, W. H. R., Braun, C. H., & Coles, M. G. H. (1997). Event-related brain potentials following incorrect feedback in a time-estimation task: Evidence for a generic neural system for error detection. *Journal of Cognitive Neuroscience, 9*(6), 788–798. https://doi.org/10.1162/jocn.1997.9.6.788

Muro, A., Gomà-i-Freixanet, M., & Adan, A. (2012). Circadian typology and sensation seeking in adolescents. *Chronobiology International, 29*(10), 1376–1382. https://doi.org/10.3109/07420528.2012.728665

Murray, G., Nicholas, C. L., Kleiman, J., Dwyer, R., Carrington, M. J., Allen, N. B., & Trinder, J. (2009). Nature's clocks and human mood: The circadian system modulates reward motivation. *Emotion, 9*(5), 705–716. https://doi.org/10.1037/a0017080

Niv, Y., Edlund, J. A., Dayan, P., & O'Doherty, J. P. (2012). Neural prediction errors reveal a risk-sensitive reinforcement-learning process in the human brain. *Journal of Neuroscience, 32*(2), 551–562. https://doi.org/10.1523/JNEUROSCI.5498-10.2012

Poldrack, R. A., Huckins, G., & Varoquaux, G. (2020). Establishment of best practices for evidence for prediction. *JAMA Psychiatry, 77*(5), 534–540. https://doi.org/10.1001/jamapsychiatry.2019.3671

Polich, J. (2007). Updating P300: An integrative theory of P3a and P3b. *Clinical Neurophysiology, 118*(10), 2128–2148. https://doi.org/10.1016/j.clinph.2007.04.019

Proudfit, G. H. (2015). The reward positivity: From basic research on reward to a biomarker for depression. *Psychophysiology, 52*(4), 449–459. https://doi.org/10.1111/psyp.12370

Randler, C., Baumann, V. P., & Horzum, M. B. (2014). Morningness-eveningness, Big Five and the BIS/BAS inventory. *Personality and Individual Differences, 66*, 64–67. https://doi.org/10.1016/j.paid.2014.03.010

Roenneberg, T., Kuehnle, T., Juda, M., Kantermann, T., Allebrandt, K., Gordijn, M., & Merrow, M. (2007). Epidemiology of the human circadian clock. *Sleep Medicine Reviews, 11*(6), 429–438. https://doi.org/10.1016/j.smrv.2007.07.005

Roenneberg, T., Wirz-Justice, A., & Merrow, M. (2003). Life between clocks: Daily temporal patterns of human chronotypes. *Journal of Biological Rhythms, 18*(1), 80–90. https://doi.org/10.1177/0748730402239679

Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D., & Iverson, G. (2009). Bayesian t tests for accepting and rejecting the null hypothesis. *Psychonomic Bulletin & Review, 16*(2), 225–237. https://doi.org/10.3758/PBR.16.2.225

Roy, Y., Banville, H., Albuquerque, I., Gramfort, A., Falk, T. H., & Faubert, J. (2019). Deep learning-based electroencephalography analysis: A systematic review. *Journal of Neural Engineering, 16*(5), 051001. https://doi.org/10.1088/1741-2552/ab260c

Sambrook, T. D., & Goslin, J. (2015). A neural reward prediction error revealed by a meta-analysis of ERPs using great grand averages. *Psychological Bulletin, 141*(1), 213–235. https://doi.org/10.1037/bul0000006

San Martín, R. (2012). Event-related potential studies of outcome processing and feedback-guided learning. *Frontiers in Human Neuroscience, 6*, Article 304. https://doi.org/10.3389/fnhum.2012.00304

Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping, 38*(11), 5391–5420. https://doi.org/10.1002/hbm.23730

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage, 180*, 68–77. https://doi.org/10.1016/j.neuroimage.2017.06.061

Walsh, M. M., & Anderson, J. R. (2012). Learning from experience: Event-related potential correlates of reward processing, neural adaptation, and behavioral choice. *Neuroscience & Biobehavioral Reviews, 36*(8), 1870–1884. https://doi.org/10.1016/j.neubiorev.2012.05.008

Webb, I. C., Baltazar, R. M., Wang, X., Pitchers, K. K., Coolen, L. M., & Lehman, M. N. (2009). Diurnal variations in natural and drug reward, mesolimbic tyrosine hydroxylase, and clock gene expression in the male rat. *Journal of Biological Rhythms, 24*(6), 465–476. https://doi.org/10.1177/0748730409346657

Wittmann, M., Dinich, J., Merrow, M., & Roenneberg, T. (2006). Social jetlag: Misalignment of biological and social time. *Chronobiology International, 23*(1–2), 497–509. https://doi.org/10.1080/07420520500545979

Wu, Y., & Zhou, X. (2009). The P300 and reward valence, magnitude, and expectancy in outcome evaluation. *Brain Research, 1286*, 114–122. https://doi.org/10.1016/j.brainres.2009.06.032

Yeung, N., & Sanfey, A. G. (2004). Independent coding of reward magnitude and valence in the human brain. *Journal of Neuroscience, 24*(28), 6258–6264. https://doi.org/10.1523/JNEUROSCI.4537-03.2004

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
- [ ] Fold `methodology_dl.md` §5-8 into Methods 2.7-2.11 (GRU, RL, EEGNet,
      fusion/evaluator, single-trial coupling) — structural map at the top.
- [x] Parallel-analysis results integrated: Bayes factors (G-B, §3.2/§3.11),
      equivalence tests (G-F, §3.2), P300 specification curve (G-E, §3.4),
      hierarchical RL (G-D, §3.10 — MLE mechanism did **not** survive pooling; RL
      now framed as exploratory). No `[PENDING GPT]` markers remain.
- [ ] Confirm mechanistic interpretation (RL / reward-sensitivity) against the
      team's prior chronotype constructs and references.
