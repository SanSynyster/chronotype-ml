# Chronotype Differences in Feedback-Related P300 and Risky-Choice Dynamics

Authors: Sahab Taali, Mahsima Hajiaboo, Dr Sommayye Heysiattalab.

Target journal: **Psychophysiology**.

---

## Abstract

Chronotype has been associated with reward sensitivity and risky decision-making,
but little is known about how morning and evening types process decision feedback
at the neural level. We recorded EEG from 39 participants (19 Evening, 20 Morning)
during a risky decision-making task with gain and loss feedback. Chronotype was
examined using feedback-locked event-related potentials, behavioural sequence
models, and multimodal analyses with participant-generalizing validation. The
clearest neural difference was in the posterior P300. The parietal loss-minus-gain
contrast separated Morning and Evening groups with large effects (Pz Cohen's d =
-1.04, FDR p = 0.034; POz d = -0.92, FDR p = 0.045), remained stable across
participant-exclusion analyses (d approximately -1.0, p < 0.011 throughout), and
showed the same direction of association with continuous MEQ score. Behaviour carried
convergent information: out-of-fold scores from a causal GRU model of risky-choice
dynamics distinguished Morning and Evening participants without EEG (ROC AUC 0.713,
permutation p = 0.027). Out-of-fold scores that combined the behavioural embedding
with the validated ERP contrasts separated the groups more strongly (AUC 0.797, p =
0.004; bootstrap 95% CI [0.64, 0.92]) and tracked the continuous MEQ score (r =
0.34, p = 0.027), indicating that behavioural dynamics and feedback ERPs carry
partly independent, rather than diagnostic, information about chronotype. That the
same measures related to continuous MEQ score is consistent with a graded,
dimensional trait rather than a strict dichotomy. An asymmetric
reinforcement-learning analysis suggested possible differences in outcome-specific
learning and choice consistency, but these estimates were weak and did not survive
hierarchical partial pooling. Two boundary analyses constrained the
interpretation. EEGNet decoded feedback valence across participants (AUC 0.64) but
did not decode chronotype from learned EEG features, and single-trial P300 amplitude
did not predict the next risky choice differently by chronotype (d = -0.36, p =
0.22). These findings provide preliminary evidence that chronotype is reflected in
the neural evaluation of feedback, most clearly in the posterior P300, and that this
neural signature complements behavioural choice dynamics. [AUTHOR INPUT: confirm
task name and preregistration status.]

**Keywords:** chronotype, P300, feedback processing, reinforcement learning,
risky decision-making, EEG, multimodal fusion, individual differences.

---

## 1. Introduction

Chronotype captures stable individual differences in preferred timing of sleep,
activity, and alertness. It is commonly measured with the Morningness-Eveningness
Questionnaire and related instruments such as the Munich ChronoType Questionnaire
(Horne & Ostberg, 1976; Roenneberg et al., 2003). Chronotype is not simply a
preference for going to bed early or late. It is linked to circadian physiology and
to differences in affect, cognition, and self-regulation (Adan et al., 2012).
Decision-making is one domain in which these differences may matter. Evening types
have been reported to show greater sensation seeking and reward sensitivity (Muro
et al., 2012), higher impulsivity (Adan et al., 2010; McGowan & Coogan, 2018),
stronger approach-related tendencies (Randler et al., 2014), and higher risk-taking
(Killgore, 2007). Neuroimaging work has also linked eveningness to altered
reward-related brain responses (Hasler et al., 2013; Hasler & Clark, 2013). What is
less clear is whether these behavioural and trait-level differences are visible in
the electrophysiology of feedback processing itself. [AUTHOR INPUT: the team may
add or substitute preferred chronotype-reward empirical citations here.]

Feedback during value-based decisions evokes several event-related potentials that
index different stages of outcome evaluation. The feedback-related negativity
(FRN), a frontocentral component peaking about 250-300 ms after feedback, is
sensitive to outcome valence and is often interpreted in relation to prediction
error signalling (Gehring & Willoughby, 2002; Hajcak et al., 2007; Holroyd & Coles,
2002; Miltner et al., 1997; Sambrook & Goslin, 2015). Closely related work also
describes the reward positivity as the positive-going counterpart of this response
(Proudfit, 2015). The feedback P300 is later and more posterior. It varies with the
salience, significance, and attentional processing of outcomes (Polich, 2007; San
Martín, 2012; Wu & Zhou, 2009; Yeung & Sanfey, 2004). Together, these components
separate early feedback-valence processing from later salience-weighting (Walsh &
Anderson, 2012). If chronotype is related to how gains and losses are evaluated,
the strongest evidence should appear in feedback-locked responses to outcome
valence, especially in the posterior P300.

Testing this question is difficult in the sample sizes typical of ERP research.
Classical group comparisons can identify interpretable neural effects, but
multivariate models can overfit or leak information across train and test sets if
they are not evaluated carefully (Kriegeskorte et al., 2009; Varoquaux, 2018). A
useful account therefore needs two complementary parts. First, it should test a
clear ERP hypothesis with appropriate correction and robustness checks. Second, it
should ask whether behaviour carries related information about chronotype under
participant-generalizing validation. If the neural and behavioural measures add to
one another, they would suggest convergent but non-redundant expressions of the same
trait.

The present study examined these questions in a single cohort performing a
feedback-based risky-choice task. We tested whether Morning and Evening chronotypes
differed in the posterior P300 response to loss versus gain feedback. We then asked
whether chronotype could be decoded from the dynamics of risky choice using a causal
recurrent model, and whether combining behavioural dynamics with validated ERP
features improved prediction beyond either source alone. Additional analyses tested
whether an interpretable reinforcement-learning model could explain the behavioural
difference, whether EEGNet could recover chronotype from single-trial EEG, and
whether trial-to-trial P300 variation predicted subsequent risky choice differently
by chronotype.

---

## 2. Methods

### 2.1 Participants

Thirty-nine participants were analysed (19 Evening, 20 Morning chronotype).
[AUTHOR INPUT: recruitment, age/sex distribution, inclusion/exclusion criteria,
ethics/IRB approval and consent statement.]

### 2.2 Sample and statistical power

With 19 Evening and 20 Morning participants, a two-group comparison has 80% power
at alpha = 0.05 only for large effects, approximately Cohen's d = 0.90 by normal
approximation. Power is therefore adequate for the observed posterior P300 effect
(d approximately 1.0), but low for medium effects (about 0.35 for d = 0.5). Null or
weak results for medium-sized contrasts should be interpreted as inconclusive.

### 2.3 Task and EEG acquisition

Participants completed a risky decision-making task with trial-by-trial feedback.
On each trial two boxes were shown, each displaying an absolute magnitude; the sign
(gain or loss) of each box was hidden at the moment of choice and revealed only as
feedback. On free trials the two boxes differed in magnitude, so choosing the
higher-magnitude box constituted the risky choice (a larger potential gain but also
a larger potential loss), whereas choosing the lower-magnitude box was the safe
choice. Feedback was then coded by the valence of the chosen box (gain if its
revealed sign was positive, loss if negative) and by correctness (whether the box
with the higher revealed signed value had been chosen), producing four feedback
conditions: Gain-Correct, Gain-Error, Loss-Correct, and Loss-Error. Valence
(gain vs loss) therefore indexes the affective outcome of the choice, and
correctness indexes whether that choice was, in hindsight, the better of the two
options. [AUTHOR INPUT: number of trials/blocks, magnitudes used, stimulus and
feedback timing, response mapping, and the free-versus-forced trial split.] EEG was recorded during the task, and feedback-locked
single-trial amplitudes were exported by channel and time window. EEG acquisition,
preprocessing, and ERP measurement follow the reporting recommendations of Keil et
al. (2014). [AUTHOR INPUT: EEG system, electrode montage, reference, sampling rate,
filtering, artifact rejection, ERP windowing.]

### 2.4 Chronotype labels and data linkage

Primary chronotype labels were Morning and Evening categories from study metadata.
The metadata workbooks did not contain the same explicit participant identifier as
the behavioural files, so metadata rows were linked to behavioural UserIDs by an
optimal one-to-one assignment based on standardized previous-feedback behavioural
aggregates. The assignment was unambiguous: the smallest assignment margin was
0.157, much larger than typical match distances (approximately 0.013).

Two independent metadata sources agreed on the chronotype label for every
participant. The chronotype column in the raw behavioural trials disagreed for two
participants (1027 and 1036), but both cases were resolved by the continuous MEQ
score. MEQ scores were strongly separated by the binary labels in the expected
Horne-Ostberg direction (Evening mean 37.3; Morning mean 57.7; Horne & Ostberg,
1976; Roenneberg et al., 2007; Wittmann et al., 2006). All 26 participants outside
the intermediate MEQ band (42-58) were consistent with their binary label. The two
raw-behaviour conflict cases had decisive MEQ scores supporting the primary label
(1027 MEQ = 61, Morning; 1036 MEQ = 27, Evening). Twelve participants fell in the
intermediate range, where a binary classification is less sharp.
[AUTHOR INPUT: confirm the binary cutoff/median-split rule.]

### 2.5 Feature engineering

Behavioural trials were aligned to the EEG single-trial exports by participant and
trial index. Participant-level analyses used one row per participant. The compact
feature set contained 12 theory-driven predictors: behavioural adaptation measures
(post-error slowing, RT slope, late-minus-early change in risky choice, and
condition-specific risky-choice rates) and feedback-locked ERP contrasts
(frontocentral FRN error-vs-correct and loss-vs-gain contrasts, plus
parietal/posterior P300 loss-vs-gain and error-vs-correct contrasts). Behavioural
adaptation features used information from previous feedback. ERP contrasts used the
current trial's feedback condition.

For the secondary trial-level risky-choice analysis, predictors were limited to
pre-choice values, previous-trial information, and rolling choice history. Same-
trial outcome, correctness, score, feedback, and feedback-locked EEG were excluded
from these models.

### 2.6 Statistical analysis

For each theory-driven feature, Morning and Evening groups were compared with
Welch's t-test and the Mann-Whitney U test. Effect sizes are reported as Cohen's d
with percentile-bootstrap 95% confidence intervals (10,000 resamples), and also as
Hedges g. P-values were corrected across the theory-driven feature family using the
Benjamini-Hochberg false discovery rate procedure (Benjamini & Hochberg, 1995). For
planned null or near-null findings, we also report JZS Bayes factors (Rouder et al.,
2009) and two one-sided equivalence tests (Lakens, 2017).

Participant-level chronotype classification used the 12-feature compact set. All
preprocessing was performed inside the cross-validation folds, including imputation,
standardization, and categorical encoding, to avoid preprocessing leakage
(Kriegeskorte et al., 2009; Varoquaux, 2018). Five classifiers were compared on the
same folds: L2- and L1-regularized logistic regression, random forest, RBF-kernel
support vector machine, and histogram gradient boosting. The primary model was
pre-specified as L2 logistic regression. Generalization was estimated with repeated
nested cross-validation. The outer loop was a repeated stratified 5-fold split with
10 repeats; the inner loop was a stratified 3-fold grid search. Balanced accuracy
was the primary metric, and class imbalance was handled by class weighting. The
primary model was also tested against a label-permutation null in which the full
nested pipeline was re-fit for each of 200 permutations (Combrisson & Jerbi, 2015;
Poldrack et al., 2020). Model interpretation used standardized coefficients from a
full-sample refit and held-out permutation importance.

To avoid selecting a favourable feature set after the fact, five pre-specified
literature feature packs were permutation-tested as a family with FDR correction.
Higher-dimensional models with 47-171 features were retained as exploratory because
the feature-to-sample ratio was high. Trial-level risky choice was evaluated with
participant-grouped cross-validation and compared with majority-class,
previous-choice-persistence, and participant-mean-oracle baselines. Sensitivity
analyses repeated the primary and secondary chronotype analyses after excluding
participant 1013, the two label-conflict participants (1027 and 1036), and all three
flagged participants.

### 2.7 Recurrent sequence model of choice dynamics (GRU)

Choice dynamics were modelled with a causal single-layer gated recurrent unit (GRU;
Cho et al., 2014). The model predicted trial-level risky choice from the current
pre-choice context and the previous trial's outcome. Rolling-history variables were
not supplied, so temporal structure had to be learned from the sequence. The model
used 20 input features, a hidden size of 64, binary cross-entropy loss over valid
time steps, Adam optimization (lr = 3×10⁻³), 40 epochs, and seed 0. Evaluation used
5-fold participant-grouped cross-validation with feature standardization estimated
from training participants only. For each held-out participant, the trial-wise
hidden states were averaged to produce a 64-dimensional behavioural embedding for
chronotype prediction.

### 2.8 Asymmetric reinforcement-learning model

To obtain interpretable parameters, we fit an asymmetric reward-learning model to
each participant's free choices. Because the sign of each box was hidden and random,
the model represented outcome-driven changes in risk preference rather than
learning about stable stimulus values. The model included safe and risky action
values, a logistic choice rule, separate learning rates for gains and losses, an
inverse-temperature parameter, and a choice bias. The outcome was the signed chosen
value scaled to [-1, 1]. This specification follows standard reinforcement-learning
logic (Sutton & Barto, 2018) and allows valence-asymmetric learning (Frank et al.,
2007; Gershman, 2015; Lefebvre et al., 2017). Participant-level parameters were fit
by maximum likelihood with L-BFGS-B and eight restarts (Daw, 2011). As a robustness
check, the same model was refit with Bayesian hierarchical partial pooling and a
chronotype group offset on each parameter (Ahn et al., 2017).

### 2.9 EEG deep learning (EEGNet)

EEGNet (Lawhern et al., 2018) was used as a compact convolutional model for
single-trial EEG decoding (Roy et al., 2019; Schirrmeister et al., 2017). The model
was applied to cleaned feedback-locked epochs with 64 channels and 251 time points.
It was trained across participants to decode feedback valence (loss vs gain), which
served as a chronotype-independent positive control. Cross-validation was grouped by
participant. Per-channel z-scoring used training-fold statistics only. Penultimate
layer features were then aggregated within participant, using both the mean feature
vector and a loss-minus-gain contrast, to form learned EEG embeddings for chronotype
prediction.

### 2.10 Chronotype-decoding evaluator, fusion, and continuous-MEQ prediction

Behavioural, EEG, reinforcement-learning, and fused embeddings were evaluated with
the same chronotype-decoding procedure. The classifier used standardization, PCA,
and L2 logistic regression within nested cross-validation. The outer loop was
leave-one-participant-out; the inner loop selected PCA dimensionality and logistic
regularization. This produced one out-of-fold chronotype score per participant. For
inference, the full nested procedure was repeated for 1000 label permutations.
Out-of-fold scores were also correlated with continuous MEQ score.

The main fusion analysis concatenated the GRU behavioural embedding with the six
validated ERP contrast features. Behaviour-only and ERP-only models were evaluated
with the same procedure. Continuous MEQ was also predicted directly with nested
leave-one-out Ridge regression, using predicted-versus-observed Pearson r and a
1000-permutation test. Robustness of the fused model was assessed with a
participant-level bootstrap confidence interval, pre-defined participant exclusions,
and leave-one-subject-out influence analysis.

### 2.11 Single-trial P300 → next-choice coupling

The single-trial coupling analysis tested whether feedback P300 on one trial was
related to the next risky choice, and whether this relationship differed by
chronotype. Each trial was paired with the following trial. The P300 at trial *t*
therefore occurred after the current choice but before the next choice. For each
participant, we estimated a logistic slope predicting next-trial risky choice from
within-participant z-scored trial-*t* P300. We tested both an overall P300 slope and
a valence-resolved P300 by loss interaction. Participant slopes were compared
between chronotypes with Mann-Whitney tests and bootstrap effect sizes. A binomial
mixed model with a P300 by chronotype interaction and random P300 slopes provided a
confirmatory analysis.

All computational analyses used fixed seed 0. Reproducibility scripts are listed in
the repository and in the supplementary materials.

### 2.12 Analysis plan and inference hierarchy

To keep interpretation disciplined given the sample size, analyses are organized
into a fixed hierarchy. The **primary** analysis is the theory-driven group
contrast in the posterior P300 loss-minus-gain response (Pz and POz), FDR-corrected
across the theory-driven feature family. **Confirmatory robustness** analyses test
the stability of that effect: participant-exclusion sensitivity analyses, the
window specification curve, and the direction of the continuous-MEQ association.
**Secondary, convergent** analyses ask whether behaviour carries related chronotype
information under participant-generalizing validation: the compact 12-feature
classifier, the GRU behavioural embedding, and its combination with the validated
ERP contrasts. **Exploratory** analyses, interpreted cautiously and not used to
support the main claims, comprise the higher-dimensional models, the
reinforcement-learning parameters, the learned EEGNet embeddings, and the
single-trial P300-to-choice coupling. Predictive results are reported as evidence
that behavioural and neural measures carry convergent chronotype information, not as
a deployable classifier.

---

## 3. Results

### 3.1 Primary: posterior P300 distinguishes chronotypes (Figure 1)

The main ERP result was a group difference in the posterior P300 response to
feedback valence. At Pz, Evening participants showed a negative loss-minus-gain
contrast, whereas Morning participants showed a positive contrast (Evening mean =
-0.96, Morning mean = 0.31; Cohen's d = -1.04, 95% CI [-1.63, -0.59]; Welch p =
0.0028; Mann-Whitney p = 0.005). The same pattern appeared at POz (d = -0.92, 95%
CI [-1.55, -0.39]; Welch p = 0.0076; Mann-Whitney p = 0.002). These were the only
features that survived Benjamini-Hochberg correction across the theory-driven
feature family (FDR p = 0.034 and 0.045).

### 3.2 Supporting: behavioural risk-taking

Evening participants tended to choose the risky option more often than Morning
participants. The largest uncorrected effects were for loss-error risky rate (d =
0.81), free risky rate (d = 0.80), and gain-correct risky rate (d = 0.77), with all
Welch p-values below 0.025. These effects did not survive FDR correction, and their
confidence intervals remained compatible with small effects. The behavioural group
differences are therefore supportive rather than primary.

Frontocentral FRN contrasts did not show reliable group differences. For example,
the Fz FRN error-minus-correct contrast had d = -0.60 and FDR p = 0.14. Bayes
factors were most consistent with a null group difference at FCz (BF01 = 2.46) and
Cz (BF01 = 3.17), while the Fz result remained inconclusive (BF01 = 0.82).
Equivalence tests with bounds of ±0.5 SD did not establish formal equivalence (for
example, Fz TOST p = 0.43; Supplementary Tables S7 and S8). Thus, the FRN results do not support a chronotype
effect, but the sample is not large enough to rule out medium-sized FRN differences.

### 3.3 Continuous MEQ association (Figure 2)

The P300 effect was also examined against continuous MEQ score (n = 38). Both
posterior electrodes showed positive associations in the expected direction. At Pz,
Pearson r was 0.29 (95% CI [0.06, 0.49]) and Spearman rho was 0.32. At POz, Pearson
r was 0.24 (95% CI [-0.01, 0.46]) and Spearman rho was 0.30. These associations
were modest, as expected for a sample with limited power for correlations
(minimum detectable r approximately 0.44 at 80% power) and 12 participants in the
MEQ intermediate band. The continuous and categorical analyses point in the same
direction, but the group contrast is estimated more precisely in this dataset.

### 3.4 Robustness across exclusions and analysis windows (Figure 1)

The posterior P300 group difference was stable across participant exclusions. In
the full sample and in all three exclusion scenarios, the Pz loss-minus-gain effect
remained close to d = -1.0 (range -1.00 to -1.07), with Welch p < 0.011 and a 95%
confidence interval excluding zero in every case. POz showed the same pattern
(Supplementary Table S3). By contrast, the participant-level classifier was more
sample-sensitive: in the exclusion-sensitivity analysis its permutation p rose from
0.034 in the full sample to 0.38 when the two label-conflict participants were
removed (this is the exclusion-tracking permutation test, distinct from the primary
nested-CV permutation test of the classifier reported in Section 3.5, p = 0.020).
The univariate P300 contrast is therefore the more stable result.

The P300 result was not dependent on one narrow analysis window. A 72-cell
specification curve (Supplementary Table S1 and Figure S1) varied channel (Pz or
POz), window centre (400-600 ms), window
width (50 or 100 ms), and summary measure (mean or peak). The loss-minus-gain group
difference had the expected sign in 64 of 72 cells. Nineteen cells had both a large
effect (d < -0.8) and p < 0.05. The strongest cells clustered around the
pre-specified 450-550 ms P300 window. The independently recomputed anchor cells were
slightly smaller than the validated-feature estimates (Pz d = -0.84, POz d =
-0.66), but they supported the same conclusion.

### 3.5 Participant-level chronotype classification (Figure 5)

The theory-driven 12-feature set also predicted chronotype in nested
cross-validation. Among the five classifiers, the pre-specified L2 logistic
regression performed best on balanced accuracy (0.717 +/- 0.14), with accuracy =
0.715, ROC AUC = 0.750, and, averaged across outer folds, Morning sensitivity =
0.695 and specificity = 0.738. The random
forest had the highest AUC (0.772), but the primary model was retained as planned.
Its selected regularization was strong (C = 0.01), consistent with the small sample.

The primary model correctly labelled 28 of 39 participants when out-of-fold
predictions were pooled (pooled accuracy = 0.718; Morning sensitivity = 0.75;
specificity = 0.68; pooled ROC AUC = 0.79; confusion matrix in Supplementary
Table S5). The pooled sensitivity/specificity differ slightly from the fold-averaged
values above because they are computed on the pooled out-of-fold predictions. A label-permutation test that re-fit the full nested
pipeline on each permutation indicated above-chance performance (observed balanced
accuracy = 0.717, null mean = 0.509, p = 0.020).

The model's strongest standardized coefficient was Pz P300 loss-minus-gain (+0.34,
predicting Morning), followed by loss-error risky rate (-0.31), gain-correct risky
rate (-0.24), and POz P300 loss-minus-gain (+0.24). Held-out permutation importance
showed the same ordering (Supplementary Table S6). Thus, the classifier relied most
strongly on the same posterior P300 contrast identified in the univariate analysis.

The predictive result should be interpreted cautiously. No pre-specified feature
pack survived FDR correction across the feature-pack family (best raw permutation p
= 0.051, FDR p = 0.175). The classifier was also sensitive to the label-conflict
participants, unlike the P300 group contrast. Higher-dimensional models performed
better numerically but are reported only as exploratory because the sample was small
relative to the number of predictors. The classifier is therefore an interpretable
multivariate check on the neural result, not a diagnostic model.

### 3.6 Secondary trial-level risky-choice prediction

As a preliminary check that trial-level choice is predictable at all under
participant-generalizing validation, leakage-safe models reached balanced accuracy
of approximately 0.587 (ROC AUC approximately 0.62) on 10,669 free-choice trials —
modestly above the majority-class (0.50) and previous-choice-persistence (0.554)
baselines and approaching the participant-mean oracle ceiling (0.604) without using
held-out-participant data. Most of the signal came from previous-trial and rolling
choice history, and previous-trial EEG added no clear value in this representation
(Supplementary Table S4). This motivates the sequence model in the next section,
which learns such temporal structure directly rather than from hand-engineered
history features.

### 3.7 Chronotype information in risky-choice dynamics (Figure 3)

The GRU model predicted trial-level risky choice from pre-choice context and the
previous outcome with balanced accuracy = 0.603 and AUC = 0.647. The participant
embeddings derived from this model then predicted Morning versus Evening chronotype
under nested leave-one-participant-out cross-validation (ROC AUC = 0.713; balanced
accuracy = 0.691; label-permutation p = 0.027; null AUC mean = 0.457). The same
out-of-fold scores were related to continuous MEQ score (r = -0.31). These results
show that chronotype is expressed in behavioural choice dynamics even without EEG
features.

### 3.8 Combining behaviour and ERP features improves chronotype prediction (Figure 3)

The six validated ERP contrast features predicted chronotype at AUC = 0.668 (p =
0.032). When these ERP features were combined with the GRU behavioural embedding,
prediction improved to AUC = 0.797 (balanced accuracy = 0.742, permutation p =
0.004). This exceeded the behavioural and ERP-only models. The fused out-of-fold
score also showed the strongest relation to continuous MEQ score (r = -0.42, p =
0.009).

| Model | n features | ROC AUC | Balanced acc | Perm p | MEQ r |
|---|---|---|---|---|---|
| Behavioural (GRU embedding) | 64 | 0.713 | 0.691 | 0.027 | -0.31 |
| Neural (validated ERP P300/FRN) | 6 | 0.668 | 0.667 | 0.032 | -0.10 |
| **Fused (behaviour + ERP)** | 70 | **0.797** | **0.742** | **0.004** | **-0.42** |

The improvement from fusion suggests that behavioural dynamics and validated ERP
features contain partly independent chronotype information. In contrast, combining
the GRU embedding with the learned EEGNet embedding reduced performance (AUC =
0.65), supporting the use of low-dimensional ERP features for this sample. The fused
model remained above chance across the pre-defined exclusion analyses (AUC range =
0.70-0.80). Its bootstrap 95% CI was [0.639, 0.924], and leave-one-subject-out
analysis gave an AUC range of 0.653-0.853. The most influential participant was
1001.

### 3.9 Continuous MEQ prediction (Figure 3)

Nested leave-one-out Ridge regression was used to predict continuous MEQ score
(n = 38). Behavioural embeddings predicted MEQ (r = 0.310, p = 0.039). ERP features
alone were weaker (r = 0.145). The fused feature set performed best (r = 0.344,
p = 0.027), matching the pattern observed for binary chronotype prediction.

### 3.10 Reinforcement-learning parameters were weakly identified

The per-participant maximum-likelihood reinforcement-learning fits suggested a
possible mechanism, but the evidence was not stable. Point estimates indicated
stronger gain learning in Evening participants (α_gain Evening = 0.23, Morning =
0.05; group p = 0.040) and lower choice consistency (β; MEQ r = 0.36, p = 0.027).
There was also a trend toward relatively greater loss learning in Morning
participants (learning-asymmetry p = 0.072). As a feature set, however, the RL
parameters classified chronotype only weakly (AUC = 0.532).

The apparent MLE contrasts did not persist under Bayesian hierarchical partial
pooling. The sampler diagnostics were acceptable (max R-hat = 1.02; no
divergences), but the α_gain group contrast collapsed to approximately zero
(Evening minus Morning approximately 0.00, 94% HDI [-0.010, 0.011], P(contrast > 0)
= 0.46). The β contrast reversed sign and remained uncertain (+0.38, HDI [-0.53,
1.28]). The only remaining tendency was weaker loss learning in Evening
participants (α_loss contrast = -0.012, HDI [-0.032, 0.002]; learning-asymmetry
P(contrast > 0) = 0.06). MLE and hierarchical subject-level estimates were weakly
correlated (α_loss r = 0.02, β r = 0.13, α_gain r = 0.45; Supplementary Table S2).
Given the limited number of free trials and the hidden, random signs of the boxes,
these parameters were not identified well enough to support a mechanistic conclusion.

### 3.11 EEGNet decoded feedback valence but not chronotype

EEGNet provided a useful positive control. When trained across participants, it
decoded single-trial feedback valence (loss vs gain) at AUC = 0.641 on held-out
participants. Thus, the cleaned epochs contained decodable single-trial feedback
information. The same learned representations did not predict chronotype. The
mean-pooled EEG embedding had AUC = 0.426 (p = 0.61), and the loss-minus-gain
contrast embedding had AUC = 0.389 (p = 0.71). A density-ratio Bayes factor based on
the permutation null was weakly informative (BF01 approximately 1.2), partly because
the observed AUCs were slightly below chance (Supplementary Table S7). The permutation tests are therefore
the clearest summary: learned EEG embeddings did not recover chronotype in this
sample. The positive-control valence AUC of 0.64 comes from the canonical EEGNet run
with 30 training epochs and 5-fold participant-grouped cross-validation; a separate
2-epoch/2-fold sanity run gave AUC = 0.59 and is not used here.

### 3.12 Single-trial P300 did not predict next-choice shifts by chronotype (Figure 4)

The final analysis tested whether feedback P300 on trial *t* predicted risky choice
on trial *t*+1 differently by chronotype. Across 10,630 consecutive trial pairs,
the per-participant coupling slope did not differ reliably between groups. The
overall slope was slightly negative in Evening participants and slightly positive in
Morning participants (Evening = -0.033, Morning = 0.025; d = -0.36, 95% CI [-1.12,
0.25]; Mann-Whitney p = 0.22). The valence-resolved P300 by loss interaction also
did not differ by chronotype (d = -0.08, p = 0.62). A confirmatory binomial mixed
model gave p = 0.12 for the P300 by chronotype interaction. The chronotype effect
therefore appears to be a between-participant difference in feedback evaluation and
choice dynamics, rather than a strong within-participant trial-to-trial coupling
from P300 to the next choice.

---

## 4. Discussion

### 4.1 Overview

In a single cohort of 39 participants, chronotype was related to both the neural and
behavioural evaluation of decision feedback. Four findings anchor the account.
First, Morning and Evening chronotypes differed in the feedback-locked posterior
P300 to outcome valence. Second, chronotype could be predicted from risky-choice
dynamics alone. Third, combining behavioural dynamics with validated ERP contrasts
predicted chronotype better than either modality alone and also predicted continuous
MEQ score. Fourth, two planned negative tests, end-to-end EEG decoding and
within-subject single-trial coupling, placed the effect at the between-participant
trait level rather than at the level of moment-to-moment neural control of choice.
All predictive analyses used participant-generalizing cross-validation and
permutation testing. The main evidentiary value is the agreement across distinct
analyses, not the accuracy of any single model.

### 4.2 The posterior P300 and the neural evaluation of feedback

The pre-specified neural hypothesis was supported. Evening participants showed a
more negative loss-minus-gain P300 contrast at posterior sites, with a large effect
(d approximately 1.0) that survived FDR correction. The effect was stable across
participant-exclusion analyses and across the 72-cell specification curve, and it
was directionally consistent with the continuous MEQ analysis. Because posterior
P300 amplitude is linked to motivational salience and subjective outcome
significance (Polich, 2007; Yeung & Sanfey, 2004), this pattern suggests that
chronotype is related to how losses and gains are weighted during feedback
evaluation.

The direction of the effect fits prior work showing greater risk-taking,
reward-sensitivity, and impulsivity in Evening types (Adan et al., 2010; Killgore,
2007; Muro et al., 2012), as well as chronotype-related differences in reward brain
function (Hasler et al., 2013; Hasler & Clark, 2013) and circadian modulation of
reward motivation (Murray et al., 2009). A smaller posterior-P300 response to losses
than gains in Evening participants is consistent with a profile in which losses
carry less relative weight during feedback evaluation (cf. loss aversion; Kahneman &
Tversky, 1979). [AUTHOR INPUT: the team may sharpen this direction-of-effect link
to the specific prior findings it wishes to foreground.] The dissociation between the P300 and the FRN is theoretically informative rather
than merely a pattern of one significant and one null test. The FRN, and its
positive-going counterpart the reward positivity, indexes a fast, relatively
automatic valence and reward-prediction-error evaluation of outcomes (Holroyd &
Coles, 2002; Proudfit, 2015; Sambrook & Goslin, 2015), whereas the posterior P300
indexes a later, more elaborative weighting of outcome salience and significance
(Polich, 2007; Yeung & Sanfey, 2004). That chronotype was expressed in the P300 but
not the FRN suggests the difference lies in how much motivational significance is
assigned to gains relative to losses during this later appraisal stage, rather than
in the initial registration of outcome valence — a more specific account than a
general claim that chronotype affects feedback processing.

### 4.3 Combining behavioural and neural evidence

The strongest support for the interpretation comes from combining behavioural and
neural evidence. Chronotype was predicted from risky-choice dynamics alone (GRU AUC
= 0.713) and from validated ERP contrasts alone (AUC = 0.668). Combining the two
raised prediction to AUC = 0.797 and strengthened the association with continuous
MEQ score (r = -0.42). This improvement suggests that long-run choice dynamics and
feedback-locked neural responses capture related but non-identical aspects of
chronotype.

The fusion result also aligns with the interpretable feature analysis. The most
influential multivariate feature was the same posterior P300 contrast that drove the
univariate group effect. At the same time, the predictive findings should not be
overstated. The single-model classifier was modest, did not survive correction
across the feature-set family, and was sensitive to two participants, although their
labels were supported by MEQ. The fused estimate also carries uncertainty
(bootstrap CI [0.64, 0.92]). These models are best viewed as evidence for a
cross-modal trait signature, not as tools for individual classification.

### 4.4 A tentative computational mechanism

The asymmetric reinforcement-learning model was included to ask whether the
behavioural and neural findings might reflect altered learning from gains and
losses (Frank et al., 2007; Lefebvre et al., 2017). The initial MLE estimates were
compatible with greater gain-driven learning and lower choice consistency in Evening
participants. That account would fit a broader profile of reward sensitivity and
risk-sensitive learning (Niv et al., 2012), possibly under circadian or dopaminergic
modulation of reward function (McClung, 2007; Murray et al., 2009; Webb et al.,
2009).

The model did not provide stable mechanistic evidence. The MLE parameter contrasts
did not survive hierarchical partial pooling, and subject-level MLE estimates agreed
poorly with the hierarchical estimates. In this hidden-sign task, with a limited
number of free trials per participant, the RL parameters were weakly identified. The
mechanistic account should therefore be treated as a hypothesis for a trial-richer
follow-up, not as a confirmed finding. The main behavioural and ERP results do not
depend on the RL model. [AUTHOR INPUT: align this tentative interpretation with the
constructs/references in the team's prior chronotype work.]

### 4.5 Two informative negatives and a methodological message

Two planned negative tests sharpen rather than weaken the account. First,
end-to-end deep learning on single-trial EEG (EEGNet) decoded the feedback task
across unseen participants but did not recover chronotype from learned single-trial
features, whereas a small set of theory-driven, FDR-validated ERP contrasts did.
At the sample sizes typical of ERP research, validated low-dimensional features can
thus outperform representation learning for subtle individual-difference signals,
and naive feature learning can dilute a real effect. This is a practical message
for small-sample individual-differences EEG, where deep models are data-hungry
(Roy et al., 2019) and small samples inflate cross-validated error (Varoquaux,
2018). Second, the chronotype effect was not expressed as a within-subject,
single-trial coupling between feedback P300 and the subsequent choice. The combined
model therefore joins two trait-level signatures: a stable difference in feedback
evaluation and a stable difference in choice dynamics. It should not be interpreted
as evidence for moment-to-moment neural control of the next risky choice.

### 4.6 Chronotype as a dimensional trait

Although the primary analyses used a binary Morning/Evening label, the same feature
sets predicted continuous MEQ score (fused r = 0.34). This matters because 12 of 39
participants fell in the MEQ intermediate band, where the dichotomy is inherently
soft. The dimensional result indicates that the effect is not only a product of
dichotomization. Future work should model continuous morningness-eveningness as the
primary outcome.

### 4.7 Conclusion

This single-cohort study provides preliminary evidence that chronotype is related to
the neural evaluation of decision feedback, with the clearest effect in the
posterior P300. Behavioural choice dynamics carried additional chronotype
information, and combining behavioural and ERP features gave the strongest
prediction of both categorical chronotype and continuous MEQ score. The findings
are internally validated and require independent replication, but they show how
carefully constrained ERP features and behavioural sequence models can be combined
to study stable individual differences in modest samples.

---

## 5. Limitations

- The sample is a single cohort of 39 participants and is powered only for large
  effects; medium and small effects are inconclusive. Findings require
  independent replication.
- Chronotype is analysed as a binary Morning/Evening label. The labels are
  MEQ-derived and were checked against continuous MEQ score. All 26 decisively
  scored participants were consistent with the binary label, and both raw-behaviour
  conflict cases were MEQ-confirmed. However, 12 of 39 participants fell in the MEQ
  intermediate band, where the dichotomy is inherently soft.
- ERP features are window-level single-trial means and may miss peak-latency,
  time-frequency, or trial-quality effects.
- Participant 1013 has a known EEG/trigger agreement issue after block 10.
- Although the pre-specified classifier is significant under nested
  cross-validation (permutation p = 0.02), it does not survive FDR correction
  across the family of feature sets and is sensitive to two participants; it is
  therefore interpreted as an interpretable complement to, not independent
  confirmation of, the neural effect. Larger samples are needed for a robust
  predictive model.
- All predictive results are internally validated only on one cohort of 39.
  Robustness was assessed within sample using bootstrap intervals, exclusion
  analyses, and leave-one-subject-out influence checks, but not against a new
  dataset. Removing the most influential participant lowered the fused AUC from
  0.80 to 0.65, so the estimate carries real uncertainty (CI [0.64, 0.92]).
- The reinforcement-learning parameters were weakly identified (limited to ~270
  free trials per participant with hidden random signs). The MLE contrasts did not
  survive a hierarchical partial-pooling refit, and MLE and hierarchical subject
  estimates were weakly correlated. The RL account is therefore exploratory, not a
  confirmed mechanism; the predictive and neural findings do not depend on it.
- Findings concern one feedback-based risky-choice paradigm with hidden signs;
  generalization to other reward/decision tasks is untested.
- There is no external validation cohort.

---

## 6. Reproducibility and data availability

Analysis code is available in this repository. The environment is pinned in
`requirements.txt` (Python 3.11) with a full freeze in `requirements.lock.txt`.
Raw data are held locally and not committed. [AUTHOR INPUT: decide what to share
because the derived participant-level table (39 rows) is a candidate for public
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
- [ ] Confirm whether any details from `methodology_dl.md` sections 5-8 still need
      to be folded into Methods 2.7-2.11.
- [x] Parallel-analysis results integrated: Bayes factors (G-B, §3.2/§3.11),
      equivalence tests (G-F, §3.2), P300 specification curve (G-E, §3.4),
      hierarchical RL (G-D, §3.10). The MLE mechanism did not survive pooling, and
      RL is now framed as exploratory.
- [ ] Confirm mechanistic interpretation (RL / reward-sensitivity) against the
      team's prior chronotype constructs and references.
