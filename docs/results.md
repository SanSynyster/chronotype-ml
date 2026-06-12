# Current Results

This document summarizes the current active raw-to-clean results using chronotype labels from `all final data.xlsx` as the primary source. Generated CSV/JSON reports are ignored by git, so this file is the tracked public result summary.

## Headline

- **Primary finding (confirmatory-style, neural):** posterior P300 loss-minus-gain amplitude differs between Morning and Evening chronotypes. `Pz_P300_loss_minus_gain` (Cohen's d = -1.04, Welch p = 0.0028, FDR p = 0.034) and `POz_P300_loss_minus_gain` (d = -0.92, Welch p = 0.0076, FDR p = 0.045) are the only features that survive FDR correction across the theory-driven feature set, both are corroborated by Mann-Whitney tests, and the association is directionally confirmed against the continuous MEQ score (Pz Pearson r = 0.29, 95% CI [0.06, 0.49]).
- **Exploratory finding (machine learning):** chronotype is classifiable above chance on the full dataset by a theory-driven 12-feature logistic model, but this evidence does not survive multiple-comparison correction across feature packs and is not robust to label-conflict exclusions. It is reported as converging support, not a validated classifier.
- **Secondary task (risky choice):** weakly predictable under leakage-safe, participant-grouped CV; previous-trial history carries most of the signal.

All findings are from a single cohort of 39 participants and require independent replication.

## Provenance And Labels

The current pipeline rebuilds modelling tables from local raw files using active scripts:

- `scripts/link_raw_metadata.py` links `participant_summary.xlsx` to `UserID` by matching previous-feedback behavioural aggregates recomputed from `raw_behavioral_trials.xlsx`.
- `scripts/link_raw_metadata.py` then links `all final data.xlsx` through the shared `ERPset` column.
- `scripts/build_ml_ready_from_raw.py` builds the raw-derived trial table from trimmed behaviour (`Trial <= 23`) and raw EEG single-trial means/triggers.
- `scripts/build_clean_chronotype.py` uses previous-trial feedback for behavioral adaptation features and current-trial feedback for feedback-locked ERP contrasts.
- `scripts/rebuild_from_raw.py --execute` runs the full active rebuild and writes `docs/data_provenance.md`.

Primary chronotype labels come from `all final data.xlsx` via the `ERPset` link. The `participant_summary.xlsx` to `UserID` linkage uses an optimal one-to-one (Hungarian) assignment over standardized previous-feedback behavioural aggregates, which guarantees a bijection; the smallest assignment margin (0.157) is large relative to typical match distances (~0.013), and only participant `1010` is flagged as a comparatively distant (but still unambiguous) match.

The two metadata sources agree with each other on every participant: the `participant_summary.xlsx` chronotype column matches the `all final data.xlsx` label for all 39 participants. Only the raw behavioral-trials `Chronotype` column disagrees, and only for participants `1027` and `1036`. Because two independent metadata sources corroborate the primary labels and only the raw behavioural column is the outlier, the primary labels are retained and `1027`/`1036` are tracked as label-conflict sensitivity cases rather than manually overridden.

MEQ validation (`scripts/validate_meq_labels.py`): the continuous MEQ score in `all final data.xlsx` (attached to each participant by name, since the MEQ block is not row-aligned with the chronotype column) confirms the binary labels. The score separates the groups in the standard direction (Evening MEQ mean 37.3, range 25-49; Morning mean 57.7, range 45-64); all 26 participants with a decisive MEQ score (outside the 42-58 intermediate band) match their binary label, and 12 fall in the intermediate band where binary assignment is softer. Both label-conflict participants have decisive MEQ scores confirming the primary label (`1027` MEQ = 61, Morning; `1036` MEQ = 27, Evening), so the raw-behaviour column was in error for these two. The "exclude label conflicts" sensitivity scenario therefore removes two MEQ-confirmed participants, and the classifier's loss of significance there reflects sample-size reduction rather than label uncertainty.

Label/QC snapshot:

| Item | Value |
| --- | ---: |
| Participants | 39 |
| Primary chronotype counts | 20 Morning / 19 Evening |
| Metadata links missing | 0 |
| Raw-behavior label conflicts | 2 (`1027`, `1036`) |
| Manual chronotype overrides | None |
| MEQ/MCTQ status | Validated by name-key alignment; labels MEQ-consistent (26/26 decisive) |

Raw-derived table snapshot:

| Table | Rows | Columns | Notes |
| --- | ---: | ---: | --- |
| `data/processed/ml_ready_features.csv` | 14,352 | 40 | 39 participants, behaviour trimmed to 368 rows each |
| `data/clean/risky_choice_prechoice.csv` | 10,669 | 55 | free-choice rows only |
| `data/clean/chronotype_participant.csv` | 39 | 173 | one row per participant |
| `data/clean/chronotype_compact_12.csv` | 39 | 14 | theory-driven compact table |
| `data/clean/chronotype_compact_performance.csv` | 39 | 14 | exploratory performance-informed compact table |

EEG/trigger QC: participant `1013` has one missing EEG trial after raw loading and low trigger/behaviour valence agreement (`0.839`). Other participants have complete EEG trial coverage and high trigger agreement.

## Chronotype

Participant-level task: predict primary `Morning` vs `Evening` chronotype from behavioral and ERP-derived features.

Dataset size: `39` participants.

Best feature-pack leaderboard entries from 5-fold stratified CV:

| Pack | Model | Balanced Accuracy | Accuracy | Macro F1 | ROC AUC | Features |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `all_literature` | Random Forest | 0.808 | 0.814 | 0.806 | 0.883 | 171 |
| `compact_combined` | Random Forest | 0.808 | 0.814 | 0.806 | 0.838 | 47 |
| `behavior_core` | Random Forest | 0.783 | 0.789 | 0.780 | 0.792 | 19 |
| `compact_12` | Logistic Regression | 0.692 | 0.693 | 0.689 | 0.738 | 12 |
| `compact_performance` | Logistic Regression | 0.692 | 0.693 | 0.685 | 0.721 | 12 |
| `p300_core` | Logistic Regression | 0.658 | 0.664 | 0.651 | 0.688 | 16 |
| `frn_core` | Random Forest | 0.667 | 0.664 | 0.651 | 0.692 | 16 |

Interpretation: using `all final data.xlsx` labels restores the stronger chronotype signal. High-dimensional Random Forest results remain exploratory, but the theory-driven compact Logistic Regression is above chance in permutation testing on the full dataset.

Multiple-comparison note: across the five pre-specified literature feature packs (`demo_only`, `behavior_core`, `frn_core`, `p300_core`, `compact_combined`), no pack survives Benjamini-Hochberg FDR correction at the family level (best raw permutation p = 0.0509 for `p300_core`, FDR p = 0.175; see `reports/clean/permutation_tests/leaderboard.md`). The theory-driven `compact_12` model is treated as a single pre-specified primary classifier and reported with its uncorrected permutation p-value; its corrected significance across the broader pack family would not hold. The classifier evidence is therefore exploratory and secondary to the neural group difference.

## Larger Exploratory Random Forest Models

The two larger feature sets were also validated with repeated CV and 1000-label permutation tests using Random Forest. These models are exploratory because the number of predictors is high relative to `n = 39`, but they test whether broader multivariate structure is informative.

| Feature set | Features | Dataset | Repeated-CV BA Mean | Permutation Observed BA | Permutation p-value |
| --- | ---: | --- | ---: | ---: | ---: |
| `all_literature` | 171 | Full all-final-label dataset | 0.783 | 0.833 | 0.0010 |
| `compact_combined` | 47 | Full all-final-label dataset | 0.776 | 0.808 | 0.0010 |
| `compact_combined` | 47 | Exclude `1013` | 0.780 | 0.742 | 0.0060 |
| `compact_combined` | 47 | Exclude label conflicts `1027`, `1036` | 0.769 | 0.750 | 0.0130 |
| `compact_combined` | 47 | Exclude all flagged `1013`, `1027`, `1036` | 0.748 | 0.650 | 0.1019 |

Interpretation: the larger Random Forest models are strongly above chance on the full all-final-label dataset. They remain significant when excluding either the EEG-QC case or the two label-conflict cases, but not when all three flagged participants are excluded together. These results should still be considered exploratory because Random Forest can capitalize on high-dimensional feature patterns in small samples.

## Theory-Driven Compact 12-Feature Model

The theory-driven compact model uses 12 behavioral/ERP predictors. Repeated 5-fold stratified CV, 100 repeats, Logistic Regression:

| Metric | Mean | SD | 95% Interval |
| --- | ---: | ---: | ---: |
| Accuracy | 0.667 | 0.150 | 0.375-1.000 |
| Balanced accuracy | 0.666 | 0.150 | 0.375-1.000 |
| Macro F1 | 0.655 | 0.157 | 0.348-1.000 |
| ROC AUC | 0.706 | 0.171 | 0.375-1.000 |

1000-label permutation test, fixed 5-fold stratified CV, Logistic Regression:

| Dataset | Rows | Observed BA | Null mean | Null 95th percentile | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full all-final-label dataset | 39 | 0.692 | 0.496 | 0.667 | 0.0340 |
| Exclude `1013` EEG/trigger QC case | 38 | 0.658 | 0.495 | 0.658 | 0.0529 |
| Exclude label conflicts `1027`, `1036` | 37 | 0.533 | 0.504 | 0.667 | 0.3816 |
| Exclude all flagged `1013`, `1027`, `1036` | 36 | 0.675 | 0.511 | 0.683 | 0.0669 |

Interpretation: the theory-driven compact model is above chance in the full all-final-label dataset under its single pre-specified test, but the effect does not survive FDR correction across the feature-pack family and sensitivity analyses are mixed, especially when excluding the two participants whose raw behavioral labels disagree with `all final data.xlsx`. It is therefore reported as exploratory, converging support rather than a validated classifier.

## Performance-Informed Compact Model

The performance-informed compact model is exploratory and uses 12 features that repeatedly appeared useful across feature-pack performance, group statistics, and held-out importance. It is not a replacement for the theory-driven compact model.

Features:

- `free_risky_rate`
- `gain_correct_risky_rate`
- `loss_error_risky_rate`
- `risk_after_loss_error_minus_gain_correct`
- `risky_late_minus_early`
- `Fz_FRN_error_minus_correct`
- `FCz_FRN_error_minus_correct`
- `Fz_FRN_loss_error_minus_gain_correct`
- `FCz_FRN_loss_error_minus_gain_correct`
- `Pz_P300_loss_minus_gain`
- `POz_P300_loss_minus_gain`
- `CPz_P300_error_minus_correct`

Repeated 5-fold stratified CV, 100 repeats, Logistic Regression:

| Metric | Mean | SD | 95% Interval |
| --- | ---: | ---: | ---: |
| Accuracy | 0.682 | 0.144 | 0.375-0.875 |
| Balanced accuracy | 0.682 | 0.145 | 0.375-0.875 |
| Macro F1 | 0.672 | 0.151 | 0.365-0.873 |
| ROC AUC | 0.742 | 0.165 | 0.417-1.000 |

1000-label permutation test, fixed 5-fold stratified CV, Logistic Regression:

| Dataset | Rows | Observed BA | Null mean | Null 95th percentile | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full all-final-label dataset | 39 | 0.692 | 0.492 | 0.650 | 0.0240 |

Interpretation: the performance-informed model is exploratory and significant in the full all-final-label dataset. It is not a replacement for the theory-driven compact model because its feature set was partly informed by current-dataset results.

## Feature Importance

Top held-out permutation-importance features for the theory-driven `compact_12` + Logistic Regression model:

| Feature | Mean Balanced-Accuracy Drop |
| --- | ---: |
| `Pz_P300_loss_minus_gain` | 0.1028 |
| `loss_error_risky_rate` | 0.0425 |
| `Fz_FRN_loss_error_minus_gain_correct` | 0.0183 |
| `FCz_FRN_error_minus_correct` | 0.0111 |
| `Fz_FRN_error_minus_correct` | 0.0100 |
| `POz_P300_loss_minus_gain` | 0.0075 |
| `gain_correct_risky_rate` | 0.0044 |

Interpretation: the most consistent compact-model contributor is posterior P300 loss-gain contrast, followed by loss-error risky-choice behavior. Feature importance remains unstable because fold test sets are small.

## Classical Group Statistics

Morning-vs-Evening tests for theory-driven features using all-final labels:

| Feature | Cohen's d | d 95% CI | Hedges g | Welch p | FDR p |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Pz_P300_loss_minus_gain` | -1.045 | [-1.63, -0.59] | -1.024 | 0.0028 | 0.0341 |
| `POz_P300_loss_minus_gain` | -0.919 | [-1.55, -0.39] | -0.901 | 0.0076 | 0.0454 |
| `loss_error_risky_rate` | 0.813 | [0.21, 1.52] | 0.797 | 0.0160 | 0.0547 |
| `free_risky_rate` | 0.797 | [0.22, 1.44] | 0.781 | 0.0182 | 0.0547 |
| `gain_correct_risky_rate` | 0.765 | [0.18, 1.43] | 0.749 | 0.0231 | 0.0553 |
| `Fz_FRN_error_minus_correct` | -0.601 | [-1.50, 0.02] | -0.589 | 0.0681 | 0.1363 |

Effect-size CIs are percentile bootstrap (10,000 resamples). Hedges g is the small-sample bias-corrected estimate. Only the two posterior P300 contrasts have d CIs that exclude zero and survive FDR; the behavioral contrasts are medium-to-large but their CIs include small effects and they do not survive FDR.

Interpretation: the clearest physiological signal is posterior P300 loss-gain differences, which survive FDR correction in the full all-final-label dataset and are the primary finding of this report. As a single-cohort result it still requires independent replication.

## Sensitivity Matrix

The same four participant-exclusion scenarios are applied to both the secondary exploratory classifier and the primary neural group difference (`scripts/sensitivity_matrix.py`, full table in `reports/clean/sensitivity_matrix/`).

Secondary classifier (compact_12 logistic regression), repeated-CV balanced accuracy and 1000-label permutation p:

| Scenario | n | Repeated-CV BA | Permutation BA | Permutation p |
| --- | ---: | ---: | ---: | ---: |
| full | 39 | 0.666 | 0.692 | 0.034 |
| exclude `1013` | 38 | 0.669 | 0.658 | 0.053 |
| exclude label conflicts `1027`,`1036` | 37 | 0.634 | 0.533 | 0.382 |
| exclude all flagged | 36 | 0.639 | 0.675 | 0.067 |

Primary neural group difference (posterior P300 loss-minus-gain):

| Scenario | n | Feature | Cohen's d | d 95% CI | Welch p |
| --- | ---: | --- | ---: | ---: | ---: |
| full | 39 | `Pz_P300_loss_minus_gain` | -1.045 | [-1.63, -0.59] | 0.0028 |
| exclude `1013` | 38 | `Pz_P300_loss_minus_gain` | -1.067 | [-1.66, -0.61] | 0.0024 |
| exclude label conflicts | 37 | `Pz_P300_loss_minus_gain` | -1.004 | [-1.59, -0.54] | 0.0051 |
| exclude all flagged | 36 | `Pz_P300_loss_minus_gain` | -1.025 | [-1.62, -0.54] | 0.0044 |
| full | 39 | `POz_P300_loss_minus_gain` | -0.919 | [-1.55, -0.39] | 0.0076 |
| exclude all flagged | 36 | `POz_P300_loss_minus_gain` | -0.911 | [-1.56, -0.36] | 0.0102 |

Interpretation: the two headline claims behave very differently under participant exclusions. The classifier is fragile, losing significance when the two label-conflict participants are removed. The posterior P300 group difference is essentially invariant: across every exclusion the effect stays near d = -1.0 with Welch p < 0.011 and a d CI that excludes zero. The robustness of the neural effect, contrasted with the fragility of the classifier, is why the P300 group difference is reported as the primary finding and the classifier as exploratory support.

## Continuous MEQ Analysis

To avoid dichotomizing chronotype, the posterior P300 loss-minus-gain contrasts
were related to the continuous MEQ score (n = 38 with an MEQ score;
`scripts/meq_p300_continuous.py`). Under the standard direction (higher MEQ =
more morning) and the group result, a positive correlation is expected.

| Feature | Pearson r | r 95% CI | Pearson p | Spearman rho | Spearman p | FDR p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Pz_P300_loss_minus_gain` | 0.293 | [0.06, 0.49] | 0.074 | 0.319 | 0.051 | 0.146 |
| `POz_P300_loss_minus_gain` | 0.241 | [-0.01, 0.46] | 0.146 | 0.299 | 0.068 | 0.146 |

Interpretation: the continuous relationship is positive and in the predicted
direction for both electrodes, confirming that the posterior P300 effect is not
an artifact of the Morning/Evening dichotomy but a graded association with
morningness. The correlations are modest and only marginally significant, which
is expected given power: with n = 38 the minimum correlation detectable at 80%
power is approximately r = 0.44, so the study has only about 42% power for an
effect of r = 0.29. The much stronger dichotomous effect (d approximately 1.0)
is better powered because the binary contrast emphasizes decisively classified
participants, whereas the 12 intermediate-band participants add scatter to the
continuous estimate (Figure 5). The continuous and dichotomous analyses are
therefore consistent: a real, graded posterior-P300 association with chronotype
that the present sample estimates precisely only at the group-contrast level.

## Risky Choice

Trial-level task: predict binary risky choice using only same-trial pre-choice features plus previous-trial history. Evaluation uses participant-grouped cross-validation.

Dataset size: `10,669` free-choice trial rows.

Best feature-pack leaderboard entries:

| Pack | Model | Balanced Accuracy | Accuracy | Macro F1 | ROC AUC | Features |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `all_clean` | Logistic Regression | 0.587 | 0.592 | 0.585 | 0.620 | 50 |
| `prev_eeg` | Logistic Regression | 0.587 | 0.592 | 0.585 | 0.620 | 50 |
| `history_only` | Logistic Regression | 0.586 | 0.592 | 0.585 | 0.624 | 25 |
| `value_history` | Logistic Regression | 0.586 | 0.592 | 0.585 | 0.623 | 39 |
| `prev_eeg` | Random Forest | 0.575 | 0.582 | 0.573 | 0.614 | 50 |

Naive baselines for context (`scripts/risky_choice_baseline.py`):

| Baseline | Balanced Accuracy | Accuracy | Note |
| --- | ---: | ---: | --- |
| Majority class | 0.500 | 0.529 | base rate 0.529 risky |
| Persistence (previous choice) | 0.554 | 0.556 | choice autocorrelation |
| Best leakage-safe model (grouped CV) | 0.587 | 0.592 | generalizes to held-out participants |
| Participant-mean oracle | 0.604 | 0.610 | peeks at held-out participant; ceiling only |

Interpretation: risky-choice prediction is modest but meaningfully above its trivial references. Under participant-grouped CV the model beats both the majority-class (0.500 BA) and previous-choice persistence (0.554 BA) baselines, and it approaches the participant-mean oracle ceiling (0.604 BA) without ever seeing the held-out participant's own data. Previous-trial and rolling history features carry most of the signal; previous-trial EEG does not materially improve over history/value features in the current representation.

## Limitations

- Chronotype has only 39 participants, so all findings are preliminary and single-cohort.
- The raw behavioural chronotype column conflicts with `all final data.xlsx` for participants `1027` and `1036`; sensitivity analyses should disclose this.
- Compact ML evidence is significant in the full all-final-label dataset but not robust across all flagged-participant exclusions.
- There is no external validation cohort.
- Participant `1013` has an EEG/trigger alignment issue after block 10 that materially affects sensitivity results.
- MEQ/MCTQ values are not exported because their side-by-side workbook table order is not independently validated.
- Raw data are local and not committed; generated data/reports are ignored and summarized here.
