# Current Results

This document summarizes the current active raw-to-clean results using chronotype labels from `all final data.xlsx` as the primary source. Generated CSV/JSON reports are ignored by git, so this file is the tracked public result summary.

## Provenance And Labels

The current pipeline rebuilds modelling tables from local raw files using active scripts:

- `scripts/link_raw_metadata.py` links `participant_summary.xlsx` to `UserID` by matching previous-feedback behavioural aggregates recomputed from `raw_behavioral_trials.xlsx`.
- `scripts/link_raw_metadata.py` then links `all final data.xlsx` through the shared `ERPset` column.
- `scripts/build_ml_ready_from_raw.py` builds the raw-derived trial table from trimmed behaviour (`Trial <= 23`) and raw EEG single-trial means/triggers.
- `scripts/build_clean_chronotype.py` uses previous-trial feedback for behavioral adaptation features and current-trial feedback for feedback-locked ERP contrasts.
- `scripts/rebuild_from_raw.py --execute` runs the full active rebuild and writes `docs/data_provenance.md`.

Primary chronotype labels come from `all final data.xlsx` via the `ERPset` link. The raw behavioral `Chronotype` column disagrees for participants `1027` and `1036`, so those participants are tracked as label-conflict sensitivity cases rather than manually overridden.

Label/QC snapshot:

| Item | Value |
| --- | ---: |
| Participants | 39 |
| Primary chronotype counts | 20 Morning / 19 Evening |
| Metadata links missing | 0 |
| Raw-behavior label conflicts | 2 (`1027`, `1036`) |
| Manual chronotype overrides | None |
| MEQ/MCTQ status | Not exported; side-by-side table order is unvalidated |

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

Interpretation: the theory-driven compact model is statistically significant in the full all-final-label dataset. Sensitivity analyses are mixed, especially when excluding the two participants whose raw behavioral labels disagree with `all final data.xlsx`, so this remains pilot evidence rather than a validated classifier.

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

| Feature | Evening Mean | Morning Mean | Cohen's d | Welch p | FDR p |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Pz_P300_loss_minus_gain` | -0.963 | 0.308 | -1.045 | 0.0028 | 0.0341 |
| `POz_P300_loss_minus_gain` | -0.465 | 0.553 | -0.919 | 0.0076 | 0.0454 |
| `loss_error_risky_rate` | 0.634 | 0.530 | 0.813 | 0.0160 | 0.0547 |
| `free_risky_rate` | 0.583 | 0.477 | 0.797 | 0.0182 | 0.0547 |
| `gain_correct_risky_rate` | 0.552 | 0.423 | 0.765 | 0.0231 | 0.0553 |
| `Fz_FRN_error_minus_correct` | -2.895 | -2.023 | -0.601 | 0.0681 | 0.1363 |

Interpretation: the clearest physiological signal is posterior P300 loss-gain differences, which survive FDR correction in the full all-final-label dataset. This should still be framed as pilot evidence requiring replication.

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

Interpretation: risky-choice prediction remains modest. Previous-trial and rolling history features carry most of the signal; previous-trial EEG does not materially improve over history/value features in the current representation.

## Limitations

- Chronotype has only 39 participants, so findings are pilot-level.
- The raw behavioural chronotype column conflicts with `all final data.xlsx` for participants `1027` and `1036`; sensitivity analyses should disclose this.
- Compact ML evidence is significant in the full all-final-label dataset but not robust across all flagged-participant exclusions.
- There is no external validation cohort.
- Participant `1013` has an EEG/trigger alignment issue after block 10 that materially affects sensitivity results.
- MEQ/MCTQ values are not exported because their side-by-side workbook table order is not independently validated.
- Raw data are local and not committed; generated data/reports are ignored and summarized here.
