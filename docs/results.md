# Current Results

This document summarizes the current active raw-to-clean results. Generated CSV/JSON reports are ignored by git, so this file is the tracked public result summary.

## Provenance And Labels

The current pipeline rebuilds modelling tables from local raw files using active scripts:

- `scripts/link_raw_metadata.py` links `participant_summary.xlsx` to `UserID` by matching previous-feedback behavioural aggregates recomputed from `raw_behavioral_trials.xlsx`.
- `scripts/link_raw_metadata.py` then links `all final data.xlsx` through the shared `ERPset` column.
- `scripts/build_ml_ready_from_raw.py` builds the raw-derived trial table from trimmed behaviour (`Trial <= 23`) and raw EEG single-trial means/triggers.
- `scripts/build_clean_chronotype.py` uses previous-trial feedback for behavioral adaptation features and current-trial feedback for feedback-locked ERP contrasts.
- `scripts/rebuild_from_raw.py --execute` runs the full active rebuild and writes `docs/data_provenance.md`.

Primary chronotype labels come from `participant_summary.xlsx` / `all final data.xlsx`. The metadata link is complete for all `39` participants.

Label/QC snapshot:

| Item | Value |
| --- | ---: |
| Participants | 39 |
| Primary chronotype counts | 20 Morning / 19 Evening |
| Metadata links missing | 0 |
| Mean summary-to-behaviour match distance | 0.0147 |
| Max summary-to-behaviour match distance | 0.0687 |
| Raw-behaviour label conflicts | 2 participants: `1027`, `1036` |
| MEQ/MCTQ status | Not exported; side-by-side table order is unvalidated |

Raw-derived table snapshot:

| Table | Rows | Columns | Notes |
| --- | ---: | ---: | --- |
| `data/processed/ml_ready_features.csv` | 14,352 | 40 | 39 participants, behaviour trimmed to 368 rows each |
| `data/clean/risky_choice_prechoice.csv` | 10,669 | 55 | free-choice rows only |
| `data/clean/chronotype_participant.csv` | 39 | 173 | one row per participant |
| `data/clean/chronotype_compact_12.csv` | 39 | 14 | 12 features plus ID/target |

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
| `frn_core` | Random Forest | 0.667 | 0.664 | 0.651 | 0.692 | 16 |
| `p300_core` | Logistic Regression | 0.658 | 0.664 | 0.651 | 0.688 | 16 |
| `behavior_core` | Logistic Regression | 0.633 | 0.639 | 0.617 | 0.771 | 19 |

Interpretation: the high-dimensional Random Forest results are exploratory. The more defensible compact result is the 12-feature Logistic Regression model, because it uses a small theory-driven feature set relative to `n = 39`.

## Compact 12-Feature Chronotype Model

The journal-oriented compact model uses 12 theory-driven predictors. Repeated 5-fold stratified CV, 100 repeats, Logistic Regression:

| Metric | Mean | SD | 95% Interval |
| --- | ---: | ---: | ---: |
| Accuracy | 0.667 | 0.150 | 0.375-1.000 |
| Balanced accuracy | 0.666 | 0.150 | 0.375-1.000 |
| Macro F1 | 0.655 | 0.157 | 0.348-1.000 |
| ROC AUC | 0.706 | 0.171 | 0.375-1.000 |

1000-label permutation test, fixed 5-fold stratified CV, Logistic Regression:

| Observed balanced accuracy | Null mean | Null SD | Null 95th percentile | p-value |
| ---: | ---: | ---: | ---: | ---: |
| 0.692 | 0.496 | 0.103 | 0.667 | 0.0340 |

Sensitivity analyses for the compact 12-feature Logistic Regression model:

| Dataset | Rows | Repeated-CV Balanced Accuracy Mean | Permutation Observed BA | Permutation p-value |
| --- | ---: | ---: | ---: | ---: |
| Full primary dataset | 39 | 0.666 | 0.692 | 0.0340 |
| Exclude `1013` EEG/trigger QC case | 38 | 0.669 | 0.658 | 0.0529 |
| Exclude label conflicts `1027`, `1036` | 37 | 0.634 | 0.533 | 0.3816 |
| Exclude all flagged participants | 36 | 0.639 | 0.675 | 0.0669 |

Interpretation: the compact model is significant in the full primary dataset but sensitivity-dependent. The result should be framed as promising pilot evidence, not as a robust validated classifier.

## Feature Importance

Top held-out permutation-importance features for `compact_combined` + Logistic Regression after correcting behavioral adaptation features:

| Feature | Mean Balanced-Accuracy Drop |
| --- | ---: |
| `Fz_FRN_error_minus_correct` | 0.0408 |
| `Fz_FRN_loss_error_minus_gain_correct` | 0.0267 |
| `POz_P300_loss_minus_gain` | 0.0217 |
| `gain_error_rt_mean` | 0.0183 |
| `CPz_P300_mean` | 0.0117 |
| `parietal_eeg_mean` | 0.0100 |
| `FCz_FRN_loss_error_minus_gain_correct` | 0.0100 |

Interpretation: importance is unstable because fold test sets are small, but the dominant signal remains physiologically plausible: FRN error/feedback contrasts and posterior P300 features.

## Classical Group Statistics

Morning-vs-Evening tests for theory-driven features show the largest effects in posterior P300 loss-gain contrasts and risky-choice rates after previous feedback:

| Feature | Evening Mean | Morning Mean | Cohen's d | Welch p | FDR p |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Pz_P300_loss_minus_gain` | -0.963 | 0.308 | -1.045 | 0.0028 | 0.0341 |
| `POz_P300_loss_minus_gain` | -0.465 | 0.553 | -0.919 | 0.0076 | 0.0454 |
| `loss_error_risky_rate` | 0.634 | 0.530 | 0.813 | 0.0160 | 0.0547 |
| `free_risky_rate` | 0.583 | 0.477 | 0.797 | 0.0182 | 0.0547 |
| `gain_correct_risky_rate` | 0.552 | 0.423 | 0.765 | 0.0231 | 0.0553 |
| `Fz_FRN_error_minus_correct` | -2.895 | -2.023 | -0.601 | 0.0681 | 0.1363 |

Sensitivity: P300 effects remain directionally large after excluding flagged participants, but FDR significance is sample-sensitive. After excluding all flagged participants, `Pz_P300_loss_minus_gain` has FDR p `0.0531` and `POz_P300_loss_minus_gain` has FDR p `0.0614`.

Interpretation: the classical statistics provide the clearest physiological signal, especially for P300 loss-gain contrasts. These should be emphasized alongside the sensitivity-dependent ML results.

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
- The strongest chronotype leaderboard results use too many features for the sample size and should be treated as exploratory.
- The compact model has wide repeated-CV uncertainty and sensitivity-dependent permutation support.
- There is no external validation cohort.
- Participant `1013` has an EEG/trigger alignment issue after block 10 that should be disclosed or handled in sensitivity analyses.
- Primary metadata labels conflict with raw behavioural labels for participants `1027` and `1036`.
- MEQ/MCTQ values are not exported because their side-by-side workbook table order is not independently validated.
- Raw data are local and not committed; generated data/reports are ignored and summarized here.
