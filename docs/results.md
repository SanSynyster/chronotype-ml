# Current Results

This document summarizes the current clean-pipeline results. Generated report CSV/JSON files are ignored by git, so this file is the tracked public result summary.

## Chronotype

Participant-level task: predict `Morning` vs `Evening` chronotype from behavioral and ERP-derived features.

Dataset size: `39` participants.

Best feature-pack leaderboard entries:

| Pack | Model | Balanced Accuracy | Accuracy | Macro F1 | ROC AUC | Features |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `compact_combined` | Logistic Regression | 0.750 | 0.746 | 0.741 | 0.800 | 47 |
| `compact_combined` | Random Forest | 0.733 | 0.743 | 0.724 | 0.746 | 47 |
| `behavior_core` | Random Forest | 0.708 | 0.718 | 0.704 | 0.692 | 19 |
| `behavior_core` | Logistic Regression | 0.650 | 0.650 | 0.640 | 0.725 | 19 |
| `p300_core` | Random Forest | 0.600 | 0.593 | 0.568 | 0.658 | 16 |
| `frn_core` | Logistic Regression | 0.475 | 0.482 | 0.440 | 0.513 | 16 |

Permutation tests for Logistic Regression chronotype packs, 200 label permutations:

| Pack | Observed Balanced Accuracy | Null Mean | Null 95th Percentile | p-value |
| --- | ---: | ---: | ---: | ---: |
| `compact_combined` | 0.750 | 0.508 | 0.667 | 0.0149 |
| `behavior_core` | 0.650 | 0.504 | 0.667 | 0.0647 |
| `demo_only` | 0.592 | 0.498 | 0.667 | 0.2637 |
| `p300_core` | 0.575 | 0.504 | 0.642 | 0.2488 |
| `frn_core` | 0.475 | 0.504 | 0.658 | 0.6468 |

Top held-out permutation-importance features for `compact_combined` + Logistic Regression:

| Feature | Mean Balanced-Accuracy Drop |
| --- | ---: |
| `Fz_FRN_error_minus_correct` | 0.1055 |
| `POz_P300_loss_minus_gain` | 0.0880 |
| `FCz_FRN_error_minus_correct` | 0.0755 |
| `post_error_slowing` | 0.0747 |
| `rt_slope` | 0.0653 |
| `CPz_P300_error_minus_correct` | 0.0460 |
| `Fz_FRN_loss_error_minus_gain_correct` | 0.0430 |
| `risky_late_minus_early` | 0.0405 |
| `Pz_P300_loss_minus_gain` | 0.0395 |
| `free_rt_std` | 0.0380 |

Interpretation: the chronotype signal appears to be driven by a compact combination of behavioral adaptation and ERP contrast features, especially FRN error/correct contrasts, posterior P300 loss/gain contrasts, post-error slowing, and RT dynamics.

## Risky Choice

Trial-level task: predict binary risky choice using only same-trial pre-choice features plus previous-trial history. Evaluation uses participant-grouped cross-validation.

Dataset size: `10,626` free-choice trial rows.

Best feature-pack leaderboard entries:

| Pack | Model | Balanced Accuracy | Accuracy | Macro F1 | ROC AUC | Features |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `value_history` | Logistic Regression | 0.584 | 0.591 | 0.582 | 0.624 | 39 |
| `history_only` | Logistic Regression | 0.584 | 0.591 | 0.582 | 0.626 | 25 |
| `all_clean` | Logistic Regression | 0.583 | 0.590 | 0.581 | 0.620 | 50 |
| `prev_eeg` | Logistic Regression | 0.583 | 0.590 | 0.581 | 0.620 | 50 |
| `value_only` | Random Forest | 0.531 | 0.529 | 0.528 | 0.543 | 14 |

Interpretation: risky-choice prediction is modest. History features add more signal than current values alone. Previous-trial EEG did not improve performance over value/history features in the current representation.

## Limitations

- Chronotype has only 39 participants, so findings are pilot-level.
- The chronotype compact feature set still has 47 features, which is high relative to sample size.
- Permutation testing used 200 permutations for project-level validation; manuscript-level reporting should use more.
- There is no external validation cohort.
- EEG features are summarized as ERP mean/contrast features and may miss latency, peak, spectral, or trial-quality information.
