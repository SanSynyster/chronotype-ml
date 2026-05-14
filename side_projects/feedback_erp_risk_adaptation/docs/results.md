# Results

## Initial Dataset Build

The first implemented target is `next_free_choice_risky`: for each current feedback trial `t`, predict whether the participant's next free-choice trial is risky.

Source input:

```text
data/processed/ml_ready_features.csv
```

Generated side-project tables are written under:

```text
side_projects/feedback_erp_risk_adaptation/data/
```

Dataset snapshot:

| Item | Value |
| --- | ---: |
| Rows | 14,313 |
| Participants | 39 |
| Target `0` count | 6,681 |
| Target `1` count | 7,632 |
| Current free-choice rows | 10,630 |
| Current forced-choice rows | 3,683 |
| Median gap to next free-choice trial | 1 trial |
| Max gap to next free-choice trial | 6 trials |

Feature packs:

| Pack | Columns | Notes |
| --- | ---: | --- |
| `context_only` | 30 | task context, value context, feedback condition |
| `history_only` | 33 | current and previous behavioral state/history |
| `eeg_only` | 16 | feedback-locked FRN/P300 features only |
| `context_eeg` | 41 | context plus ERP features |
| `history_context` | 58 | strongest non-ERP baseline |
| `history_context_eeg` | 69 | full first-pass model |

Next-free-trial index/gap metadata are kept only in the full audit table, not in modeling feature-pack CSVs.

## Preliminary Grouped-CV Baselines

These are first-pass baselines using participant-grouped 5-fold CV. They are not final journal-level results and have not yet been permutation-tested.

| Feature pack | Best model | Balanced accuracy | ROC AUC | Interpretation |
| --- | --- | ---: | ---: | --- |
| `context_only` | Logistic Regression | 0.541 | 0.547 | weak context signal |
| `history_only` | Logistic Regression | 0.577 | 0.610 | behavioral history carries most signal |
| `eeg_only` | Random Forest | 0.502 | 0.498 | feedback ERP alone is near chance |
| `context_eeg` | Logistic Regression | 0.531 | 0.535 | ERP does not improve context-only in first pass |
| `history_context` | Logistic Regression | 0.580 | 0.612 | strongest first-pass baseline |
| `history_context_eeg` | Logistic Regression | 0.576 | 0.607 | adding ERP does not improve over history/context |

Initial interpretation:

The `next_free_choice_risky` target is scientifically valid and leakage-aware, but first-pass prediction is modest. Behavioral history and task context explain more than current feedback-locked ERP features. The next journal-level step is to test whether a more specific target, such as risk persistence after negative feedback, shows stronger ERP contribution.

## Next Analyses

## Negative-Feedback Persistence Subsets

The builder now creates four negative-feedback scenario datasets using the same `next_free_choice_risky` target.

| Scenario | Rows | Participants | Target `0` | Target `1` | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `loss_error` | 5,488 | 39 | 2,237 | 3,251 | strict negative feedback |
| `all_loss` | 7,050 | 39 | 3,044 | 4,006 | all loss feedback |
| `all_error` | 7,179 | 39 | 3,160 | 4,019 | all error feedback |
| `loss_or_error` | 8,741 | 39 | 3,967 | 4,774 | broad negative/unfavorable feedback |

## Negative-Feedback Grouped-CV Baselines

Participant-grouped 5-fold CV, Logistic Regression unless noted.

| Scenario | Feature pack | Best model | Balanced accuracy | ROC AUC | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| `loss_or_error` | `history_context` | Logistic Regression | 0.582 | 0.614 | strongest negative-feedback baseline |
| `loss_or_error` | `history_context_eeg` | Logistic Regression | 0.576 | 0.610 | adding ERP did not improve performance |
| `loss_or_error` | `eeg_only` | Logistic Regression | 0.510 | 0.509 | ERP alone near chance |
| `loss_error` | `history_context` | Logistic Regression | 0.574 | 0.605 | modest strict negative-feedback signal |
| `loss_error` | `history_context_eeg` | Logistic Regression | 0.573 | 0.603 | adding ERP did not improve performance |
| `loss_error` | `eeg_only` | Random Forest | 0.508 | 0.504 | ERP alone near chance |
| `all_loss` | `history_context` | Logistic Regression | 0.568 | 0.595 | modest loss-specific signal |
| `all_loss` | `history_context_eeg` | Logistic Regression | 0.564 | 0.590 | adding ERP did not improve performance |
| `all_error` | `history_context` | Logistic Regression | 0.574 | 0.614 | modest error-specific signal |
| `all_error` | `history_context_eeg` | Logistic Regression | 0.574 | 0.612 | ERP is effectively neutral |

## Permutation Screening

The strongest negative-feedback Logistic Regression models were screened with 200 label permutations using participant-grouped CV.

| Scenario | Feature pack | Observed BA | Null mean | Null 95th percentile | p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| `loss_or_error` | `history_context` | 0.582 | 0.500 | 0.513 | 0.0050 |
| `loss_or_error` | `history_context_eeg` | 0.576 | 0.500 | 0.513 | 0.0050 |
| `all_error` | `history_context` | 0.574 | 0.500 | 0.512 | 0.0050 |
| `all_error` | `history_context_eeg` | 0.574 | 0.500 | 0.512 | 0.0050 |

Interpretation:

The negative-feedback models are reliably above permutation chance, but the current evidence does not support an ERP-increment claim. Behavioral history and task context explain the signal; feedback-locked ERP features are near chance alone and do not improve the best first-pass models.

## Updated Next Analyses

- Run 1000-permutation confirmation only if this target remains central.
- Add an explicit incremental-delta permutation test: `history_context_eeg - history_context`.
- Build stricter feature packs that remove current-choice variables (`CurrentRisky`, `ChoiceMade`, `CorrectChoice`) to test whether models rely too heavily on immediate behavior.
- Run sensitivity excluding participant `1013`.
- Consider reframing the side project around behavioral prediction of risk persistence, with ERP as a tested but currently unsupported incremental predictor.

## Revised Strict-Pack Analyses

The builder was extended with stricter lag-history feature packs, participant-centered ERP features, and adaptation-specific targets. Alternative target CSVs are written separately so unused outcome columns cannot leak into model predictors.

New feature packs:

| Pack | Features | Rationale |
| --- | --- | --- |
| `lag_history_context` | task/value context, current feedback, previous-trial variables, rolling history | removes immediate current-choice/state variables |
| `lag_history_context_eeg_raw` | `lag_history_context` plus raw FRN/P300 amplitudes | tests raw ERP increment |
| `lag_history_context_eeg_centered` | `lag_history_context` plus participant-z-scored ERP and regional FRN/P300 summaries | tests within-person ERP increment |
| `eeg_centered_only` | participant-centered/z-scored ERP features only | ERP-only sanity check after scale normalization |

New targets:

| Target | Rows in full dataset | Interpretation |
| --- | ---: | --- |
| `next_free_choice_risky` | 14,313 | broad next risky-choice prediction |
| `risk_switch` | 14,313 | next free choice differs from current choice |
| `risk_persistence_given_current_risky` | 7,252 | risky choice persists after current risky trial |
| `risk_initiation_given_current_safe` | 7,061 | risky choice starts after current safe trial |

## Revised Grouped-CV Screen

Focused Logistic Regression screen using participant-grouped 5-fold CV. The table reports the strongest all-participant revised models.

| Scenario | Target | Pack | Rows | BA | ROC AUC | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `all_error` | `risk_initiation_given_current_safe` | `lag_history_context` | 3,548 | 0.585 | 0.620 | strongest revised model; no ERP needed |
| `loss_error` | `risk_initiation_given_current_safe` | `lag_history_context_eeg_raw` | 1,877 | 0.584 | 0.612 | raw ERP slightly improves BA over strict baseline, but sample is smaller |
| `all_error` | `risk_initiation_given_current_safe` | `lag_history_context_eeg_raw` | 3,548 | 0.584 | 0.616 | near strict baseline |
| `all_error` | `risk_initiation_given_current_safe` | `lag_history_context_eeg_centered` | 3,548 | 0.583 | 0.622 | slightly lower BA but highest AUC among top revised models |
| `loss_error` | `risk_initiation_given_current_safe` | `lag_history_context` | 1,877 | 0.578 | 0.605 | strict negative-feedback initiation baseline |
| `loss_or_error` | `next_free_choice_risky` | `lag_history_context` | 8,741 | 0.577 | 0.609 | broad negative-feedback model remains modest |

Sensitivity excluding participant `1013` did not remove the signal. The best revised sensitivity result was `all_error` / `risk_initiation_given_current_safe` / `lag_history_context_eeg_centered`, with BA 0.585 and ROC AUC 0.622 across 38 participants.

## ERP Increment Check

Across revised all-participant screens, ERP increments were small and inconsistent.

| Scenario | Target | ERP comparison | Delta BA | Delta ROC AUC |
| --- | --- | --- | ---: | ---: |
| `loss_or_error` | `risk_initiation_given_current_safe` | raw ERP minus strict baseline | +0.010 | +0.001 |
| `loss_or_error` | `risk_initiation_given_current_safe` | centered ERP minus strict baseline | +0.007 | +0.001 |
| `loss_error` | `risk_initiation_given_current_safe` | raw ERP minus strict baseline | +0.006 | +0.007 |
| `all_error` | `next_free_choice_risky` | centered ERP minus strict baseline | +0.006 | +0.001 |
| `all_error` | `risk_initiation_given_current_safe` | centered ERP minus strict baseline | -0.002 | +0.002 |
| `loss_or_error` | `next_free_choice_risky` | raw ERP minus strict baseline | -0.005 | -0.004 |

Interpretation:

The revised analyses strengthen the behavioral adaptation framing. Risk initiation after error feedback is predictably above chance with participant-grouped validation, but the evidence still does not support a robust ERP-increment claim. ERP features are scientifically relevant and worth reporting as tested, but current predictive signal remains mainly behavioral history and task context.

## Revised Permutation Screening

The strongest revised all-participant model and its centered-ERP counterpart were screened with 200 label permutations.

| Scenario | Target | Pack | Observed BA | Null mean | Null 95th percentile | p-value |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `all_error` | `risk_initiation_given_current_safe` | `lag_history_context` | 0.585 | 0.500 | 0.516 | 0.0050 |
| `all_error` | `risk_initiation_given_current_safe` | `lag_history_context_eeg_centered` | 0.583 | 0.500 | 0.516 | 0.0050 |

Current conclusion:

The side project is best framed as leakage-aware prediction of post-error risk initiation/adaptation, with ERP features included as a negative or weak-increment test rather than the primary effect.
