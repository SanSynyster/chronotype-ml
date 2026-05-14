# Methods Draft

This side project predicts subsequent risk adaptation from feedback-locked ERP activity.

The first implemented target is `next_free_choice_risky`: for each current feedback trial `t`, the target is whether the participant's next free-choice trial is risky. Predictors are restricted to trial `t` or earlier. No next-trial variables are used except the target and index/gap metadata.

Validation must use participant-grouped cross-validation because rows are trial-level but independence is participant-level.

The builder writes first-pass modeling feature packs: `context_only`, `history_only`, `eeg_only`, `eeg_centered_only`, `context_eeg`, `history_context`, `history_context_eeg`, `lag_history_context`, `lag_history_context_eeg_raw`, and `lag_history_context_eeg_centered`. Next-free-trial index/gap metadata are kept only in the full audit table and are excluded from modeling packs.

The strict lag-history packs remove immediate current-choice/state variables (`CurrentRisky`, `ChoiceMade`, `CorrectChoice`, current response time, current score, and current score delta). This tests whether models remain predictive without relying on the participant's just-completed choice. These packs keep task/value context, current feedback condition, previous-trial variables, and rolling behavioral history computed only from prior trials.

Centered ERP features are computed within participant from feedback-locked raw channel/window amplitudes. The builder adds participant-centered and participant-z-scored ERP features, plus regional summaries for frontocentral FRN (`Fz`, `FCz`, `Cz`) and posterior P300 (`Pz`, `POz`, `CPz`). It also adds FRN/P300 interactions with strict `Loss-Error` feedback. The centered ERP packs allow ERP effects to be tested after reducing between-person amplitude scale differences.

Alternative adaptation targets are written to separate target-specific CSV files so unused outcome columns cannot leak into model predictors:

| Target | Definition |
| --- | --- |
| `next_free_choice_risky` | next free-choice trial is risky |
| `risk_switch` | next free-choice risk label differs from current trial risk label |
| `risk_persistence_given_current_risky` | among currently risky trials, next free-choice trial is also risky |
| `risk_initiation_given_current_safe` | among currently safe trials, next free-choice trial is risky |

Negative-feedback persistence is implemented as scenario-specific subsets of the same target:

| Scenario | Definition |
| --- | --- |
| `loss_error` | current feedback is `Loss-Error` |
| `all_loss` | current feedback is any loss condition |
| `all_error` | current feedback is any error condition |
| `loss_or_error` | current feedback is either loss or error |

These subsets test whether subsequent risky-choice persistence is more predictable after unfavorable feedback than in the full task.

All model evaluation reported here uses participant-grouped 5-fold cross-validation. Revised side-project screens focus on Logistic Regression because it was the strongest and most stable first-pass model family for these feature packs.
