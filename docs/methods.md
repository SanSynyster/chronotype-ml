# Methods Draft

## Study Framing

The primary analysis predicts participant chronotype, Morning vs Evening, from participant-level summaries of risky decision-making behavior and feedback-locked ERP contrasts. Trial-level risky-choice prediction is treated as a secondary analysis.

Primary chronotype labels are taken from `participant_summary.xlsx` / `all final data.xlsx`. Because those workbooks do not contain explicit `UserID` values, `scripts/link_raw_metadata.py` maps them to participants by matching previous-feedback behavioural aggregates recomputed from `raw_behavioral_trials.xlsx`; `all final data.xlsx` is then linked through the shared `ERPset` column. Raw behavioural chronotype labels are retained for QC, not used as the primary target.

## Feature Engineering

The active raw-to-clean path trims behaviour to `Trial <= 23`, yielding 368 behavioural rows per participant to match the EEG single-trial exports. EEG single-trial means are pivoted to trial-level `channel_window` columns and merged by participant/global trial index. Trigger/behaviour valence agreement is reported as QC rather than used to silently drop rows. MEQ/MCTQ fields are not exported because their side-by-side workbook table order has not been independently validated.

Chronotype features are aggregated to one row per participant. The most defensible compact set is limited to behavioral adaptation and ERP contrasts:

- post-error slowing
- RT slope across task progression
- risky-choice late-minus-early change
- risky-choice rates after gain-correct and loss-error feedback
- frontocentral FRN error-vs-correct contrasts
- FRN loss-error minus gain-correct contrast
- posterior/parietal P300 loss-vs-gain contrasts

Behavioral adaptation features use previous-trial feedback labels. Feedback-locked ERP condition contrasts use the current trial feedback label.

Risky-choice features use only same-trial pre-choice values plus previous-trial and rolling history. Same-trial outcome, correctness, score, feedback, and feedback-locked EEG are excluded as predictors.

## Validation

Chronotype models are evaluated with stratified cross-validation, repeated cross-validation, and label-permutation testing. Risky-choice models use participant-grouped cross-validation to test generalization to held-out participants.

The manuscript-oriented compact chronotype model uses 12 theory-driven predictors to reduce overfitting risk relative to the broader 47-feature pilot model.

Sensitivity analyses exclude participant `1013`, participants with raw-behaviour/metadata label conflicts (`1027`, `1036`), and all flagged participants together.

## Statistical Reporting

Classical Morning-vs-Evening group comparisons are reported for theory-driven features with effect sizes and FDR-adjusted p-values.

For manuscript reporting, model performance should emphasize repeated-CV mean and uncertainty, permutation-test p-values, and classical effect sizes rather than a single best split.
