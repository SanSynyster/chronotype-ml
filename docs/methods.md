# Methods Draft

## Study Framing

The primary analysis predicts participant chronotype, Morning vs Evening, from participant-level summaries of risky decision-making behavior and feedback-locked ERP contrasts. Trial-level risky-choice prediction is treated as a secondary analysis.

Primary chronotype labels are taken from `all final data.xlsx`, linked through the shared `ERPset` column after mapping `participant_summary.xlsx` rows to `UserID`. Because the metadata workbooks do not contain explicit `UserID` values, `scripts/link_raw_metadata.py` maps them to participants by matching previous-feedback behavioural aggregates recomputed from `raw_behavioral_trials.xlsx`. The raw behavioural `Chronotype` column conflicts with `all final data.xlsx` for participants `1027` and `1036`; these are retained under the `all final data.xlsx` labels and handled in sensitivity analyses.

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

The manuscript-oriented compact chronotype model uses 12 theory-driven predictors to reduce overfitting risk relative to the broader 47-feature exploratory model.

A second exploratory performance-informed compact model uses 12 features selected from recurrent signals in the current dataset's feature-pack performance, group statistics, and held-out importance. This model is kept separate from the a priori compact model because its feature set is partially data-driven.

Sensitivity analyses exclude participant `1013`, who has an EEG/trigger QC issue, the two raw-behaviour/metadata label-conflict participants `1027` and `1036`, and all three flagged participants together.

## Sample and Power

The analysed sample is 39 participants (19 Evening, 20 Morning). For a two-group comparison with this allocation, the minimum effect detectable with 80% power at a two-sided alpha of 0.05 is Cohen's d of approximately 0.90 (normal approximation, d = (z_{1-alpha/2} + z_{1-beta}) * sqrt(1/n1 + 1/n2)). The study is therefore powered to detect only large between-group effects: estimated power is about 0.90 for the observed posterior P300 effect (d approximately 1.0) but only about 0.35 for a medium effect (d = 0.5). Null or weak results for medium-sized effects, including most behavioral and FRN contrasts, are consequently inconclusive rather than evidence of absence.

## Analysis Hierarchy

The primary analysis is the classical Morning-vs-Evening group comparison of theory-driven behavioral and ERP features, with the posterior P300 loss-minus-gain contrast as the pre-specified neural hypothesis. Machine-learning classification of chronotype is a secondary, exploratory analysis reported as converging evidence. Trial-level risky-choice prediction is a secondary task.

## Statistical Reporting

Classical Morning-vs-Evening group comparisons are reported for theory-driven features with Cohen's d, Welch t-test p-values, nonparametric Mann-Whitney p-values, and Benjamini-Hochberg FDR-adjusted p-values across the theory-driven feature family.

For the exploratory classifier, model performance is reported as repeated-CV mean and uncertainty (95% interval) and label-permutation p-values. Permutation p-values across the five pre-specified literature feature packs are FDR-corrected as a family; the theory-driven `compact_12` model is reported separately as a single pre-specified classifier with its uncorrected permutation p-value. Single best-split numbers are used only for ranking and are not reported as primary evidence. The metadata-to-participant linkage uses optimal one-to-one assignment with reported match-distance and match-margin QC.
