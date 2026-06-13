# Methods Draft

## Study Framing

The primary analysis predicts participant chronotype, Morning vs Evening, from participant-level summaries of risky decision-making behavior and feedback-locked ERP contrasts. Trial-level risky-choice prediction is treated as a secondary analysis.

Primary chronotype labels are taken from `all final data.xlsx`, linked through the shared `ERPset` column after mapping `participant_summary.xlsx` rows to `UserID`. Because the metadata workbooks do not contain explicit `UserID` values, `scripts/link_raw_metadata.py` maps them to participants by matching previous-feedback behavioural aggregates recomputed from `raw_behavioral_trials.xlsx`. The raw behavioural `Chronotype` column conflicts with `all final data.xlsx` for participants `1027` and `1036`; these are retained under the `all final data.xlsx` labels and handled in sensitivity analyses.

The binary labels were validated against the continuous MEQ score
(`scripts/validate_meq_labels.py`). The MEQ/MCTQ scores live in a table block of
`all final data.xlsx` that is not row-aligned with the chronotype column, but
every block is keyed by participant name (the `ERPset` field), so the MEQ score
was attached by name rather than row position. The score separated the labels in
the standard Horne-Ostberg direction (Evening MEQ mean 37.3, range 25-49; Morning
mean 57.7, range 45-64), and all 26 participants with a decisive MEQ score
(outside the 42-58 intermediate band) matched their binary label. Both
label-conflict participants had decisive MEQ scores confirming the primary label
(1027 = 61, Morning; 1036 = 27, Evening), so the raw-behaviour column was in error
for these two rather than the metadata. This also resolves the earlier concern
that the side-by-side MEQ/MCTQ table order was unvalidated: alignment by name key
is well defined and was used here.

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

The study uses two mutually-validating primary analyses of the same participant-level features. The first is an interpretable, leakage-aware machine-learning framework that classifies chronotype (Morning vs Evening) from combined behavioural and feedback-locked ERP features. The second is a classical Morning-vs-Evening group comparison of the same theory-driven features, with the posterior P300 loss-minus-gain contrast as the pre-specified neural hypothesis. The two are interpreted jointly: the machine-learning model's feature importance and coefficients are expected to converge with the univariate group statistics, providing cross-validating multivariate and univariate evidence for the same neural signal. Trial-level risky-choice prediction is a secondary task.

## Statistical Reporting

Classical Morning-vs-Evening group comparisons are reported for theory-driven features with Cohen's d, Welch t-test p-values, nonparametric Mann-Whitney p-values, and Benjamini-Hochberg FDR-adjusted p-values across the theory-driven feature family.

To avoid dichotomizing chronotype, the posterior P300 contrasts were additionally related to the continuous MEQ score with Pearson correlations (percentile-bootstrap 95% CIs), Spearman correlations, and OLS slopes (`scripts/meq_p300_continuous.py`). This continuous analysis is interpreted alongside its power: with n = 38 the minimum correlation detectable at 80% power is approximately r = 0.44.

The metadata-to-participant linkage uses optimal one-to-one assignment with reported match-distance and match-margin QC.

## Machine-Learning Analysis

Chronotype was classified at the participant level (one example per participant) from the theory-driven 12-feature set (`compact_12`: behavioural adaptation measures and frontocentral/posterior ERP contrasts). A broader 47-feature set (`compact_combined`) and 171-feature literature set (`all_literature`) were retained as high-dimensionality exploratory comparisons.

**Pipeline.** All preprocessing was encapsulated in a scikit-learn `Pipeline` and fit only on training folds, eliminating preprocessing leakage. Numeric features were median-imputed and standardized; any categorical features were most-frequent-imputed and one-hot encoded. Class imbalance (20 Morning / 19 Evening) was handled with class weighting and balanced accuracy as the primary metric.

**Models compared.** Five classifiers were evaluated on identical folds: L2- and L1-regularized logistic regression, random forest, an RBF-kernel support vector machine, and histogram gradient boosting. The L2 logistic regression was pre-specified as the primary model for its interpretability and suitability for small samples.

**Nested cross-validation.** Generalization was estimated with nested cross-validation: an outer repeated stratified 5-fold loop (10 repeats) estimated out-of-fold performance, while an inner stratified 3-fold `GridSearchCV` tuned each model's hyperparameters (logistic-regression `C`; random-forest depth/leaf/trees; SVM `C`/`gamma`; boosting learning-rate/leaves) using balanced accuracy. Because tuning occurred only within inner training folds, no test fold informed model selection. Performance is reported as the mean and standard deviation of balanced accuracy, accuracy, ROC AUC, sensitivity (Morning), specificity (Evening), and macro F1 across outer folds. Out-of-fold predicted probabilities (averaged over repeats) were used for the ROC curve and confusion matrix.

**Significance and multiplicity.** The tuned primary model was tested against a label-permutation null (shuffled labels, full nested pipeline re-fit each permutation). Separately, the five pre-specified literature feature packs were each permutation-tested and the p-values FDR-corrected as a family; single best-split numbers were used only for ranking. The primary `compact_12` model is reported with its own pre-specified permutation p-value; the pack-family FDR result is reported to bound optimism from comparing feature sets.

**Interpretability.** The tuned primary model was refit on the full sample and its standardized coefficients reported, and held-out permutation importance was computed by cross-validated feature shuffling, to test whether the multivariate model relied on the same features identified by the univariate group comparison.

**Robustness.** The full nested-CV analysis and the permutation test were repeated under the participant-exclusion sensitivity scenarios (exclude 1013; exclude label conflicts 1027/1036; exclude all three).
