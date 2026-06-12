# Machine-Learning Reporting Checklist

A leakage-and-validity checklist for the predictive analyses, with how this study
addresses each item. Intended for the supplement. Items follow common
recommendations for machine learning in neuroscience/neuroimaging and for
avoiding data leakage and over-optimistic reporting.

| # | Item | Status | How it is addressed |
| --- | --- | --- | --- |
| 1 | Preprocessing fit inside CV folds | Met | Imputation and scaling are inside scikit-learn `Pipeline`s fit per fold (`train_clean_baseline.py`, `repeated_cv_clean.py`, `permutation_test_clean.py`). |
| 2 | No same-trial / target leakage in features | Met | Same-trial outcome, correctness, score, feedback, and feedback-locked EEG are excluded from risky-choice predictors (`build_clean_risky_choice.py`). |
| 3 | Generalization to held-out participants | Met | Trial-level risky choice uses `GroupKFold` on `participant_id`. |
| 4 | Uncertainty, not single split | Met | Repeated stratified CV (100 repeats) with mean and 95% interval (`repeated_cv_clean.py`); single best split used only for ranking. |
| 5 | Chance level established empirically | Met | 1000-iteration label-permutation tests (`permutation_test_clean.py`). |
| 6 | Correction for multiple models/feature sets | Met | Benjamini-Hochberg FDR across the pre-specified feature-pack family (`run_chronotype_permutation_tests.py`); single pre-specified primary classifier reported separately. |
| 7 | Primary vs exploratory analyses declared | Met | Neural group comparison is primary; classification is secondary/exploratory (`docs/methods.md`, `docs/manuscript_draft.md`). |
| 8 | Feature-to-sample ratio disclosed | Met | High-dimensional (47-171 feature) models are flagged exploratory for n = 39; primary classifier is 12 features. |
| 9 | Hyperparameter tuning leakage | Met (by design) | No data-driven hyperparameter search; fixed default models, so there is no inner-tuning leakage to control. Stated explicitly in Methods. |
| 10 | Class imbalance handled honestly | Met | Balanced accuracy is the primary metric; `class_weight="balanced"`; near-balanced classes (20/19). |
| 11 | Robustness / sensitivity analyses | Met | Participant-exclusion sensitivity matrix for classifier and group effect (`sensitivity_matrix.py`). |
| 12 | Effect sizes with uncertainty | Met | Cohen's d with bootstrap 95% CIs and Hedges g (`group_stats_chronotype.py`). |
| 13 | Naive baselines for context | Met | Majority, persistence, and participant-mean-oracle baselines for risky choice (`risky_choice_baseline.py`). |
| 14 | Statistical power reported | Met | Minimum detectable effect and power at relevant effect sizes (`docs/methods.md`). |
| 15 | Reproducible environment and pipeline | Met | Pinned `requirements.txt` + `requirements.lock.txt`; one-command rebuild (`rebuild_from_raw.py --execute`) and figure regeneration (`make_figures.py`). |
| 16 | Data provenance documented | Met | Optimal one-to-one metadata linkage with QC; provenance record (`docs/data_provenance.md`). |
| 17 | External validation | Not met | No external cohort; disclosed as a limitation. Single-cohort result requiring replication. |
| 18 | Ground-truth labels | Met | Labels are MEQ-derived and validated against the continuous MEQ score by name-key alignment (26/26 decisive participants consistent; both raw-behaviour conflict cases MEQ-confirmed; `validate_meq_labels.py`). |
