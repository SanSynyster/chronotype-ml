# Chronotype ML: EEG and Behavioral Decision Modelling

This repository contains a leakage-aware machine-learning workflow for EEG and behavioral decision-making data. The project studies two related questions:

- Can chronotype, Morning vs Evening, be predicted from participant-level behavioral and ERP features?
- Can trial-level risky choice be predicted from pre-choice values and previous-trial history?

The current codebase is organized around clean, reproducible modelling scripts. Older exploratory scripts are archived under `archive/legacy/scripts/`.

## Key Findings

- The strongest chronotype model is a compact, literature-guided feature pack using behavior, FRN/P300 contrasts, and RT dynamics.
- Chronotype `compact_combined` with Logistic Regression reached balanced accuracy `0.750` with permutation-test `p = 0.0149` using 200 label permutations.
- The most important chronotype drivers were `Fz_FRN_error_minus_correct`, `POz_P300_loss_minus_gain`, `FCz_FRN_error_minus_correct`, `post_error_slowing`, and `rt_slope`.
- Risky-choice prediction was modest: the best clean pack, `value_history` with Logistic Regression, reached balanced accuracy `0.584` and ROC AUC `0.624` under participant-grouped CV.
- Same-trial leakage features such as `ChoiceMade`, `CorrectChoice`, `CurrentScore`, same-trial feedback, and same-trial feedback-locked EEG are excluded from risky-choice predictors.

See `docs/results.md` for result tables and interpretation.

## Repository Layout

```text
chronotype_ml/
├── archive/legacy/scripts/        # old exploratory scripts, kept for reference
├── data/raw/                      # local raw data, ignored by git
├── data/clean/                    # generated clean datasets, ignored by git
├── docs/results.md                # tracked summary of current results
├── reports/clean/                 # generated evaluation reports, ignored by git
├── scripts/                       # active clean pipeline
└── requirements.txt
```

Generated data, reports, and model artifacts are intentionally ignored. Raw data is not committed.

## Active Scripts

- `scripts/build_clean_risky_choice.py`: builds leakage-aware trial-level risky-choice datasets.
- `scripts/build_clean_chronotype.py`: builds participant-level chronotype datasets and literature-guided feature packs.
- `scripts/train_clean_baseline.py`: evaluates Logistic Regression, Random Forest, and HGB baselines.
- `scripts/run_literature_feature_packs.py`: builds/evaluates all literature-guided feature packs and creates a leaderboard.
- `scripts/permutation_test_clean.py`: permutation-tests classifier performance against shuffled labels.
- `scripts/run_chronotype_permutation_tests.py`: batch permutation tests for chronotype packs.
- `scripts/feature_importance_clean.py`: cross-validated held-out permutation importance.
- `scripts/run_chronotype_importance.py`: focused importance analysis for the best chronotype model.

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

This project was run with the local virtual environment at `env/`.

## Workflow

Build clean datasets:

```bash
python scripts/build_clean_risky_choice.py --include-prev-eeg --write-packs
python scripts/build_clean_chronotype.py --write-packs
```

Evaluate literature-guided feature packs:

```bash
python scripts/run_literature_feature_packs.py
```

Permutation-test chronotype packs:

```bash
python scripts/run_chronotype_permutation_tests.py --permutations 200
```

Analyze feature importance for the significant chronotype model:

```bash
python scripts/run_chronotype_importance.py --repeats 50
```

## Feature Packs

Risky-choice packs:

- `value_only`: same-trial pre-choice value/context features only.
- `history_only`: previous-trial and rolling history features only.
- `value_history`: current value features plus previous-trial history.
- `prev_eeg`: `value_history` plus previous-trial EEG lags.
- `all_clean`: all leakage-screened risky-choice features.

Chronotype packs:

- `demo_only`: age and gender.
- `behavior_core`: risky-choice rates, RT dynamics, post-error slowing, early/late adaptation.
- `frn_core`: frontocentral FRN summaries and condition contrasts.
- `p300_core`: parietal/posterior P300 summaries and condition contrasts.
- `compact_combined`: compact behavior + FRN + P300 feature set.
- `all_literature`: all generated literature-guided participant features.

## Modelling Assumptions

- Chronotype is modelled at the participant level, one row per participant.
- Risky choice is modelled at the trial level with participant-grouped CV.
- Risky-choice predictors only use same-trial pre-choice information plus previous-trial history.
- Feedback-locked EEG from the same trial is not used to predict the same trial's choice.
- Chronotype results are interpreted cautiously because the dataset has only 39 participants.

## Current Limitations

- Small participant sample, `n = 39`, limits generalizability.
- The current clean builders depend on a processed local feature table produced during earlier exploration; raw-to-clean reconstruction should be rebuilt as a future cleanup.
- Report CSV/JSON outputs are ignored and summarized in `docs/results.md` instead of committed directly.
- No external validation cohort is available.
- Feature importance is unstable because chronotype fold test sets are small.

## Recommended Next Work

- Rebuild the raw-to-clean preprocessing path as active clean scripts.
- Increase permutation tests to 1,000+ for manuscript-level reporting.
- Add repeated/nested CV and confidence intervals.
- Reduce chronotype compact features further and test a 5-15 feature model.
- Validate on more participants or an external cohort.

## Status

This is suitable as a public research/portfolio repository. Treat the chronotype result as a promising pilot finding, not a deployed or clinically validated predictor.
