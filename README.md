# Chronotype ML: EEG and Behavioral Decision Modelling

This repository contains a leakage-aware machine-learning workflow for EEG and behavioral decision-making data. The project studies two related questions:

- Can chronotype, Morning vs Evening, be predicted from participant-level behavioral and ERP features?
- Can trial-level risky choice be predicted from pre-choice values and previous-trial history?

The current codebase is organized around clean, reproducible modelling scripts. Older exploratory scripts are archived under `archive/legacy/scripts/`.

## Key Findings

- The active pipeline now rebuilds from local raw behaviour, EEG single-trial means/triggers, and linked participant metadata.
- Primary chronotype labels come from `participant_summary.xlsx` / `all final data.xlsx`; raw behavioural labels are retained for QC.
- The 12-feature compact Logistic Regression model reached repeated-CV balanced accuracy mean `0.666` and 1000-permutation `p = 0.0340` in the full primary dataset, but sensitivity analyses weaken this evidence.
- The clearest physiological signal is in posterior P300 loss-gain contrasts; FRN feedback/error contrasts also contribute to exploratory models.
- Risky-choice prediction was modest: the best clean Logistic Regression packs reached balanced accuracy about `0.587` and ROC AUC about `0.62` under participant-grouped CV.
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
- `scripts/link_raw_metadata.py`: links raw participant summary/final metadata to `UserID`.
- `scripts/build_ml_ready_from_raw.py`: builds the raw-derived ML-ready trial table.
- `scripts/build_chronotype_sensitivity.py`: builds chronotype datasets excluding flagged participants.
- `scripts/train_clean_baseline.py`: evaluates Logistic Regression, Random Forest, and HGB baselines.
- `scripts/run_literature_feature_packs.py`: builds/evaluates all literature-guided feature packs and creates a leaderboard.
- `scripts/permutation_test_clean.py`: permutation-tests classifier performance against shuffled labels.
- `scripts/run_chronotype_permutation_tests.py`: batch permutation tests for chronotype packs.
- `scripts/feature_importance_clean.py`: cross-validated held-out permutation importance.
- `scripts/run_chronotype_importance.py`: focused importance analysis for the best chronotype model.
- `scripts/rebuild_from_raw.py`: raw-to-clean orchestration and provenance wrapper.
- `scripts/build_compact_chronotype.py`: theory-driven 12-feature chronotype model table.
- `scripts/repeated_cv_clean.py`: repeated stratified CV with confidence intervals.
- `scripts/group_stats_chronotype.py`: Morning-vs-Evening statistics with effect sizes.
- `scripts/qc_report_clean.py`: clean dataset QC summaries.

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

This project was run with the local virtual environment at `env/`.

## Workflow

Rebuild from local raw data and build clean datasets:

```bash
python scripts/rebuild_from_raw.py --execute
```

Build clean datasets from an existing local `data/processed/ml_ready_features.csv`:

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

Run manuscript-level support analyses:

```bash
python scripts/build_compact_chronotype.py
python scripts/repeated_cv_clean.py --data data/clean/chronotype_compact_12.csv --target Chronotype --model logreg --repeats 100
python scripts/permutation_test_clean.py --data data/clean/chronotype_compact_12.csv --target Chronotype --group-col "" --model logreg --permutations 1000
python scripts/group_stats_chronotype.py
python scripts/qc_report_clean.py
python scripts/rebuild_from_raw.py --execute
python scripts/build_chronotype_sensitivity.py
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
- Primary chronotype labels come from linked participant summary/final metadata.
- Chronotype behavioral adaptation features use previous-trial feedback; ERP contrasts use current feedback-locked labels.
- Risky choice is modelled at the trial level with participant-grouped CV.
- Risky-choice predictors only use same-trial pre-choice information plus previous-trial history.
- Feedback-locked EEG from the same trial is not used to predict the same trial's choice.
- Chronotype results are interpreted cautiously because the dataset has only 39 participants.

## Current Limitations

- Small participant sample, `n = 39`, limits generalizability.
- Participant `1013` has an EEG/trigger agreement issue after block 10 and one missing EEG trial in the raw-derived table.
- Raw behavioural chronotype labels conflict with the primary metadata labels for participants `1027` and `1036`.
- MEQ/MCTQ values are not exported because their workbook table order is not independently validated.
- Compact chronotype ML evidence is sensitivity-dependent after excluding flagged participants.
- Report CSV/JSON outputs are ignored and summarized in `docs/results.md` instead of committed directly.
- No external validation cohort is available.
- Feature importance is unstable because chronotype fold test sets are small.

## Recommended Next Work

- Add sensitivity analyses excluding participant `1013` and the two label-conflict participants.
- Add nested CV or simpler pre-registered models for manuscript-level claims.
- Validate the new 12-feature compact chronotype model with more participants.
- Validate on more participants or an external cohort.

## Manuscript Support Docs

- `docs/results.md`: current tracked result summary.
- `docs/methods.md`: methods draft for manuscript development.
- `docs/data_provenance.md`: generated provenance plan/record for raw-to-clean reconstruction.
- `docs/limitations.md`: limitations to disclose in a paper.

## Status

This is suitable as a public research/portfolio repository. Treat the chronotype result as a promising pilot finding, not a deployed or clinically validated predictor.
