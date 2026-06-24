# Clean Modelling Pipeline

This folder contains the new leakage-aware workflow.

1. Build risky-choice features:

```bash
python scripts/build_clean_risky_choice.py
```

2. Build participant-level chronotype features:

```bash
python scripts/build_clean_chronotype.py
```

3. Train clean baselines:

```bash
python scripts/train_clean_baseline.py --data data/clean/risky_choice_prechoice.csv --target risky-choice --group-col participant_id
python scripts/train_clean_baseline.py --data data/clean/chronotype_participant.csv --target Chronotype --group-col ""
```

4. Build and evaluate literature-guided feature packs:

```bash
python scripts/run_literature_feature_packs.py
```

5. Permutation-test chronotype packs:

```bash
python scripts/run_chronotype_permutation_tests.py --permutations 200
```

6. Analyze feature importance for the significant compact chronotype model:

```bash
python scripts/run_chronotype_importance.py --repeats 50
```

Risky-choice packs: `value_only`, `history_only`, `value_history`, `prev_eeg`, `all_clean`.

Chronotype packs: `demo_only`, `behavior_core`, `frn_core`, `p300_core`, `compact_combined`, `all_literature`.

Permutation outputs are written under `reports/clean/permutation_tests/`.

Feature-importance outputs are written under `reports/clean/feature_importance/`.

The old scripts and generated artifacts are archived under `archive/legacy/`.

## Deep-learning & computational strand

The `scripts/dl/` subfolder holds the deep-learning / computational-modelling work (causal
GRU sequence model, EEGNet on single-trial EEG, asymmetric reinforcement-learning model,
multimodal behaviour+ERP fusion, continuous-MEQ regression, and a robustness battery). It
runs in its own `env_dl/` virtualenv (`requirements-dl.txt`) to keep the pinned analysis env
untouched. Headline: fused behaviour + validated ERP P300/FRN predicts chronotype at AUC
`0.797` (p = `0.004`). See `scripts/dl/README.md` and `docs/{methodology,results,discussion}_dl.md`.
