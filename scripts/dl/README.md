# Deep-learning & computational track

Self-contained PyTorch / computational-modelling experiments kept **separate** from the
main analysis pipeline. They run in their own virtualenv (`env_dl/`, git-ignored) so the
pinned MNE-1.7 analysis env in `env/` is never touched.

```bash
python -m venv env_dl && source env_dl/bin/activate
pip install -r requirements-dl.txt
```

All chronotype decoding uses one shared, permutation-clean evaluator (`chrono_eval.py`):
hyper-parameters (PCA dim, L2 `C`) are tuned **inside** nested leave-one-out CV, and the
whole nested procedure is re-run under every label permutation, so reported p-values need
no further multiple-comparison correction. Methods/results/discussion write-ups are in
`docs/methodology_dl.md`, `docs/results_dl.md`, `docs/discussion_dl.md`.

## Scripts

**Behavioural (idea #5 — risky-choice dynamics)**
- `risky_choice_seq.py` — causal (unidirectional) GRU for trial-level risky choice.
  Grouped 5-fold over participants, leakage-safe (current pre-choice context + previous
  outcome only; `Rolling*` history features withheld so the net learns them). Writes
  out-of-fold metrics + per-participant embedding. Exposes `compute_oof()`.
- `chronotype_from_dynamics.py` — permutation-clean test of chronotype from the GRU
  behavioural embedding; continuous MEQ as corroboration.
- `make_dynamics_figure.py` — permutation null + MEQ-scatter figure.

**EEG (idea #1 — single-trial EEG)**
- `load_clean_epochs.py` — loads the cleaned epoched EEGLAB `.set` (`data/raw/shifted_set/`)
  into a `(13522, 64, 251)` tensor, 39 subjects, with feedback-valence labels.
- `eegnet.py` — EEGNet (Lawhern 2018) in PyTorch; exposes `.features()`.
- `run_eegnet_feedback.py` — cross-subject single-trial loss-vs-gain decoder (memory-frugal).
- `eeg_chronotype.py` — trains EEGNet on the chronotype-agnostic valence task, tests
  chronotype on out-of-fold per-subject `mean` and `contrast` (loss−gain) embeddings.

**Fusion, mechanism, validation, robustness**
- `chrono_eval.py` — shared permutation-clean nested-LOO evaluator (`evaluate`, `nested_auc`).
- `multimodal_chronotype.py` — fuses GRU + *learned* EEG embedding (control: it dilutes).
- `fusion_gru_p300.py` — fuses GRU + validated ERP P300/FRN features (the headline: it adds).
- `rl_model.py` — per-subject asymmetric reward-learning model (α_gain, α_loss, β, bias).
- `rl_analysis.py` — RL-parameter group/MEQ stats + predictive check.
- `continuous_meq.py` — nested-LOO Ridge regression predicting the continuous MEQ score.
- `robustness.py` — bootstrap CI, exclusion sensitivity, leave-one-subject-out influence.

## Headline results (N = 39, permutation-clean)

**Prediction**
- Risky-choice GRU: out-of-fold balanced acc **0.603 / AUC 0.647** vs. baseline 0.587 / 0.62,
  without the `Rolling*` features.
- Chronotype from behaviour: AUC **0.713**, p **0.027**, MEQ r −0.31.
- Chronotype from validated ERP P300/FRN: AUC **0.668**, p **0.032**.
- **Fusion (behaviour + ERP): AUC 0.797, p 0.004, MEQ r −0.42** — exceeds either alone.
- Continuous MEQ regression: fused r **0.344**, p **0.027** (behaviour 0.310; not reliant on
  the binary split).

**Mechanism (RL model)**
- Evening types learn more from gains (α_gain d=0.59, group p=0.040, MEQ r=−0.32) and choose
  less consistently (β MEQ r=0.36, p=0.027); Morning types weight losses more (asymmetry
  trend p=0.072). RL params *jointly* classify chronotype only weakly (AUC 0.532) — mechanism,
  not prediction. (Five comparisons uncorrected → exploratory.)

**Honest negative (deep EEG)**
- EEGNet decodes feedback valence cross-subject (AUC **0.641**), but chronotype is **not**
  decodable from learned single-trial embeddings (mean 0.426 p=0.61; contrast 0.389 p=0.71;
  contrast weakly tracks MEQ r=0.30). Fusing the learned EEG embedding *reduces* performance
  (0.650) — validated features, not learned ones, are what add.

**Robustness (fused model)**
- Bootstrap 95% CI **[0.639, 0.924]**; holds under all exclusions (drop 1013 →0.762,
  label-conflicts →0.795, all-flagged →0.775, MEQ-intermediate n=27 →0.698); leave-one-subject-out
  AUC range 0.653–0.853 (most influential = participant 1001).

## Reproduce

```bash
# behavioural
python scripts/dl/risky_choice_seq.py
python scripts/dl/chronotype_from_dynamics.py --permutations 1000
# eeg
python scripts/dl/eeg_chronotype.py --epochs 25 --permutations 1000
# fusion / mechanism / validation / robustness
python scripts/dl/rl_model.py
python scripts/dl/rl_analysis.py --permutations 1000
python scripts/dl/fusion_gru_p300.py --permutations 1000
python scripts/dl/continuous_meq.py --permutations 1000
python scripts/dl/robustness.py --permutations 1000
```
