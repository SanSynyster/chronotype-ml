# Deep-learning track (idea #5: risky-choice dynamics)

Self-contained PyTorch experiments kept **separate** from the main analysis pipeline.
They run in their own virtualenv (`env_dl/`, git-ignored) so the pinned MNE-1.7
analysis env in `env/` is never touched.

```bash
python -m venv env_dl && source env_dl/bin/activate
pip install torch pandas numpy scikit-learn joblib scipy
```

## Scripts

- `risky_choice_seq.py` — causal (unidirectional) GRU for trial-level risky choice.
  Grouped 5-fold over participants, leakage-safe: fed only current pre-choice context
  plus the *previous* trial's outcome, with the hand-engineered `Rolling*` history
  features deliberately removed so the network must learn temporal integration itself.
  Writes out-of-fold metrics and a per-participant embedding
  (`reports/clean/risky_choice_seq/`). Exposes `compute_oof()` for reuse.

- `chronotype_from_dynamics.py` — permutation-clean test of whether **chronotype** is
  decodable from those behavioral embeddings (the second-paper hook). The GRU never
  sees chronotype, so its embeddings are a leakage-safe feature set. The downstream
  classifier's hyperparameters (PCA dim, L2 `C`) are tuned **inside** nested
  leave-one-out CV, and the whole nested procedure is re-run under every label
  permutation — so the reported p-value already accounts for model selection and needs
  no further multiple-comparison correction. Continuous MEQ is used as independent
  corroboration. Writes `reports/clean/chronotype_from_dynamics/`.

## EEG track scripts (idea #1: single-trial EEG)

- `load_clean_epochs.py` — loads the cleaned epoched EEGLAB `.set`
  (`data/raw/shifted_set/`) into a `(13522, 64, 251)` tensor, 39 subjects, with
  feedback-valence labels parsed from the ERPLAB `B#(code)` bins.
- `eegnet.py` — EEGNet (Lawhern 2018) in PyTorch, exposes `.features()`.
- `run_eegnet_feedback.py` — cross-subject (GroupKFold) single-trial loss-vs-gain
  decoder; memory-frugal (per-batch normalization).
- `eeg_chronotype.py` — trains EEGNet on the chronotype-agnostic valence task, then
  tests chronotype on out-of-fold per-subject embeddings via `chrono_eval`. Builds
  two embeddings: `mean` (all trials) and `contrast` (loss − gain, matching the prior
  P300 loss-minus-gain finding).
- `multimodal_chronotype.py` — fuses the behavioral GRU embedding with the EEG
  embedding; reports each modality alone vs. combined.

## Headline results

**Behavioral (works):**
- GRU vs. baseline (risky choice): out-of-fold balanced acc **0.603 / AUC 0.647** vs.
  baseline **0.587 / 0.62** — without the `Rolling*` features.
- Chronotype from dynamics: nested-LOO ROC AUC **0.713**, perm **p = 0.027**,
  MEQ r = −0.31.

**EEG (honest negative):**
- EEGNet auxiliary task — cross-subject single-trial loss-vs-gain decoding works:
  AUC **0.641**, so the clean epochs carry decodable neural signal.
- But chronotype is **not** decodable from the per-subject EEG embeddings at n=39:
  `mean` AUC **0.426** (p=0.61), `contrast` AUC **0.389** (p=0.71). The `contrast`
  embedding does weakly track *continuous* MEQ (r=0.30, p=0.068).
- Multimodal fusion (behavioral + mean EEG): AUC **0.650** — adding the null EEG
  features *dilutes* the behavioral signal (0.713 → 0.650).

Interpretation: chronotype is robust in behavioral choice dynamics but elusive in
learned single-trial EEG features at this sample size, even though the hand-crafted
univariate P300 loss-minus-gain contrast separated the groups at the group level.
Likely n-limited; a contrast/condition-difference representation or self-supervised
pretraining (using the 13 extra unlabeled subjects) is the natural next attempt.

## Reproduce

```bash
# behavioral
python scripts/dl/risky_choice_seq.py
python scripts/dl/chronotype_from_dynamics.py --permutations 1000
# eeg
python scripts/dl/run_eegnet_feedback.py --epochs 25        # feedback decoding baseline
python scripts/dl/eeg_chronotype.py --epochs 25 --permutations 1000
python scripts/dl/multimodal_chronotype.py --permutations 1000
```
