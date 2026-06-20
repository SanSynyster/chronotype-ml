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

## Headline results

- GRU vs. baseline (risky choice): out-of-fold balanced acc **0.603 / AUC 0.647** vs.
  baseline **0.587 / 0.62** — and the GRU does it without the `Rolling*` features.
- Chronotype from dynamics: nested-LOO ROC AUC ≈ **0.71**, corroborated by a negative
  correlation with the continuous MEQ score. See `metrics.json` for the
  permutation-tested p-value from the latest run.

## Reproduce

```bash
python scripts/dl/risky_choice_seq.py                       # GRU + embeddings
python scripts/dl/chronotype_from_dynamics.py --permutations 1000
```
