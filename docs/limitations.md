# Limitations

- The chronotype analysis has only 39 participants, so all predictive findings are pilot-level.
- The compact chronotype model must be externally validated before making generalizable claims.
- ERP features are window-level summaries and may miss peak latency, time-frequency, or trial-quality effects.
- The active raw-to-clean path is now implemented in active scripts, but participant `1013` has a known EEG/trigger agreement issue after block 10 and one missing EEG trial.
- Primary chronotype labels come from linked `all final data.xlsx` metadata; the raw behavioral chronotype column conflicts for `1027` and `1036`.
- MEQ/MCTQ values are not exported because their side-by-side workbook table order is not independently validated.
- Compact chronotype evidence is significant in the full all-final-label dataset, but sensitivity analyses are mixed when excluding flagged participants.
- The performance-informed compact model is exploratory because feature selection was informed by the current dataset.
- Larger Random Forest models are above permutation chance but remain exploratory because they use 47-171 features with only 39 participants.
- Raw data are local and not committed to the repository.
- The risky-choice model is intentionally leakage constrained, which lowers performance but makes the task scientifically valid.
