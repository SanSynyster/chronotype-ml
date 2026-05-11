# Limitations

- The chronotype analysis has only 39 participants, so all predictive findings are pilot-level.
- The compact chronotype model must be externally validated before making generalizable claims.
- ERP features are window-level summaries and may miss peak latency, time-frequency, or trial-quality effects.
- The active raw-to-clean path is now implemented in active scripts, but participant `1013` has a known EEG/trigger agreement issue after block 10 and one missing EEG trial.
- Primary chronotype labels come from linked participant summary/final metadata; raw behavioural labels conflict for participants `1027` and `1036` and should be disclosed in manuscripts.
- MEQ/MCTQ values are not exported because their side-by-side workbook table order is not independently validated.
- The compact chronotype model is significant in the full primary dataset but sensitivity-dependent after excluding flagged participants.
- Raw data are local and not committed to the repository.
- The risky-choice model is intentionally leakage constrained, which lowers performance but makes the task scientifically valid.
