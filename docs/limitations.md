# Limitations

- The chronotype analysis has only 39 participants from a single cohort, so all predictive findings are preliminary and require independent replication.
- The chronotype classifier does not survive FDR correction across the family of feature packs; the predictive ML is reported as exploratory, converging support for the neural group difference, which is the primary finding.
- The compact chronotype model must be externally validated before making generalizable claims.
- With 19 vs 20 participants the study is powered (80%) only for large effects (minimum detectable Cohen's d ~ 0.90); medium effects (d ~ 0.5) have ~35% power, so weak or null results for behavioral and FRN contrasts are inconclusive rather than evidence of no effect.
- ERP features are window-level summaries and may miss peak latency, time-frequency, or trial-quality effects.
- The active raw-to-clean path is now implemented in active scripts, but participant `1013` has a known EEG/trigger agreement issue after block 10 and one missing EEG trial.
- Primary chronotype labels come from linked `all final data.xlsx` metadata; the raw behavioral chronotype column conflicts for `1027` and `1036`, but the continuous MEQ score decisively confirms the metadata labels for both (1027 MEQ = 61, Morning; 1036 MEQ = 27, Evening).
- Binary chronotype is a dichotomization of an underlying continuous MEQ score; 12 of 39 participants fall in the MEQ intermediate band (42-58) where the Morning/Evening split is inherently soft. Future work should model the continuous score.
- Compact chronotype evidence is significant in the full all-final-label dataset, but sensitivity analyses are mixed when excluding flagged participants. Because the excluded label-conflict participants are MEQ-confirmed, the loss of significance under that exclusion reflects reduced sample size rather than label error.
- The performance-informed compact model is exploratory because feature selection was informed by the current dataset.
- Larger Random Forest models are above permutation chance but remain exploratory because they use 47-171 features with only 39 participants.
- Raw data are local and not committed to the repository.
- The risky-choice model is intentionally leakage constrained, which lowers performance but makes the task scientifically valid.
