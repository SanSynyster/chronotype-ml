# Corrected Discovery Findings

Chronotype is treated as a null side-variable; analyses join by integer participant id only.

- **T1 EEG feedback valence:** EEGNet cross-subject AUC 0.575 (95% CI 0.561-0.590), balanced accuracy 0.545, 10-permutation p=0.091. Verdict: above-chance neural signal, but p-value resolution is coarse because full EEGNet permutations were computationally expensive.
- **T1 EEG feedback correctness:** EEGNet cross-subject AUC 0.628 (95% CI 0.610-0.650), balanced accuracy 0.585, 10-permutation p=0.091. Verdict: strongest EEG decoding anchor under the fast corrected run.
- **T1 EEG 4-way condition:** EEGNet cross-subject macro OVR AUC 0.604 (95% CI 0.589-0.619), accuracy 0.446, 10-permutation p=0.091. Verdict: condition information is present, but inferential precision is limited.
- **T2 choice dynamics:** AUC 0.632 (95% CI 0.604-0.658), permutation p=0.010; real.
- **T2 temporal drivers:** risk choice is driven most by recent risky-choice rate, previous RT, previous reward, trial progress, and recent loss rate.
- **T3 RL asymmetry:** mean loss-gain learning-rate asymmetry 0.140 (CI 0.038-0.243), p=0.012.
- **T4 ERP to next choice:** AUC 0.572 (CI 0.541-0.600), permutation p=0.010; real.
- **T4 ERP slopes:** per-subject FRN slope mean 0.045 (CI 0.006-0.085), p=0.036; P300 slope mean -0.045 (CI -0.090--0.001), p=0.059. Verdict: weak but coherent ERP-next-choice coupling; behavioural carryover still explains more than ERP alone.
- **T5 individual differences:** Exploratory CV metrics are reported in summary.json; do not interpret as confirmatory without correction/replication.
- **T5 MEQ/chronotype side-variable:** continuous MEQ prediction was null/negative under CV (R2=-0.167); this supports keeping chronotype out of the main story.

## Honest Paper Verdict
A defensible non-chronotype paper probably exists, but it should be framed narrowly as feedback processing and decision dynamics, not chronotype. The strongest corrected results are behavioural sequence predictability, asymmetric loss-vs-gain updating, and modest feedback-ERP coupling to the next choice. The EEG feedback anchor is present but needs a longer confirmatory EEGNet run with more permutations before making strong inferential claims.
