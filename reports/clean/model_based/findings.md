# Model-Based EEG Findings

Primary question: does feedback-locked single-trial EEG encode RL prediction error beyond raw outcome valence/magnitude?

Requested random-slope MixedLM fits did not converge reliably, so fixed-effect betas/p-values below use the scripted participant fixed-effects model with participant-clustered robust SEs. The two-stage subject slopes, bootstrap CIs, and within-participant permutation p-values are the main robustness checks.

## FRN
- Signed RPE before outcome covariates: beta=0.2028, p=1.959e-34.
- Signed RPE after outcome covariates: beta=0.1973, p=1.329e-15, FDR q=3.987e-15.
- |RPE| after outcome covariates: beta=0.0649, p=0.1249.
- Two-stage signed-RPE slope: mean=0.1891, 95% bootstrap CI [0.1451, 0.2342], Wilcoxon p=8.635e-09, within-participant permutation p=0.000999.
- Verdict: yes.

## P300
- Signed RPE before outcome covariates: beta=0.0164, p=0.2812.
- Signed RPE after outcome covariates: beta=0.0250, p=0.3044, FDR q=0.3044.
- |RPE| after outcome covariates: beta=0.0561, p=0.2176.
- Two-stage signed-RPE slope: mean=0.0039, 95% bootstrap CI [-0.0425, 0.0516], Wilcoxon p=0.8913, within-participant permutation p=0.7353.
- Verdict: no.

## THETA
- Signed RPE before outcome covariates: beta=-0.1057, p=6.001e-10.
- Signed RPE after outcome covariates: beta=-0.1204, p=2.503e-05, FDR q=3.754e-05.
- |RPE| after outcome covariates: beta=0.0321, p=0.4643.
- Two-stage signed-RPE slope: mean=-0.1335, 95% bootstrap CI [-0.1829, -0.0843], Wilcoxon p=1.72e-05, within-participant permutation p=0.000999.
- Verdict: yes.

## Behavioural Relevance
- Grouped-CV next-choice AUC with behaviour/RL covariates: 0.656.
- Grouped-CV next-choice AUC after adding EEG features: 0.655.
- Grouped-CV next-choice AUC after adding RPE/outcome-independent EEG residuals: 0.655.
- FRN subject-level next-choice coefficient: mean=-0.0153, Wilcoxon p=0.4176.
- FRN RPE/outcome-independent residual next-choice coefficient: mean=-0.0148, Wilcoxon p=0.4124.
- P300 subject-level next-choice coefficient: mean=0.0140, Wilcoxon p=0.4443.
- P300 RPE/outcome-independent residual next-choice coefficient: mean=0.0141, Wilcoxon p=0.4335.
- THETA subject-level next-choice coefficient: mean=0.0028, Wilcoxon p=0.9564.
- THETA RPE/outcome-independent residual next-choice coefficient: mean=0.0026, Wilcoxon p=0.9492.

## Robustness
- MLE RPE signed effects: FRN beta=0.1559, p=2.941e-16; P300 beta=0.0205, p=0.3116; theta beta=-0.0496, p=0.06323.
- Excluding top 1% |RPE|: FRN p=4.638e-16; P300 p=0.2353; theta p=3.712e-05.
- Theta beyond ERP amplitude: signed RPE beta=-0.1237, p=9.775e-06; |RPE| beta=0.0312, p=0.4768.
- Alternative +/-50 ms time-window results are in summary.json.

## Bottom Line
The mechanistic verdict is based on the outcome-adjusted, FDR-corrected RPE terms and the within-participant permutation/two-stage checks. FRN/reward-positivity and frontal theta track signed RPE beyond raw outcome; P300 and theta surprise/|RPE| do not. EEG did not add leakage-safe next-choice prediction beyond behaviour/RL in this run.
