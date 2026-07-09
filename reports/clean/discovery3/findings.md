# Discovery3: Risk/Agency and Individual Differences

Chronotype ignored. All joins use integer participant id; averaged ERP workbooks use the whitespace-stripped ERPset column parsed as participant id. Primary inference is subject-level with bootstrap CIs, sign-flip/permutation p-values, and FDR within families.

## P1 Risk x Outcome ERP Effects
- frn_gain_correct_high_minus_low: dz=0.575, dz 95% CI [0.309, 0.847], mean diff=1.334, mean-diff 95% CI [0.705, 1.965], Wilcoxon p=0.0001357, permutation p=0.0002, FDR q=0.0003999; verdict=real.
- frn_loss_error_high_minus_low: dz=0.643, dz 95% CI [0.368, 0.909], mean diff=1.468, mean-diff 95% CI [0.835, 2.074], Wilcoxon p=3.033e-05, permutation p=0.0002, FDR q=0.0003999; verdict=real.
- p300_gain_correct_high_minus_low: dz=-0.371, dz 95% CI [-0.653, -0.109], mean diff=-0.709, mean-diff 95% CI [-1.248, -0.218], Wilcoxon p=0.02122, permutation p=0.009598, FDR q=0.0128; verdict=real.
- p300_loss_error_high_minus_low: dz=-0.288, dz 95% CI [-0.553, -0.022], mean diff=-0.481, mean-diff 95% CI [-0.927, -0.021], Wilcoxon p=0.04711, permutation p=0.04819, FDR q=0.04819; verdict=real.

Sanity checks:
- frn_lowrisk_loss_minus_gain: dz=-0.450, permutation p=0.0022, FDR q=0.004399; verdict=real.
- frn_lowrisk_error_minus_correct: dz=-1.397, permutation p=0.0002, FDR q=0.0007998; verdict=real.
- p300_lowrisk_loss_minus_gain: dz=-0.204, permutation p=0.145, FDR q=0.1933; verdict=null.
- p300_lowrisk_error_minus_correct: dz=0.023, permutation p=0.896, FDR q=0.896; verdict=null.

## P2 Risk-Modulated RPE Encoding
- FRN risky-minus-safe signed-RPE slope: mean diff=0.1988, median diff=0.1169, 95% bootstrap CI [-0.3534, 0.7991], Wilcoxon p=0.8698, permutation p=0.5391, FDR q=0.5391; verdict=null.
- THETA risky-minus-safe signed-RPE slope: mean diff=-0.6201, median diff=-0.9688, 95% bootstrap CI [-1.1264, -0.1025], Wilcoxon p=0.02335, permutation p=0.0224, FDR q=0.06719; verdict=weak.
- P300 risky-minus-safe signed-RPE slope: mean diff=-0.6181, median diff=-0.9019, 95% bootstrap CI [-1.3511, 0.0679], Wilcoxon p=0.1306, permutation p=0.1004, FDR q=0.1506; verdict=null.

## P3 Individual Differences
N=52 gives roughly 80% power only for correlations around |r| >= 0.38, so CIs and corrected p-values matter more than nominal hits.

Strongest neural behavior correlations by Pearson FDR:
- frn_riskmod_gain_correct vs risky_rate: Pearson r=-0.586, 95% CI [-0.733, -0.396], perm p=0.0002, FDR q=0.0168; Spearman rho=-0.587, FDR q=0.0168; verdict=real.
- frn_error_minus_correct_low vs post_error_slowing: Pearson r=0.434, 95% CI [0.172, 0.631], perm p=0.0012, FDR q=0.05039; Spearman rho=0.404, FDR q=0.1764; verdict=weak.
- p300_riskmod_loss_error vs late_minus_early_risk: Pearson r=-0.358, 95% CI [-0.560, -0.111], perm p=0.008398, FDR q=0.2352; Spearman rho=-0.260, FDR q=0.6353; verdict=weak.
- frn_loss_minus_gain_all vs mean_rt: Pearson r=-0.380, 95% CI [-0.675, 0.097], perm p=0.0116, FDR q=0.2436; Spearman rho=-0.135, FDR q=0.7834; verdict=weak.
- theta_rpe_slope_risky vs risky_rate: Pearson r=0.316, 95% CI [0.077, 0.522], perm p=0.0204, FDR q=0.3427; Spearman rho=0.293, FDR q=0.5159; verdict=weak.
- p300_error_minus_correct_low vs late_minus_early_risk: Pearson r=0.312, 95% CI [-0.014, 0.528], perm p=0.02879, FDR q=0.4031; Spearman rho=0.174, FDR q=0.6587; verdict=weak.
- frn_loss_minus_gain_all vs post_error_slowing: Pearson r=0.238, 95% CI [-0.045, 0.498], perm p=0.09258, FDR q=0.5879; Spearman rho=0.176, FDR q=0.6587; verdict=null.
- frn_loss_minus_gain_all vs win_stay: Pearson r=0.249, 95% CI [-0.001, 0.447], perm p=0.07918, FDR q=0.5879; Spearman rho=0.193, FDR q=0.6587; verdict=null.

Strongest neural rl correlations by Pearson FDR:
- frn_riskmod_gain_correct vs bias: Pearson r=-0.572, 95% CI [-0.716, -0.392], perm p=0.0002, FDR q=0.014; Spearman rho=-0.585, FDR q=0.014; verdict=real.
- frn_riskmod_gain_correct vs alpha_gain: Pearson r=-0.353, 95% CI [-0.539, -0.140], perm p=0.0104, FDR q=0.3126; Spearman rho=-0.363, FDR q=0.2729; verdict=weak.
- frn_rpe_slope_risky vs alpha_loss: Pearson r=0.340, 95% CI [0.087, 0.569], perm p=0.0134, FDR q=0.3126; Spearman rho=0.309, FDR q=0.2819; verdict=weak.
- p300_loss_minus_gain_all vs lr_asymmetry: Pearson r=0.313, 95% CI [0.104, 0.503], perm p=0.024, FDR q=0.3415; Spearman rho=0.317, FDR q=0.2819; verdict=weak.
- theta_rpe_slope_risky vs bias: Pearson r=0.309, 95% CI [0.066, 0.520], perm p=0.0244, FDR q=0.3415; Spearman rho=0.296, FDR q=0.3237; verdict=weak.
- frn_error_minus_correct_low vs bias: Pearson r=-0.237, 95% CI [-0.459, -0.007], perm p=0.09158, FDR q=0.5828; Spearman rho=-0.325, FDR q=0.2819; verdict=null.
- frn_riskmod_gain_correct vs lr_asymmetry: Pearson r=0.242, 95% CI [-0.037, 0.491], perm p=0.08838, FDR q=0.5828; Spearman rho=0.256, FDR q=0.4821; verdict=null.
- frn_riskmod_loss_error vs alpha_gain: Pearson r=-0.257, 95% CI [-0.550, 0.047], perm p=0.07039, FDR q=0.5828; Spearman rho=-0.308, FDR q=0.2819; verdict=null.

## Frank Assessment
Novel layer verdict beyond the established signed-RPE foundation: weak.
Risk/agency modulation of RPE coupling is null after subject-level permutation/FDR; individual-difference results are treated as exploratory unless they survive correction.
