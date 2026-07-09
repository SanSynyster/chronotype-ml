# Old vs New Headline Results

New results are from the authoritative participant-ID rebuild. Old numbers are quoted from the existing manuscript/results docs for comparison only.

| headline | old reported | new rebuilt |
| --- | --- | --- |
| Analysis sample | n = 39 | n = 39; Morning = 20, Evening = 19 |
| Pz P300 loss-minus-gain | d = -1.04; Welch p = 0.0028; FDR p = 0.034 | d = 0.077; Welch p = 0.8099; FDR p = 0.8099 |
| POz P300 loss-minus-gain | d = -0.92; Welch p = 0.0076; FDR p = 0.045 | d = -0.144; Welch p = 0.6535; FDR p = 0.7188 |
| Behaviour free risky-rate | d = 0.80; Welch p < 0.025 | d = -0.307; Welch p = 0.3406; FDR p = 0.6160 |
| Behaviour loss-error risky-rate | d = 0.81; Welch p < 0.025 | d = -0.276; Welch p = 0.3920; FDR p = 0.6160 |
| Causal GRU trial risky-choice | balanced accuracy = 0.603; AUC = 0.647 | balanced accuracy = 0.6021; AUC = 0.6522 |
| GRU chronotype decoding | AUC = 0.713; permutation p = 0.027 | AUC = 0.1895; permutation p = 0.9703; 100 permutations |
| ERP-only chronotype decoding | AUC = 0.668; permutation p = 0.032 | AUC = 0.4526; permutation p = 0.4455; 100 permutations |
| Fusion chronotype decoding | AUC = 0.797; permutation p = 0.004 | AUC = 0.2895; permutation p = 0.9010; 100 permutations |
| Continuous MEQ vs Pz P300 | r = 0.29; Spearman rho = 0.32 | Pearson r = -0.0517; p = 0.7544; Spearman rho = -0.0997 |
| Continuous MEQ vs POz P300 | r = 0.24; Spearman rho = 0.30 | Pearson r = 0.1448; p = 0.3791; Spearman rho = 0.2556 |
| Covariate-adjusted Pz P300 | adjusted Morning beta = 0.88; p = 0.041; partial d approximately 0.72 | adjusted Morning beta = -0.0859; p = 0.8230; partial d = -0.076 |
| Covariate-adjusted POz P300 | adjusted p = 0.12 | adjusted Morning beta = 0.2556; p = 0.5107; partial d = 0.225 |

## Notes

The old headline P300 and fusion effects do not reproduce after replacing the statistical linkage with the authoritative participant-ID key. The trial-level risky-choice GRU performance remains similar because that task predicts choices, not chronotype labels.

The permutation-count for rebuilt GRU/fusion chronotype analyses was 100 in this run. Single-trial EEG analyses requiring `data/raw/shifted_set/*.set` were not rerun because those files were absent in the workspace.
