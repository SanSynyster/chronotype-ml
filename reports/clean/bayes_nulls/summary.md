# Bayes Factors for Planned Nulls

BF01 values greater than 1 favour the null over the tested alternative.

## FRN Error-Minus-Correct Group Difference

| Contrast | Evening mean | Morning mean | Welch t | p | BF01 |
|---|---:|---:|---:|---:|---:|
| Fz_FRN_error_minus_correct | -2.8945 | -2.0229 | -1.8790 | 0.0681 | 0.8178 |
| FCz_FRN_error_minus_correct | -2.9119 | -2.4817 | -0.8168 | 0.4193 | 2.4621 |
| Cz_FRN_error_minus_correct | -2.0618 | -2.1496 | 0.1713 | 0.8649 | 3.1684 |

## EEGNet Chronotype Decoding

Permutation-null density-ratio BF01; see script header for assumptions.

| Embedding | Observed AUC | Null mean | Null SD | Permutation p | BF01 |
|---|---:|---:|---:|---:|---:|
| mean | 0.4263 | 0.4610 | 0.1280 | 0.6074 | 1.2515 |
| contrast | 0.3895 | 0.4595 | 0.1327 | 0.7113 | 1.1906 |

## Positive Control

EEGNet feedback-valence AUC = 0.5938; balanced accuracy = 0.5552.
