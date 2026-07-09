# Old vs New Headline Results

The new headline analyses were not run because the strict integrity rebuild stopped at the behavioural age cross-check. See `reports/clean/rebuild/integrity.md` and `reports/clean/rebuild/behavior_key_age_gender_conflicts.csv`.

| result | old reported number | new rebuilt number |
| --- | --- | --- |
| Pz P300 loss-minus-gain d and p | not recomputed here | not generated; integrity stop |
| POz P300 loss-minus-gain d and p | not recomputed here | not generated; integrity stop |
| behavioural risky-rate d and p | not recomputed here | not generated; integrity stop |
| GRU AUC | not recomputed here | not generated; integrity stop |
| fusion AUC | not recomputed here | not generated; integrity stop |
| continuous MEQ r | not recomputed here | not generated; integrity stop |
| covariate-adjusted P300 | not recomputed here | not generated; integrity stop |

No downstream value should be interpreted until the age conflicts between the authoritative key and behavioural workbook are resolved or the team explicitly changes the cross-check rule.
