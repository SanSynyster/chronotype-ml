# Data Provenance

Generated: `2026-06-12T18:34:16.304136+00:00`
Executed commands: `True`

## Raw Inputs

| path | exists | bytes | columns |
| --- | ---: | ---: | ---: |
| `data/raw/_singletrial_means` | True | 2560 |  |
| `data/raw/raw_behavioral_trials.xlsx` | True | 944183 | 17 |
| `data/raw/participant_summary.xlsx` | True | 12720 | 30 |
| `data/raw/all final data.xlsx` | True | 20527 | 35 |

## Commands

1. `/Users/sahabtaali/chronotype_ml/env/bin/python scripts/link_raw_metadata.py`
2. `/Users/sahabtaali/chronotype_ml/env/bin/python scripts/build_ml_ready_from_raw.py`
3. `/Users/sahabtaali/chronotype_ml/env/bin/python scripts/build_clean_risky_choice.py --include-prev-eeg --write-packs`
4. `/Users/sahabtaali/chronotype_ml/env/bin/python scripts/build_clean_chronotype.py --write-packs`
5. `/Users/sahabtaali/chronotype_ml/env/bin/python scripts/build_compact_chronotype.py`
6. `/Users/sahabtaali/chronotype_ml/env/bin/python scripts/build_compact_performance_chronotype.py`
7. `/Users/sahabtaali/chronotype_ml/env/bin/python scripts/build_chronotype_sensitivity.py`

## Outputs

| path | exists | rows | columns | bytes |
| --- | ---: | ---: | ---: | ---: |
| `data/processed/participant_metadata.csv` | True | 39 | 14 | 9225 |
| `data/processed/ml_ready_features.csv` | True | 14352 | 40 | 5336188 |
| `data/clean/risky_choice_prechoice.csv` | True | 10669 | 55 | 4985566 |
| `data/clean/chronotype_participant.csv` | True | 39 | 173 | 118896 |
| `data/clean/chronotype_compact_12.csv` | True | 39 | 14 | 9810 |
| `data/clean/chronotype_compact_performance.csv` | True | 39 | 14 | 9792 |
| `data/clean/sensitivity/manifest.json` | True |  |  | 2238 |

## Notes

- Raw data are local and intentionally not committed.
- Participant_summary is linked to UserID by optimal one-to-one (Hungarian) assignment over previous-feedback behavioural aggregates recomputed from raw behaviour; the assignment is a guaranteed bijection with reported match-margin QC.
- All final data is linked through the ERPset column shared with participant_summary.
- Primary chronotype labels come from participant_summary / all final data metadata, not the raw behavioural Chronotype column. Both metadata sources agree on every participant; only the raw behavioural column disagrees for 1027 and 1036.
- MEQ/MCTQ scores live in a non-row-aligned table block of all final data; they are aligned by participant name (the ERPset field) in scripts/validate_meq_labels.py, which confirms the binary labels (26/26 decisive participants consistent; conflict cases 1027/1036 MEQ-confirmed).
- The performance-informed compact model is exploratory and kept separate from the a priori theory-driven compact model.
- Active scripts build leakage-aware modelling datasets from the raw-derived ML-ready feature table.
