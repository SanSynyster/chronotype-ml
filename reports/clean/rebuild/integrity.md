# Authoritative Rebuild Integrity Report

Status: proceeded after adjudication that behavioural-vs-key age mismatches are non-blocking timing differences. The key remains authoritative for age.

## Source Files Checked

| role | path | status |
| --- | --- | --- |
| participant key | `data/raw/meq mctq scores - Sheet1.csv` | source of truth |
| behaviour | `data/raw/all behavioral-2.xlsx` | joined by `UserID` to key `participant id` |
| FRN ERP | `data/raw/frn_all_25-_350.xlsx` | joined by stripped `ERPset` |
| P300 ERP | `data/raw/p300_all_350_450.xlsx` | joined by stripped `ERPset` |

No files under `data/_outdated_raw/`, pre-existing `data/processed/`, or pre-existing `data/clean/` were read by `scripts/rebuild_authoritative.py`.

## Sample Definition From Key

| quantity | value |
| --- | ---: |
| screened participants in key | 56 |
| morning in key | 21 |
| evening in key | 20 |
| intermediate in key | 15 |
| has EEG | 52 |
| analysis sample, has EEG and Morning/Evening | 39 |
| analysis Morning | 20 |
| analysis Evening | 19 |

## Join Assertions

| check | result |
| --- | --- |
| key participant IDs unique and non-missing | passed |
| behavioural `UserID` values non-missing | passed |
| behavioural IDs present in key | passed |
| analysis IDs present in behaviour | passed |
| behavioural gender agrees with key gender | passed |
| behavioural age agrees with key age | warning only; 26 participants |
| FRN `ERPset` IDs unique and non-missing after stripping whitespace | passed |
| P300 `ERPset` IDs unique and non-missing after stripping whitespace | passed |
| FRN/P300 analysis IDs present | passed |

## Age Warnings

Behavioural age differed from key age for 26 participants. This is logged in `reports/clean/rebuild/behavior_key_age_warnings.csv` and is not treated as an identity conflict after adjudication. All downstream tables use key age.

## Hard Conflicts

No hard identity conflicts were found for ID presence/uniqueness or gender. Chronotype, MEQ, MCTQ, age, gender, and EEG availability were taken only from the key.

## Outputs Regenerated

| output | status |
| --- | --- |
| `data/clean/participant_master.csv` | regenerated |
| `data/processed/ml_ready_features.csv` | regenerated from authoritative inputs |
| `data/clean/chronotype_participant.csv` | regenerated |
| `data/clean/chronotype_compact_12.csv` | regenerated |
| `data/clean/chronotype_*` feature packs | regenerated |
| `data/clean/risky_choice_prechoice.csv` | regenerated |

## Unavailable Inputs

`data/raw/shifted_set/*.set` was not present in this workspace, so EEGNet, single-trial EEG coupling, and P300 specification-curve analyses that require `.set` epochs could not be rerun from the specified source.
