# Authoritative Rebuild Integrity Report

Status: stopped before clean-table generation because the behavioural age cross-check failed against `data/raw/meq mctq scores - Sheet1.csv`.

## Source Files Checked

| role | path | status |
| --- | --- | --- |
| participant key | `data/raw/meq mctq scores - Sheet1.csv` | read as source of truth |
| behaviour | `data/raw/all behavioral-2.xlsx` | read for trial rows and age/gender cross-check only |
| FRN ERP | `data/raw/frn_all_25-_350.xlsx` | not reached after behavioural conflict |
| P300 ERP | `data/raw/p300_all_350_450.xlsx` | not reached after behavioural conflict |

No files under `data/_outdated_raw/`, pre-existing `data/processed/`, or pre-existing `data/clean/` were read by the rebuild script.

## Sample Definition From Key

| quantity | value |
| --- | ---: |
| screened participants in key | 56 |
| morning in key | 21 |
| evening in key | 20 |
| intermediate in key | 15 |
| has EEG | 52 |
| potential analysis sample, has EEG and Morning/Evening | 39 |
| potential analysis Morning | 20 |
| potential analysis Evening | 19 |

The behavioural workbook contains 19,968 rows for 52 unique `UserID` values. All 39 potential Morning/Evening EEG analysis IDs are present in the behavioural workbook.

## Join Assertions

| check | result |
| --- | --- |
| key participant IDs unique and non-missing | passed |
| behavioural `UserID` values non-missing | passed |
| behavioural IDs present in key | passed |
| potential analysis IDs present in behaviour | passed |
| behavioural gender agrees with key gender | passed |
| behavioural age agrees with key age | failed |

## Conflicts

The rebuild stopped because these participants have an age mismatch between the authoritative key and the behavioural file. Gender agreed for all listed participants.

| participant_id | behaviour_age | behaviour_gender | key_age | key_gender |
| ---: | ---: | :--- | ---: | :--- |
| 1001 | 26 | M | 23 | M |
| 1002 | 23 | F | 24 | F |
| 1003 | 23 | F | 24 | F |
| 1009 | 22 | F | 23 | F |
| 1013 | 22 | M | 23 | M |
| 1024 | 21 | F | 22 | F |
| 1025 | 23 | F | 25 | F |
| 1028 | 20 | M | 21 | M |
| 1029 | 19 | M | 20 | M |
| 1030 | 22 | M | 23 | M |
| 1031 | 22 | M | 23 | M |
| 1032 | 22 | F | 20 | F |
| 1033 | 23 | M | 25 | M |
| 1035 | 21 | F | 22 | F |
| 1037 | 18 | M | 21 | M |
| 1038 | 21 | M | 23 | M |
| 1043 | 20 | M | 21 | M |
| 1044 | 21 | F | 22 | F |
| 1046 | 21 | M | 22 | M |
| 1047 | 21 | F | 22 | F |
| 1049 | 21 | M | 22 | M |
| 1050 | 25 | F | 26 | F |
| 1051 | 24 | M | 26 | M |
| 1052 | 24 | F | 25 | F |
| 1053 | 24 | F | 23 | F |
| 1054 | 19 | M | 20 | M |

Machine-readable conflict output is also in `reports/clean/rebuild/behavior_key_age_gender_conflicts.csv`.

## Downstream Analyses

No participant-level clean table, trial-level clean table, ERP stats, classifier, DL, or old-vs-new headline report was generated after this conflict. Continuing would violate the instruction to stop rather than guess when key and behavioural demographics disagree.
