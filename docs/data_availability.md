# Data Availability

This document is a draft data-availability statement and a description of what
can be shared. [AUTHOR INPUT: confirm the sharing decision, repository, and DOI
before submission.]

## What can be shared

- **Analysis code:** the full pipeline in this repository, with a pinned
  environment (`requirements.txt`, Python 3.11) and a complete dependency freeze
  (`requirements.lock.txt`).
- **Derived participant-level table:** a PII-free, anonymized table of
  aggregated behavioural and ERP features plus chronotype, age, and gender
  (39 rows). It contains no names, emails, phone numbers, or dates, and the study
  identifier is replaced with an anonymous label (P01-P39). Generate it with:

  ```bash
  python scripts/export_public_data.py
  ```

  The output (`data/public/chronotype_participant_public.csv`) is git-ignored;
  releasing it is a deliberate author decision and is not committed automatically.

## What is not shared by default

- **Raw data** (behavioural trials, EEG single-trial means, metadata workbooks,
  and the MEQ/MCTQ questionnaire responses) are held locally and not committed.
  The questionnaire responses in particular contain direct identifiers (names,
  emails) and must not be shared without de-identification and consent review.
- **Trial-level tables** are large and held locally; they can be released on
  request in anonymized form.

## Draft statement

> The code supporting this study is openly available at [REPOSITORY/DOI]. A
> de-identified participant-level dataset of aggregated behavioural and ERP
> features sufficient to reproduce the primary analyses is available at
> [REPOSITORY/DOI]. Raw EEG and behavioural data, and questionnaire responses,
> contain potentially identifying information and are available from the
> corresponding author on reasonable request, subject to ethics approval and a
> data-sharing agreement.

[AUTHOR INPUT: corresponding author, ethics protocol number, repository/DOI.]
