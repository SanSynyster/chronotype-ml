# Chronotype Paper — Remaining Tasks for the Team

*For Mahsima, Dr Heysiattalab, and Dr Zarean. A full draft, analyses, figures, and
references are complete and on the shared branch (PR #3); open `docs/paper.docx`
(working draft) or `docs/paper_submission.docx` (clean version) in Word for
tracked-changes comments. Most Methods details have now been sourced from the team's
2025 paper (Hajiaboo et al., Int. J. Psychophysiology) on the same cohort. The list
below is what still needs the team.*

---

## Context (please read first)

This manuscript uses the **same 39-participant cohort** as the already-published
**Hajiaboo, Zarean & Heysieattalab (2025)**, *The effect of chronotype in risky
decision making: An ERP study* (Int. J. Psychophysiology, 217, 113258). To avoid any
overlap/contradiction problem, the paper is being **repositioned as a computational
follow-up** that cites and builds on the 2025 paper: the behavioural and FRN results
are treated as previously established, and the new contribution is the posterior-P300
loss-minus-gain contrast plus the computational strand (sequence modelling,
brain–behaviour combination, continuous-MEQ prediction, leakage-safe evaluation).

## Already resolved (no action needed)

- Same 39-participant subset as the 2025 paper ✓
- Ethics: University of Tabriz committee, **IR.TABRIZU.REC.1403.130** (registered
  under M. Zarean); written informed consent ✓
- Participants, task, EEG acquisition + preprocessing, MEQ/MCTQ, and the binary
  cutoff (MEQ ≤ 41 evening / ≥ 59 morning) — taken from the 2025 paper ✓
- Demographics from the repository dataset: N = 39 (20 Morning, 19 Evening); age
  22.3 ± 3.0 (18–31); 20 M / 19 F ✓
- **Authorship:** Taali (first; computational lead), Hajiaboo, Heysiattalab, **Zarean
  (last)**; **corresponding author: Dr Heysiattalab** ✓ *(please confirm the order of
  the middle two.)*
- Data: raw subject data will **not** be shared publicly; statement set to
  "available from the corresponding author on reasonable request" ✓
- Target journal: **Psychophysiology** (a same-cohort follow-up in a sister journal
  is fine provided the 2025 paper is disclosed in the cover letter) ✓
- A sex/age imbalance across groups was found and handled: the primary Pz P300 effect
  survives adjustment for sex and age; POz attenuates and is partly age-related
  (reported in Results and Limitations) ✓

---

## Still needed from the team

### From Mahsima — measurement details to reconcile the two papers
These let us state clearly that our analyses differ from, rather than contradict, the
2025 paper.

1. **FRN measurement in the 2025 paper:** exact time window (ms), electrodes, whether
   it was a difference wave or raw mean amplitude, and the baseline used.
2. **P300 measurement in the 2025 paper:** window and electrodes, and whether the
   **loss-minus-gain contrast** was ever tested (the 2025 paper reports no P300 main
   effect / no MEQ–P300 correlation, so we need to show our contrast is a different,
   more sensitive test).
3. **Shared pipeline?** Are our single-trial ERP exports (`Pz_P300`, `POz_P300`, etc.)
   produced from the **same EEGLAB/ERPLAB pipeline and epochs** as the 2025 paper?
4. **Keep vs supplement:** how much of the behavioural/FRN material do you want kept
   as a short "prior-work / within-cohort" paragraph versus moved to the supplement?

### From Dr Heysiattalab / Dr Zarean — confirmations
5. Confirm the **repositioning** as a computational follow-up citing the 2025 paper.
6. Confirm **author order** (middle two) and provide **CRediT** roles.
7. **Funding/grants** and **conflicts of interest**.

### From Sahab (self) — to action
8. Locate and send the **OSF preregistration** link so the manuscript's
   primary/confirmatory/exploratory labels can be aligned to what was preregistered,
   and the prereg cited. (Check whether the P300 was preregistered or whether the
   prereg covered the FRN/behavioural hypotheses of the 2025 paper.)

---

*Everything else — ethics, participants, task, EEG acquisition/preprocessing, MEQ
cutoff, demographics, corresponding author, data-availability wording — is resolved.*
