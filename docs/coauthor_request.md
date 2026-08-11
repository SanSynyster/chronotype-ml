# Co-author information request — submission-blocking inputs

*Draft email to Mahsima Hajiaboo and Dr Sommayye Heysiattalab. These are the items
Claude/analysis cannot supply and that no journal will accept the paper without.
Everything else (statistics, figures, code) is final and reproducible. Please fill
in-line or reply per numbered item; items are grouped by who most likely holds the
information. Target journal: **Psychophysiology** (its reporting standards drive the
level of detail requested).*

---

Subject: Chronotype paper — the few details only the team can provide (methods + ethics)

Hi Mahsima and Dr Heysiattalab,

The analyses, effect sizes, and figures for the integrated chronotype paper are
final and reproducible. To move to submission at **Psychophysiology** I need a
small set of details that live on your side. I've grouped them and marked which
are **hard blockers** (the paper literally cannot be submitted without them).
Bullet answers are fine.

## A. Ethics & consent — HARD BLOCKER
1. Name of the approving ethics committee / IRB and the **protocol/approval
   number**.
2. Confirmation that participants gave **written informed consent**, and whether
   they were compensated (amount/form).
3. Any consent constraints on **data sharing** (affects our data-availability
   statement and whether the de-identified 39-row table can be posted).

## B. Participants & recruitment — HARD BLOCKER
4. How were participants **recruited** (source, community/student, dates/period)?
5. **How many were screened / enrolled / excluded**, and the reasons for any
   exclusions, to reach the final N = 39.
6. **Age** (mean, SD, range) and **sex/gender** breakdown, ideally per chronotype
   group (Morning vs Evening).
7. Inclusion/exclusion criteria (e.g., neurological/psychiatric history, medication,
   shift work, handedness, normal/corrected vision).

## C. Chronotype / MEQ measurement — HARD BLOCKER
8. Which questionnaire(s) and version were administered (MEQ — Horne & Östberg? —
   and/or MCTQ), and **when** relative to the EEG session.
9. The **exact rule** used to assign the binary Morning vs Evening label (median
   split? standard Horne–Östberg cut-offs? something else?). *We currently note 12
   of 39 fall in the MEQ 42–58 intermediate band; the paper's handling of these
   depends on your rule.*

## D. EEG acquisition — HARD BLOCKER (Psychophysiology requires full detail)
10. Amplifier / system (we have it as **ANT Neuro** — please confirm model, e.g.
    eego/asalab) and **cap/electrode** type.
11. **Number of channels and montage** (we assume 64-ch, 10–20 — confirm), the
    **online reference** and **ground** locations.
12. **Sampling rate** (we have 250 Hz — confirm) and any **online filters**.
13. **Electrode impedance** target (e.g., < 5 kΩ) and recording environment.

## E. Preprocessing (done on your side in EEGLAB/ERPLAB) — HARD BLOCKER
14. **Filter settings** (high-pass, low-pass, notch) and order/type.
15. **Re-referencing** scheme used offline (e.g., average, mastoids).
16. **Artifact handling**: ICA? which components removed and how identified;
    amplitude/threshold rejection criteria.
17. **Epoching**: window relative to feedback, **baseline** interval used.
18. **Final trial counts** after rejection (overall and, if easy, per condition) —
    lets us state the exact epoch N (~13,522 pre-rejection in our load).
19. Confirmation of the **feedback trigger codes** 50/60/70/80 =
    gain-correct/gain-error/loss-correct/loss-error.

## F. Task design — needed for the Methods
20. Confirm structure: **16 blocks × 24 trials**, free vs forced trials, magnitudes
    {5, 25} with hidden sign revealed as feedback.
21. **Trial timing**: stimulus/choice window, feedback duration, inter-trial
    interval; response mapping (which key = left/right).
22. Any **practice** trials, instructions about risk, and how the running **score**
    was displayed/rewarded.

## G. Reporting & authorship — needed at submission
23. **Preregistration** status (none / OSF link). *We've written the primary-vs-
    exploratory split to be preregistration-friendly.*
24. **CRediT contributions** for each author (conceptualization, methodology,
    investigation, formal analysis, writing, etc.).
25. **Funding** sources / grant numbers and any **conflicts of interest**.
26. **Corresponding author** and affiliation details.
27. **Data-availability** preference: are we clear to post the de-identified
    participant-level table (39 rows, no PII) to OSF/Zenodo with a DOI, with raw
    EEG available on request? (Depends on item 3.)

The moment I have A–E I can finalise the Methods and the reporting checklist;
F–G I need before we hit submit.

Thanks — this is the last substantive gap between the current draft and a
submittable manuscript.

Sahab
