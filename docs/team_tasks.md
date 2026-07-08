# Chronotype Paper — Tasks for the Team

*For Mahsima and Dr Heysiattalab. The analyses, figures, and a full draft are
complete and on the shared branch (PR #3); open `docs/paper.docx` (working draft,
shows what still needs filling) or `docs/paper_submission.docx` (clean version) in
Word for tracked-changes comments. The items below are the things only the team can
provide or decide. **Bold = hard blocker: the paper cannot be submitted without it.**
Please reply per numbered item or edit directly in the Word file.*

---

## A. Ethics and participants (**hard blockers**)

1. **Ethics/IRB:** approving committee name, protocol/approval number, and a
   written-informed-consent statement.
2. **Recruitment:** how and where participants were recruited, and over what period.
3. **Sample flow:** how many were screened → enrolled → excluded (and why) to reach
   the final N = 39.
4. **Demographics:** age (mean, SD, range) and sex/gender, ideally split by
   chronotype group (Morning vs Evening).
5. **Inclusion/exclusion criteria** (e.g., neurological/psychiatric history,
   medication, shift work, handedness, vision).

## B. Task design (needed for Methods)

6. Confirm the structure: number of **blocks and trials** (we have 16 × 24 — please
   confirm), the box **magnitudes** (we have {5, 25}), and the **free vs forced**
   trial split.
7. **Timing:** stimulus/choice window, feedback duration, inter-trial interval.
8. **Response mapping** (which key = left/right box), any **practice** trials, and
   how the running score / any **payment or incentive** worked.
9. **Task origin:** was the paradigm adapted from a published task? If so, the
   citation to include.

## C. EEG acquisition and preprocessing (**hard blockers**)

10. **Amplifier/system** (we have ANT Neuro — confirm model), cap/electrode type.
11. **Channels and montage** (confirm 64-channel, 10–20), **online reference** and
    **ground** locations, **sampling rate** (confirm 250 Hz), any online filters,
    impedance target.
12. **Preprocessing** (done on your side in EEGLAB/ERPLAB): filter band (high/low/
    notch), offline re-referencing, ICA/artifact-rejection method and criteria,
    epoch window, baseline interval.
13. **Final trial counts** after rejection (overall, and per condition if easy), and
    confirmation of feedback trigger codes 50/60/70/80 =
    Gain-Correct/Gain-Error/Loss-Correct/Loss-Error.

## D. Chronotype measurement (**hard blocker on item 14**)

14. **Which questionnaire(s)** and version (MEQ — Horne & Östberg? — and/or MCTQ),
    when administered, and the **exact rule** used to assign the binary Morning vs
    Evening label (median split? standard Horne–Östberg cut-offs?). *12 of 39 fall
    in the MEQ 42–58 intermediate band, so this rule matters.*
15. If a **translated MEQ** was used, the validation/norms reference for that version.

## E. Authorship, funding, data (needed at submission)

16. **Preregistration** status (none / OSF link). The draft's primary-vs-exploratory
    split is written to be preregistration-friendly.
17. **CRediT contributions** per author; **funding/grant** numbers; **conflicts of
    interest**; **corresponding author** and affiliations.
18. **Data availability:** are we clear to post the de-identified participant-level
    table (39 rows, no PII) to a public repository with a DOI, with raw EEG
    available on request? Any constraints from the ethics approval?

## F. Scientific review by the team (please read and mark up the draft)

19. **Title:** confirm or revise. Current working title: *"Chronotype Differences
    in Feedback-Related P300 and Risky-Choice Dynamics."*
20. **Direction-of-effect (Discussion §4.2):** confirm/sharpen the link between the
    posterior-P300 result and the specific prior chronotype–reward findings you want
    to foreground; add any preferred citations (one `[AUTHOR INPUT]` marker there).
21. **Interpretation check:** confirm the framing (ERP finding primary; behaviour/
    fusion convergent; RL exploratory) matches how you want the contribution read.
22. **Data-linkage method (Methods §2.4):** please sanity-check the description of
    how metadata was matched to behavioural files — reviewers will scrutinize it.

---

### Not blocking the team (handled on the analysis/writing side)
- Filling Methods once A–D arrive; final consistency pass.
- Optional: a supplementary validation table/figure for the metadata-to-behaviour
  linkage (can be produced on request).
- Converting the finalized draft to the journal's **LaTeX** template.
- Cover letter (first draft exists in `docs/cover_letter.md`).
