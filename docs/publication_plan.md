# Publication Plan — Integrated Chronotype × Feedback-Processing Paper

**Decision (2026-07-08):** One integrated paper (not two). Target journal:
**Psychophysiology** (SPR). Add high-leverage analyses before manuscript writing.
Working title: *Chronotype in the neural and behavioural evaluation of decision
feedback: converging ERP, computational, and fusion evidence.*

Supersedes the two-paper recommendation in `discussion_dl.md §8`.

---

## 1. The one-paper narrative arc

1. **Hook:** chronotype is linked to reward/risk behaviourally, but its neural
   feedback-processing signature is uncharacterised.
2. **Primary neural finding:** posterior P300 loss-minus-gain differs by
   chronotype (Pz d = −1.04, POz d = −0.92; FDR-corrected; exclusion-invariant).
3. **Behavioural convergence:** a causal GRU decodes chronotype from choice
   dynamics alone (AUC 0.713) — independent of EEG.
4. **Super-additive fusion (headline):** behaviour + validated ERP contrasts →
   AUC 0.797, p = 0.004, bootstrap CI [0.64, 0.92]; predicts continuous MEQ
   (r = 0.34). Partly independent information across modalities.
5. **Mechanism:** asymmetric-RL model — Evening = higher gain-learning, lower
   choice consistency.
6. **Honest boundary:** EEGNet learns the feedback task but not chronotype at
   n = 39 → validated low-D features beat end-to-end learning in small samples.
7. **Dimensional:** continuous-MEQ results throughout; not a dichotomisation
   artefact.

Selling point for Psychophysiology: **multi-method triangulation + leakage-safe,
permutation-clean rigor**, not scale.

---

## 2. High-leverage analyses to add (no new participants)

Prioritised by reviewer-objection payoff. Each is feasible on the existing data.

| # | Analysis | Answers objection | Effort | Priority |
|---|----------|-------------------|--------|----------|
| A | **Single-trial P300 → next-trial risk shift, moderated by chronotype** (mixed-effects logistic: does feedback-locked P300 amplitude predict the participant's subsequent risky choice, and does the slope differ Morning vs Evening?) | "Fusion is just two correlated features — where's the mechanism linking brain to behaviour?" | Med | **Highest** — this is a *new* mechanistic result, not a repackage |
| B | **Bayes factors for the nulls** (FRN group difference; EEGNet-chronotype). Quantify evidence *for* absence, not just failure to reject. | "Underpowered nulls are uninformative." | Low | High |
| C | **Continuous MEQ as the primary outcome** (re-center Ridge/regression results; report binary as secondary). | "Dichotomising a continuous trait; 12 intermediate-band Ss." | Low | High |
| D | **Hierarchical / partial-pooling RL** (stabilise per-subject α, β with ~270 trials each; report shrinkage). | "RL params hit optimisation bounds; unidentifiable." | Med | Med |
| E | **Specification-curve / multiverse for the P300** (window 400–600 ms grid, mean vs peak, baseline choices) showing the effect is not window-cherry-picked. | "Garden of forking paths on the ERP window." | Med | Med |
| F | **Equivalence / effect-size-first reporting** for behavioural contrasts (TOST where claiming null). | "Uncorrected medium effects overclaimed." | Low | Med |
| G | *(Aspirational)* **External check on a public reward/feedback-EEG dataset** — even validating only the behavioural-dynamics signature. Hard because chronotype/MEQ labels are rarely public. | "No external validation." (the #1 objection) | High / uncertain | Stretch |

**Recommended build order:** A → B → C → F → E → D → (G if a labelled public
set is found). A and B alone materially raise the ceiling.

Tooling note: Bayes factors + hierarchical models need `pymc`/`arviz` or R
`BayesFactor`/`brms`. **`env_dl` is Python 3.14 — `pymc` may not yet support
it**; plan to run Bayesian analyses in the `env` (3.11) venv or a dedicated
`env_bayes`. `pingouin` gives quick JZS Bayes factors for the t-tests (B, F).

---

## 3. Manuscript-preparation phases

**Phase 0 — Blockers (team-only; start now, parallel to analyses).** None of
these can be Claude-generated; without the ethics statement the paper cannot be
submitted. Collect from co-authors:
- [ ] Ethics/IRB approval body + protocol number + consent statement.
- [ ] Recruitment, N screened→analysed, age/sex distribution, inclusion/exclusion.
- [ ] Full EEG acquisition: amplifier (ANT Neuro model), montage, reference,
      sampling (250 Hz confirmed), online filters, ground.
- [ ] Preprocessing done upstream: filter band, ICA/artefact rejection, epoch
      window, baseline, final trial counts after rejection.
- [ ] Task: trials/blocks (16×24 stated — confirm), timing, response mapping,
      free vs forced, feedback timing, payment/incentive.
- [ ] MEQ/MCTQ administration + exact binary cutoff rule (median split vs
      Horne-Ostberg bands).
- [ ] Preregistration status (frame primary/exploratory split accordingly).
- [ ] CRediT roles, funding, conflicts, data-availability decision + DOI.

**Phase 1 — Lock analyses.** Run §2 additions; freeze all numbers; regenerate
figures. Add new figures: (F-A) single-trial P300→risk coupling by chronotype;
(F-B) Bayes-factor forest for the nulls; refresh fig9 fusion.

**Phase 2 — Merge the two manuscripts.** Fold `methodology_dl.md` /
`results_dl.md` / `discussion_dl.md` into the single IMRaD in
`manuscript_draft.md`. Restructure Results as the §1 arc. One methods section,
one discussion. Move the exploratory high-D ML + risky-choice-baseline detail to
Supplementary.

**Phase 3 — Citations & interpretation.** Fill every `[CITATION]`/`[REF]`
(chronotype-reward, FRN, P300, EEGNet, RL/computational, leakage/CV rigor).
Write the direction-of-effect interpretation vs. prior chronotype literature.

**Phase 4 — Psychophysiology formatting.** Structured abstract, author
guidelines length, reporting checklist (SPR endorses committee EEG/ERP guidelines
— Keil et al. 2014), CRediT, data/code availability. Build submission PDF +
cover letter.

**Phase 5 — Reproducibility pass.** `rebuild_from_raw.py --execute`; verify every
in-text statistic against script output (statcheck-style); freeze `requirements`.

**Deliverables:** single `manuscript_draft.md` (submission-ready), figure set,
supplementary, cover letter, response-to-reviewers scaffold, public 39-row
de-identified table + DOI.

---

## 4. Tooling / skills to add

**System tools to install:**
- `pandoc` + a LaTeX dist (`tinytex` is lightest) — md → journal PDF/DOCX.
- `pingouin` (Python) — quick Bayes factors, equivalence tests (B, F).
- `pymc` + `arviz` **or** R `brms`/`BayesFactor` — hierarchical RL, robust BFs
  (D). Run in Python 3.11 env, not 3.14.
- `statsmodels`/`bambi` — mixed-effects logistic for the P300→risk coupling (A).
- Zotero + Better BibTeX — reference management for Phase 3.
- `statcheck` (R) — automated stat-consistency check (Phase 5).

**Custom Claude Code skills worth building for this repo** (I can scaffold these):
- `/manuscript` — regenerate the merged IMRaD from the docs + latest report
  numbers, with figure callouts and the author-input checklist.
- `/cite` — WebSearch/WebFetch-backed pass that proposes references for each
  `[CITATION]`/`[REF]` with DOIs, for co-author approval.
- `/statcheck` — parse in-text stats and diff them against the script JSON/CSV
  outputs so no reported number drifts from the code.

**Existing skills already useful:** `/code-review` (reproducibility/leakage
audit of analysis scripts), `/verify` (confirm pipeline reruns clean),
`/security-review` (n/a). WebSearch/WebFetch for citation retrieval.
