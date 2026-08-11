# Parallel Work Plan — Claude × Codex/GPT

**Goal:** produce a submission-ready, single integrated manuscript for
**Psychophysiology**: *Chronotype in the neural and behavioural evaluation of
decision feedback — converging ERP, computational, and fusion evidence.*
Full strategy in `docs/publication_plan.md`. This file is the division of labour
for running Claude and a Codex/GPT agent in parallel.

**Status (2026-07-08):** C-SPEC done → `docs/specs_for_gpt.md` (GPT can start
G-E + G-D). C-A done → `scripts/dl/p300_risk_coupling.py`,
`reports/clean/p300_coupling/`, `docs/figures/fig10_p300_risk_coupling.png`.
C-A result: within-subject single-trial P300→next-risk coupling does **not**
differ by chronotype (overall d = −0.36, p = 0.22; valence-resolved d = −0.08,
p = 0.62; GLMM interaction p = 0.12) — an honest negative that localises the
chronotype effect to the between-subject/trait level and pre-empts the "is it
just a between-subject correlation?" objection.

**Guiding split:**
- **Codex/GPT** gets self-contained, tightly-specified coding/stats tasks and
  formatting/drafting where correctness is checkable against a spec — it works
  well in parallel and is fast at boilerplate + standard statistical code.
- **Claude** keeps tasks needing deep codebase context, the leakage-safe
  evaluation framework, scientific-narrative judgment, cross-file integration,
  and anything where a subtle correctness error would poison the headline result.

---

## 0. Coordination contract (both agents follow)

1. **Disjoint file ownership** (table below) — never edit a file the other owns.
2. **Handoff via artifacts, not memory.** Every new analysis script writes its
   numbers to `reports/clean/<name>/summary.json` (+ a `.md`). The manuscript
   pulls numbers from those files, so neither agent has to re-run the other's code.
3. **Branch:** GPT works on `analysis-additions` off `dl-risky-choice-dynamics`;
   Claude works on `dl-risky-choice-dynamics`. Merge GPT→base after review.
4. **Every script:** fixed seed = 0, participant-grouped/LOO evaluation, one
   `argparse` entry point, deps declared, runnable from repo root. No new numbers
   go in the manuscript until Claude has reviewed the script for leakage.
5. **Env:** Bayesian/hierarchical work needs Python 3.11 (`env/`) or a new
   `env_bayes/` — **not** `env_dl/` (3.14; pymc unsupported). See task S0.

---

## 1. Task board

### Codex/GPT — owns these (self-contained, spec-driven)

| ID | Task | Deliverable / file | Spec notes |
|----|------|--------------------|-----------|
| **S0** | Set up `env_bayes` (Py 3.11) with `pymc`, `arviz`, `pingouin`, `statsmodels`, `bambi`; pin to `requirements-bayes.txt`. | `requirements-bayes.txt`, README stanza | Verify `import pymc` works before closing. |
| **G-B** | **Bayes factors for the nulls.** JZS Bayes factor for (i) FRN error−correct group difference, (ii) EEGNet-chronotype AUC vs chance. | `scripts/dl/bayes_factors_nulls.py` → `reports/clean/bayes_nulls/summary.json` | Use `pingouin.bayesfactor_ttest` for FRN; for EEGNet null use BF on the permutation-null vs observed. Report BF01 (evidence *for* null). |
| **G-F** | **Equivalence tests (TOST)** for behavioural contrasts where we claim null/near-null (FRN contrasts). | `scripts/dl/equivalence_tests.py` → `reports/clean/tost/summary.json` | Bounds ±0.5 SD; report per-contrast TOST p and decision. |
| **G-E** | **P300 specification curve.** Recompute Pz/POz loss−gain group d across a grid of windows (e.g. 400–600 ms in 25 ms steps), mean vs peak, baseline choices. | `scripts/dl/p300_spec_curve.py` → `reports/clean/spec_curve/{summary.json,curve.png}` | Show effect is not window-cherry-picked. Read epochs the same way `scripts/dl/` already does; ask Claude for the loader entry point. |
| **G-D** | **Hierarchical RL** (partial pooling of α_gain, α_loss, β, bias across participants) replacing per-subject MLE. Report group posteriors + shrinkage vs the MLE fits. | `scripts/dl/rl_hierarchical.py` → `reports/clean/rl_hier/summary.json` | Same likelihood as existing asymmetric-RL model (see `methodology_dl.md §5.2`). PyMC, NUTS, report R-hat. Compare group contrasts to the MLE table. |
| **G-CITE** | **Citation candidates.** For every `[CITATION]`/`[REF]` in the manuscript + DL docs, propose a reference with DOI and a one-line justification, in a table. Do **not** edit the manuscript. | `docs/citation_candidates.md` | Use web sources; flag any you cannot verify a DOI for. Co-authors approve before Claude inserts. |
| **G-FMT** | Draft the **Psychophysiology formatting shell**: structured-abstract skeleton, the SPR/committee ERP reporting checklist (Keil et al. 2014) as a fillable list, CRediT + data-availability boilerplate. | `docs/psychophys_formatting.md` | Formatting only; no scientific claims. |

### Claude — keeps these (context / judgment / integration)

| ID | Task | Deliverable |
|----|------|-------------|
| **C-A** | **Analysis A — single-trial P300 → next-trial risk shift, ×chronotype** (the highest-payoff new result). Own the design, data schema, leakage check, mixed-effects logistic model, and interpretation. *May hand GPT the pure model-fitting once the design + data frame are locked (see C-A-sub).* | `scripts/dl/p300_risk_coupling.py` → `reports/clean/p300_coupling/` + new figure |
| **C-SPEC** | Write the tight specs + provide the epoch/loader entry points GPT needs for G-E and G-D; review every GPT script for leakage before its numbers enter the manuscript. | inline review notes |
| **C-MERGE** | **Merge** `methodology_dl.md` + `results_dl.md` + `discussion_dl.md` + `manuscript_draft.md` into one IMRaD following the §1 arc in `publication_plan.md`. Move exploratory high-D ML + risky-choice baselines to Supplementary. | updated `docs/manuscript_draft.md` |
| **C-INTERP** | Write continuous-MEQ-as-primary reframing (C), direction-of-effect interpretation vs prior chronotype literature, and integrate G-B/G-D/G-E/G-CITE results into the narrative. | manuscript prose |
| **C-SKILLS** | Scaffold the `/manuscript`, `/cite`, `/statcheck` repo skills. | `.claude/skills/…` |
| **C-REPRO** | Final reproducibility + statcheck pass: verify every in-text number against the `reports/clean/*/summary.json` artifacts. | check report |
| **C-EMAIL** | Draft the co-author email requesting the Phase-0 blockers (ethics/IRB, recruitment, EEG acquisition + preprocessing, task details, MEQ cutoff, prereg status). | `docs/coauthor_request.md` |

### Team-only (neither agent — submission-blocking)

Ethics/IRB statement, recruitment & demographics, full EEG acquisition +
preprocessing params, task design confirmation, MEQ binary cutoff rule,
preregistration status, CRediT/funding/conflicts, data-availability decision.
Requested via **C-EMAIL**.

---

## 2. Why this split

- **GPT gets** B, F, E, D, CITE, FMT, env setup: each is a standalone script or
  document with a checkable spec, minimal cross-file coupling, and standard
  statistical or formatting content — parallelisable and low-risk if GPT and
  Claude never touch the same file.
- **Claude keeps** A (novel mechanism, subtle leakage design), the manuscript
  merge/interpretation (whole-narrative judgment), skill scaffolding
  (harness-specific), reproducibility sign-off, and review of GPT's leakage
  safety — the places where a wrong call would compromise the headline claim.
- **The seam** is the `reports/clean/*/summary.json` artifacts + the citation and
  formatting docs: GPT produces numbers/candidates, Claude reviews and weaves
  them in. No shared source file, so parallel work won't conflict.

---

## 3. Sequencing

1. **Now, in parallel:** GPT → S0 then G-B, G-F, G-CITE, G-FMT. Claude → C-EMAIL,
   C-A design + C-SPEC, C-SKILLS.
2. **After S0 + C-SPEC:** GPT → G-E, G-D (need the loader entry points).
3. **After analyses land + reviewed:** Claude → C-MERGE, C-INTERP.
4. **Last:** Claude → C-REPRO once all `summary.json` artifacts exist and
   co-author Phase-0 inputs arrive.
