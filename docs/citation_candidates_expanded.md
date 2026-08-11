# Expanded Citation Candidates — for Q1 (Psychophysiology) literature depth

**Purpose.** The manuscript currently cites 9 verified references — a skeleton. A
Q1 empirical paper of this scope needs ~50–80. Below is a claim-mapped candidate
list (~40 additions) grouped by theme; each is tied to the specific manuscript
sentence/section it would support.

**Instructions for the verifier (GPT / task G-CITE-2).**
- For each candidate: confirm the paper exists, retrieve the **DOI**, and confirm
  the exact author list, year, title, and venue. These are proposed from memory
  and **must not be trusted as-is**.
- Flag any you cannot verify in a `NEEDS VERIFICATION` block; do not invent DOIs.
- Append verified entries to `references.bib` and `docs/references_apa7.md`
  (APA-7). **Do not edit `docs/manuscript_draft.md`** — the Claude agent inserts
  citations against the mapped sentences after verification.
- Where a candidate is wrong/weak, suggest a better-known replacement for the same
  claim rather than dropping the claim's support.

Confidence key: **H** high (canonical, almost certainly correct), **M** medium
(exists but check authors/year/venue), **L** low (verify existence carefully).

---

## A. Chronotype × reward / impulsivity / risk  (Intro ¶1; Discussion §4.2)
Currently only Adan et al. (2012) — the thinnest and most important gap; reviewers
will want primary empirical sources for the "evening = more reward-sensitive /
impulsive / risk-taking" claim and for the direction-of-effect.

| # | Candidate (verify) | Supports | Conf |
|---|---|---|---|
| A1 | Hasler, Sitnick, Shaw & Forbes (2013), *Psychiatry Research: Neuroimaging* — evening chronotype & altered reward-related neural response | Intro claim + §4.2 direction (neural reward differences by chronotype) | M |
| A2 | Hasler & Clark (2013), *Alcoholism: Clin. Exp. Res.* — circadian misalignment & reward-related brain function | §4.2 circadian-reward framing | M |
| A3 | Antúnez, Navarro & Adan (2014), *Chronobiology International* — circadian typology & impulsivity/personality | Intro "higher impulsivity" | M |
| A4 | Muro, Gomà-i-Freixanet & Adan (2009), *Personality and Individual Differences* — morningness–eveningness & sensation seeking | Intro "reward sensitivity / sensation seeking" | M |
| A5 | Randler (2008 or Randler & Saliger 2011) — morningness & temperament/BIS-BAS | Intro trait framing | M |
| A6 | Killgore (2007), *Psychological Reports* — morningness–eveningness & risk-taking | Intro "more risk-taking" | L |
| A7 | Wittmann, Dinich, Merrow & Roenneberg (2006), *Chronobiology International* — social jetlag (MCTQ) | Methods (MCTQ) / Intro circadian misalignment | H |
| A8 | Owens, Dearth-Wesley et al. (adolescent chronotype & reward/risk) OR McGowan & Coogan | Intro developmental/risk context | L |

## B. Feedback P300 / reward positivity  (Intro ¶2; Results §3.1; Discussion §4.2)
Currently only Polich (2007). The P300-as-salience and feedback/reward-ERP
literature is central to the headline neural claim.

| # | Candidate (verify) | Supports | Conf |
|---|---|---|---|
| B1 | Proudfit (2015), *Psychophysiology* — the reward positivity | §4.2 feedback reward ERP; in-journal | H |
| B2 | San Martín (2012), *Frontiers in Human Neuroscience* — review of outcome-processing ERPs | Intro FRN/P300 dissociation | H |
| B3 | Yeung & Sanfey (2004), *Journal of Neuroscience* — independent coding of reward magnitude & valence | Intro "P300 scales with salience/valence" | H |
| B4 | Wu & Zhou (2009), *Brain Research* — P300 and reward valence/magnitude/expectancy | §3.1 P300 valence sensitivity | H |
| B5 | Hajcak, Moser, Holroyd & Simons (2007), *Psychophysiology* — feedback negativity & reward-prediction violations | Intro FRN | H |
| B6 | Sambrook & Goslin (2015), *Psychological Bulletin* — meta-analysis of reward-prediction-error ERPs | §4.2 FRN-null localisation argument | H |
| B7 | Walsh & Anderson (2012), *Neurosci. Biobehav. Rev.* — ERP correlates of reward processing & choice | Intro/Discussion feedback-ERP overview | H |

## C. FRN / feedback valence  (Intro ¶2)
| # | Candidate (verify) | Supports | Conf |
|---|---|---|---|
| C1 | Gehring & Willoughby (2002), *Science* — medial frontal cortex & rapid gain/loss processing | Intro FRN valence sensitivity | H |

## D. Circadian ↔ dopamine / reward in humans  (Discussion §4.2, §4.4)
Currently only Webb et al. (2009), a rodent study — needs human/mechanistic support.

| # | Candidate (verify) | Supports | Conf |
|---|---|---|---|
| D1 | Murray et al. (2009), *Emotion* — circadian system modulates reward motivation (human) | §4.2/§4.4 circadian-reward mechanism | H |
| D2 | McClung (2007), *Pharmacology & Therapeutics* — circadian genes, dopamine & mood | §4.4 circadian/dopaminergic modulation | M |
| D3 | Byrne & Murray (2017) — reward system & circadian rhythms (review) | §4.4 mechanism | L |

## E. Reinforcement learning / computational modelling  (Methods §2.8; Results §3.10; Discussion §4.4)
Currently none — and there is a full RL section. Asymmetric gain/loss learning and
hierarchical-Bayesian fitting need grounding.

| # | Candidate (verify) | Supports | Conf |
|---|---|---|---|
| E1 | Frank, Moustafa, Haughey, Curran & Hutchison (2007), *PNAS* — dopamine & gain/loss learning | §2.8/§4.4 asymmetric learning rates & dopamine | H |
| E2 | Lefebvre, Lebreton, Meyniel, Bourgeois-Gironde & Palminteri (2017), *Nature Human Behaviour* — optimistic (asymmetric) RL | §2.8 α_gain/α_loss asymmetry | H |
| E3 | Gershman (2015), *Psychonomic Bulletin & Review* — do learning rates adapt? | §2.8 learning-rate modelling | M |
| E4 | Niv, Edlund, Dayan & O'Doherty (2012), *Journal of Neuroscience* — risk-sensitive RL in humans | §3.10/§4.4 risk & RL | H |
| E5 | Daw (2011), *Attention & Performance XXIII* — trial-by-trial computational model fitting | §2.8 model-fitting method | H |
| E6 | Ahn, Haines & Zhang (2017), *Computational Psychiatry* — hBayesDM hierarchical Bayesian RL | §2.8/§3.10 hierarchical partial pooling | H |
| E7 | Sutton & Barto (2018), *Reinforcement Learning: An Introduction* (2nd ed.), MIT Press | §2.8 Q-learning framework | H |
| E8 | Cho et al. (2014), EMNLP — GRU (RNN encoder–decoder) | §2.7 GRU architecture | H |

## F. ML rigor / decoding / deep EEG  (Methods §2.6, §2.9, §2.10; Discussion §4.5)
Currently only Lawhern (2018). The leakage-safe/permutation-clean rigor is a
selling point and should be cited to the methods literature.

| # | Candidate (verify) | Supports | Conf |
|---|---|---|---|
| F1 | Varoquaux (2018), *NeuroImage* — small-sample cross-validation error bars | §2.6/§4.5 small-n caution | H |
| F2 | Combrisson & Jerbi (2015), *J. Neurosci. Methods* — permutation vs theoretical chance in decoding | §2.10 permutation testing | H |
| F3 | Kriegeskorte, Simmons, Bellgowan & Baker (2009), *Nature Neuroscience* — circular analysis / double dipping | §2.5/§2.10 leakage avoidance | H |
| F4 | Poldrack, Huckins & Varoquaux (2020), *JAMA Psychiatry* — best practices for prediction | §2.10 predictive-modelling rigor | H |
| F5 | Roy et al. (2019), *J. Neural Engineering* — deep learning for EEG (systematic review) | §2.9/§4.5 deep EEG context | H |
| F6 | Schirrmeister et al. (2017), *Human Brain Mapping* — CNNs for EEG decoding | §2.9 EEGNet context | H |

## G. Statistics used but uncited  (Methods §2.6; Results §3.2, §3.11)
| # | Candidate (verify) | Supports | Conf |
|---|---|---|---|
| G1 | Benjamini & Hochberg (1995), *JRSS-B* — false discovery rate | §2.6 FDR correction | H |
| G2 | Rouder, Speckman, Sun, Morey & Iverson (2009), *Psychon. Bull. Rev.* — Bayesian t-tests | §3.2/§3.11 Bayes factors | H |
| G3 | Lakens (2017), *Soc. Psychol. Personal. Sci.* — equivalence tests (TOST) | §3.2 equivalence tests | H |
| G4 | Kahneman & Tversky (1979), *Econometrica* — prospect theory / loss aversion | §3.1/§4.2 loss-vs-gain framing | H |

## H. Chronotype measurement (optional support)  (Methods §2.4)
| # | Candidate (verify) | Supports | Conf |
|---|---|---|---|
| H1 | Roenneberg, Kuehnle, Juda et al. (2007), *Sleep Medicine Reviews* — epidemiology of the human clock (MCTQ) | §2.4 MCTQ | M |

---

## Team-only citations (co-authors must supply — not verifiable here)
- The **specific prior chronotype-reward findings** to foreground in the Introduction
  and the §4.2 direction-of-effect comparison (which studies, which direction).
- Any **prior work by the team/collaborators** that should be cited.
- The **origin/source of the risky-choice task** if it was adapted from a published
  paradigm (needed in Methods §2.3).
- Country/population-appropriate **MEQ validation/norms** if a translated MEQ was used.

## Target
9 (current) + ~40 (above, once verified) ≈ **50 references** — in the Q1 range,
with every citation mapped to a specific claim rather than padding.
