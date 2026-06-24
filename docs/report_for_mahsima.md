# Update: Deep-learning analyses on the chronotype EEG/behavioral data

Hi Mahsima,

Thanks for sending the cleaned, epoched `.set` files — they made all the difference.
Here is a summary of the deep-learning work and what we found.

## 1. The EEG files now open in Python

The original raw recordings (`.cnt`) are **ANT Neuro** format, not Neuroscan — which is
why they wouldn't open with the usual MNE reader and gave errors. The **cleaned,
epoched `.set` files you sent solved this completely**: they load in one line, are
already denoised and baseline-corrected, and contain the single-trial epochs we need.

What we're working with: **39 chronotype-labeled participants** (plus 13 extra subjects
with EEG but no label), 64 channels, 250 Hz, ~13,500 single-trial feedback epochs,
window −0.2 to +0.8 s, with the four feedback conditions
(gain-correct / gain-error / loss-correct / loss-error) preserved.

## 2. Methods note (so the numbers are trustworthy)

Everything is **leakage-safe and permutation-tested**: participants are never split
across train/test, model settings are chosen *inside* the cross-validation, and every
p-value comes from a 1000-iteration label-permutation test. With only 39 participants
this rigor matters, so the numbers below are conservative and honest.

## 3. Main findings

**(a) Chronotype is predictable from behavior alone.**
A sequence model (GRU) that learns how each person adapts their risky choices after
wins and losses distinguishes Morning vs Evening types:
**ROC AUC 0.713, p = 0.027**, and it correlates with the continuous MEQ score
(r = −0.31). This is a *new, independent* line of evidence from the P300 result.

**(b) The P300/FRN effect holds up.**
Using only the validated feedback ERP contrast features (the FRN and the Pz/POz P300
loss-minus-gain that survived correction), chronotype is predictable at
**AUC 0.668, p = 0.032** — confirming your neural finding within this framework.

**(c) Brain + behavior add together — the strongest result.**
Combining the behavioral model with the P300/FRN features pushes accuracy well past
either alone: **AUC 0.797, p = 0.004**, balanced accuracy 0.742, MEQ correlation −0.42.
Because the combination beats both parts, the neural and behavioral signals are
carrying **partly independent** information about chronotype. This is the headline.

| Model | AUC | p |
|---|---|---|
| Behavioral choice dynamics | 0.713 | 0.027 |
| Feedback P300/FRN (validated features) | 0.668 | 0.032 |
| **Combined** | **0.797** | **0.004** |

**(d) Honest negative: end-to-end EEG deep learning did not work — yet.**
A convolutional network (EEGNet) *can* decode the feedback type (loss vs gain) from
single trials across new participants (AUC 0.641), so the clean data clearly carries
signal. But it could **not** learn to predict chronotype from the raw single-trial EEG
(AUC ≈ 0.4, not significant). This is almost certainly a sample-size limit: 39 people
is too few for a network to *learn* the subtle chronotype difference, even though your
hand-measured P300 feature captures it. The lesson is useful — the validated,
hand-crafted ERP features are what work, not learned features at this n.

## 4. What this means

- Chronotype shows up **strongly in behavior** and **clearly in the feedback P300**, and
  the two are **complementary** — together they reach ~0.80 accuracy.
- This strengthens, rather than replaces, the P300 finding: behavior is a second,
  independent window onto the same Morning/Evening difference.

## 5. Suggested next steps / what would help

- For deep learning on the EEG itself to have a chance, we'd need **more participants**
  (or we use the 13 extra unlabeled subjects for self-supervised pre-training). Happy to
  discuss.
- If you have the **trial-rejection logs / original trial indices** for the cleaned
  epochs, that would let us line up each EEG trial with its behavioral trial even more
  precisely (useful for future trial-level brain-behavior models).
- For writing up: I think this is cleanest as **two papers** — your existing P300 paper
  unchanged, and a second computational paper led by the behavioral result and the
  brain+behavior combination (with the EEG deep-learning attempt reported honestly as a
  limitation). Keen to hear your view.

All code and results are on the project repo (branch `dl-risky-choice-dynamics`).

Best,
Sahab
