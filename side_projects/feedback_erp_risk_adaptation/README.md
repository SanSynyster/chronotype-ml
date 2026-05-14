# Feedback-Locked ERP Prediction of Subsequent Risk Adaptation

## Working Title

Feedback-Locked ERP Prediction of Subsequent Risk Adaptation During Risky Decision-Making

## Project Status

This is a planned side project derived from the main chronotype EEG/behavioral ML repository. It is intentionally separated from the active chronotype pipeline so the current project remains stable and reproducible.

No analysis scripts or generated datasets have been implemented here yet. The folders in this side project are placeholders for future work.

## Core Research Question

Can feedback-locked ERP responses predict how participants adapt their risky choices on the next free-choice opportunity?

More specifically, do FRN and P300 responses to feedback on trial `t` predict whether the participant makes a risky choice on the next free-choice trial, beyond what can be explained by task context and behavioral history alone?

## Scientific Rationale

Risky decision-making is not only determined by static preference. Participants receive gains, losses, correct outcomes, and errors, then update or fail to update their future behavior. Feedback-locked ERP components provide a temporally precise measure of this feedback evaluation process.

The proposed study uses the correct temporal direction for EEG-based prediction:

1. A participant makes a choice.
2. Feedback is presented.
3. Feedback-locked EEG/ERP responses are measured.
4. The participant later makes another free choice.
5. The model tests whether the feedback response predicts that subsequent risky choice.

This avoids the leakage problem of using same-trial feedback-locked EEG to predict the same trial's choice.

## Primary Hypothesis

Feedback-locked ERP features will improve prediction of subsequent risky-choice adaptation beyond behavioral history and feedback condition alone.

Expected neurocognitive interpretation:

- FRN features index feedback monitoring, error processing, and negative outcome sensitivity.
- P300 features index feedback salience, attention allocation, and context updating.
- Stronger feedback evaluation may predict more adaptive future risk adjustment.
- Weaker or atypical feedback evaluation may predict risk persistence after negative feedback.

## Primary Target

The primary trial-level target will be:

```text
next_free_choice_risky
```

For each feedback trial `t`, identify the next free-choice trial from the same participant, `t_next`.

```text
next_free_choice_risky = 1 if t_next is risky
next_free_choice_risky = 0 if t_next is non-risky
```

This target should be evaluated with participant-grouped cross-validation so that trials from the same participant never appear in both training and test folds.

## Secondary Target

The main secondary target will focus on negative-feedback risk persistence:

```text
risk_persistence_after_negative_feedback
```

This analysis will subset current trials to negative feedback, such as `Loss-Error`, all losses, or all errors depending on final task coding.

The target asks:

```text
After negative feedback, does the participant continue making a risky choice on the next free-choice trial?
```

This is highly relevant to risk-taking, addiction, and maladaptive feedback-learning literature.

## Data Unit

Each row should represent one trial-to-next-free-choice transition:

```text
participant_id, current feedback trial t -> next free-choice trial t_next
```

Rows are trial-level, but independence is participant-level. All validation must therefore group by `participant_id`.

## Candidate Predictors

Only variables available at or before feedback trial `t` are allowed.

### Context Features

- block index
- trial index
- global trial index
- task phase
- current feedback condition
- current gain/loss status
- current correct/error status
- current selected-option context, if available before feedback evaluation

### Behavioral History Features

- previous risky choice
- previous feedback condition
- previous response time
- previous score change
- rolling risky-choice rate before or up to trial `t`
- rolling response-time mean before or up to trial `t`
- rolling response-time variability before or up to trial `t`
- recent gain/loss history
- recent correct/error history

### Feedback-Locked ERP Features

- current-trial `Fz_FRN`
- current-trial `FCz_FRN`
- current-trial `Cz_FRN`
- current-trial `Pz_P300`
- current-trial `POz_P300`
- current-trial `CPz_P300`
- frontal FRN summary features
- posterior P300 summary features
- FRN/P300 contrast features, if constructed without target leakage

## Forbidden Predictors

The following must not be used as predictors because they occur after the prediction point or directly encode the target:

- next-trial option values
- next-trial response time
- next-trial feedback
- next-trial score
- next-trial EEG
- next-trial correctness
- next-trial choice except as the target
- rolling features that include `t_next`
- participant ID as a model feature
- any column computed using future trials beyond `t`

## Feature Set Plan

The paper should compare nested feature sets rather than only reporting one high-performing model.

| Feature Set | Purpose |
| --- | --- |
| `context_only` | Task context and current feedback baseline |
| `history_only` | Behavioral-history baseline |
| `eeg_only` | Clean neural prediction test |
| `context_eeg` | Feedback condition plus ERP features |
| `history_context` | Strong non-EEG baseline |
| `history_context_eeg` | Main full model |
| `negative_feedback_eeg` | Secondary negative-feedback persistence model |

The main scientific claim should depend on whether ERP-containing models outperform comparable non-ERP baselines.

## Primary Machine Learning Model

The primary classifier should be Logistic Regression with balanced class weights.

Reasons:

- interpretable
- defensible for a small participant sample
- less overfit-prone than complex models
- suitable for coefficient inspection
- compatible with compact ERP/behavior feature sets

Recommended primary model:

```text
LogisticRegression(max_iter=3000, class_weight="balanced")
```

Exploratory models may include:

- Random Forest
- HistGradientBoostingClassifier

Deep learning, CNNs, and large FFT/wavelet feature expansions are not recommended for this dataset because only 39 participants are available.

## Validation Design

The primary validation must use participant-grouped cross-validation.

Recommended approach:

```text
GroupKFold, group = participant_id
```

This tests generalization to unseen participants and prevents trial-level leakage.

Random trial-level splits are not acceptable for journal-level claims because they allow participant-specific patterns to leak across train and test folds.

## Evaluation Metrics

Primary metric:

- balanced accuracy

Secondary metrics:

- ROC AUC
- macro F1
- sensitivity
- specificity
- calibration or Brier score, if useful

Report fold-level values, mean performance, uncertainty intervals, and permutation-test p-values.

## Permutation Testing

Permutation testing should be used to determine whether cross-validated performance exceeds chance.

Recommended tests:

- label permutation with participant-grouped CV retained
- within-participant label permutation as a trial-level relationship test
- delta permutation test comparing `history_context_eeg` against `history_context`

The key statistical question is not only whether the full model is above chance, but whether ERP features add incremental predictive value beyond behavioral history.

## Incremental Value Analysis

The central model comparison should be:

```text
history_context_eeg vs history_context
```

Primary incremental metric:

```text
Delta balanced accuracy = BA(history_context_eeg) - BA(history_context)
```

If ERP features improve balanced accuracy and the improvement is supported by permutation testing, the main conclusion can be framed as evidence that feedback-locked neural responses contain predictive information about subsequent risk adaptation.

## Classical Statistical Support

The ML analysis should be supported by interpretable statistical models.

Recommended mixed-effects model:

```text
next_free_choice_risky ~ feedback_condition + FRN + P300 + rolling_risk + trial_index + (1 | participant)
```

If feasible:

```text
next_free_choice_risky ~ feedback_condition * P300 + feedback_condition * FRN + rolling_risk + trial_index + (1 | participant)
```

Additional non-ML analyses:

- compare ERP amplitudes on trials followed by risky vs non-risky next choices
- compare ERP amplitudes after negative feedback followed by risk persistence vs risk reduction
- test whether P300/FRN effects are stronger after losses/errors than gains/correct outcomes

## Sensitivity Analyses

Minimum sensitivity analyses for journal-level reporting:

- exclude participant `1013`, who has known EEG/trigger QC concerns in the main project
- use only current free-choice trials
- use all feedback trials with a valid next free-choice trial
- analyze negative-feedback trials separately
- compare EEG-only, history-only, and full models
- remove direct current-choice predictors and test whether ERP features still help
- test chronotype moderation only as exploratory analysis

## Optional Chronotype Moderation

Chronotype should not be the primary target in this side project. However, it can be examined as an exploratory moderator:

```text
next_free_choice_risky ~ ERP + feedback + Chronotype + ERP:Chronotype + feedback:Chronotype + history + (1 | participant)
```

This would connect the side project back to the main chronotype project without duplicating its primary aim.

## Expected Outcomes

### Strong Result

- full model outperforms history-only baseline
- ERP-only model is above chance or near chance but informative
- P300/FRN features appear in top importance rankings
- negative-feedback subset shows stronger ERP contribution

Possible claim:

> Feedback-locked ERP responses predicted subsequent risky-choice adaptation beyond behavioral history, suggesting that neural feedback evaluation contributes to trial-to-trial risk regulation.

### Moderate Result

- full model is above chance
- ERP improves performance slightly
- EEG-only model is weak

Possible claim:

> Feedback-locked ERP features showed modest incremental value for predicting subsequent risk adaptation in a pilot sample.

### Weak Result

- history-only model predicts adaptation
- ERP adds little or no value

Possible claim:

> Subsequent risk adaptation was primarily explained by behavioral history rather than feedback-locked ERP features.

This outcome is still scientifically useful because it clarifies the limits of EEG-based prediction in this dataset.

## Journal-Level Framing

The paper should not be framed as a high-accuracy classifier paper. It should be framed as a cognitive-neuroscience ML study testing whether neural feedback evaluation predicts future risk adaptation.

Recommended abstract-level claim if results support it:

> In a pilot EEG decision-making dataset, feedback-locked ERP features were tested as predictors of subsequent risky-choice adaptation. Using participant-grouped cross-validation and permutation testing, we evaluated whether FRN and P300 responses predicted next free-choice risk behavior beyond behavioral history. The study provides a leakage-aware framework for testing neural markers of feedback-guided risk adaptation.

## Relationship To The Main Project

The main project remains focused on:

```text
Chronotype-related differences in feedback processing during risky decision-making.
```

This side project focuses on:

```text
Trial-level prediction of subsequent risk adaptation from feedback-locked ERP.
```

The side project should reuse generated clean/raw-derived data from the main project only as input. It should not modify the main `scripts/`, `docs/`, or `data/clean/` pipeline unless explicitly coordinated.

## Proposed Folder Layout

```text
side_projects/feedback_erp_risk_adaptation/
├── README.md
├── data/
├── docs/
├── notebooks/
├── reports/
└── scripts/
```

Generated side-project data and reports should remain local/ignored unless summarized in tracked documentation.

## Implementation Roadmap

1. Build a trial-to-next-free-choice adaptation dataset.
2. Create a leakage manifest documenting target timing and excluded columns.
3. Build feature packs: context-only, history-only, EEG-only, context+EEG, history+context, and full history+context+EEG.
4. Train primary Logistic Regression models with participant-grouped CV.
5. Run permutation tests for each feature set.
6. Test incremental ERP value over the history/context baseline.
7. Run negative-feedback persistence analysis.
8. Run participant `1013` sensitivity analysis.
9. Add mixed-effects statistical support.
10. Summarize findings in `docs/results.md` before considering manuscript writing.

## Current Decision

Primary planned target:

```text
next_free_choice_risky
```

Secondary planned target:

```text
risk_persistence_after_negative_feedback
```

Primary planned model comparison:

```text
history_context_eeg vs history_context
```
