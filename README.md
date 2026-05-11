# EEG + Behavioral Decision Modelling

A machine learning project exploring whether risky decision-making behavior can be predicted from behavioral, contextual, and EEG-derived features.

## Current Clean Workflow

The active workflow has been reset around leakage-aware modelling. Legacy scripts, old generated datasets, reports, logs, and saved models were moved to `archive/legacy/` so the working tree now focuses on the new pipeline.

Active locations:

- `data/raw/` contains original source files.
- `data/clean/` contains newly generated clean modelling datasets.
- `scripts/` contains only the new leakage-aware scripts.
- `reports/clean/` contains new model evaluation outputs.
- `models/clean/` is reserved for new saved models.

Build clean datasets:

```bash
python scripts/build_clean_risky_choice.py
python scripts/build_clean_chronotype.py
```

Build and evaluate literature-guided feature packs:

```bash
python scripts/run_literature_feature_packs.py
```

Permutation-test chronotype packs:

```bash
python scripts/run_chronotype_permutation_tests.py --permutations 200
```

Analyze which features drive the significant chronotype model:

```bash
python scripts/run_chronotype_importance.py --repeats 50
```

The literature-guided packs compare compact feature groups instead of one large mixed feature table:

- Risky choice: `value_only`, `history_only`, `value_history`, `prev_eeg`, `all_clean`
- Chronotype: `demo_only`, `behavior_core`, `frn_core`, `p300_core`, `compact_combined`, `all_literature`

Permutation testing compares observed balanced accuracy against shuffled-label performance. This is required for chronotype because the dataset has only 39 participants.

Feature importance uses held-out fold permutation importance on original engineered columns, so the output stays interpretable.

Train clean baselines:

```bash
python scripts/train_clean_baseline.py --data data/clean/risky_choice_prechoice.csv --target risky-choice --group-col participant_id
python scripts/train_clean_baseline.py --data data/clean/chronotype_participant.csv --target Chronotype --group-col ""
```

Important modelling rule: risky-choice features use same-trial pre-choice value/context only plus lagged previous-trial history. Same-trial outcome, correctness, score, response time, feedback, and EEG response features are excluded as predictors.

---

## 🎯 Project Objective

The original goal of this project was to:

> Predict **chronotype (morning vs evening)** using EEG (ERP) and behavioral data.

However, during experimentation, we identified:

- Weak predictive signal at participant-level for chronotype
- Risk of **data leakage** and artificial signals
- Better opportunity in **trial-level behavioural modelling**

### 🔄 Final Direction

The project evolved into:

> Predicting **risky decision-making (binary choice)** from:
- Value-based features (Option1, Option2)
- Behavioral dynamics (response time)
- Temporal patterns (trial progression)
- Decision history (previous choices)

---

## 📊 Dataset Overview

- ~156,000 rows (trial-level EEG + behavioral data)
- ~39 participants
- Each trial includes:
  - Two options (values, gains/losses)
  - Reaction time
  - Choice outcome (risky vs safe)
  - EEG features (e.g., mean amplitude)

### ⚠️ Key Data Issues Identified

We discovered several important problems:

- **Forced trials mixed with free-choice trials**
  → removed to avoid misleading learning

- **Trial-level duplication (EEG channels/windows)**
  → required aggregation

- **Near-constant features**
  → many columns had no variance

- **Leakage features**
  → e.g., `CorrectChoice`, `ChoiceMade`, `CurrentScore`

---

## 🧹 Data Processing Pipeline

### 1. Filtering
- Removed **forced trials**
- Kept only `"free"` decision trials

### 2. Aggregation
Converted EEG-expanded data → **one row per trial**

Grouped by:
# EEG + Behavioral Decision Modelling

A machine learning project exploring whether risky decision-making behavior can be predicted from behavioral, contextual, and EEG-derived features.

---

## 🎯 Project Objective

### Initial Goal
The project originally aimed to:

> Predict **chronotype (morning vs evening)** using EEG (ERP) and behavioral data.

### Why This Changed
During experimentation, several issues emerged:

- Weak and unstable predictive signal for chronotype
- High risk of **data leakage**
- Dataset structure not suitable for reliable participant-level prediction

### Final Direction
The project pivoted to a more realistic and meaningful objective:

> Predict **risky decision-making (binary choice)** at the trial level

This aligns better with:
- The structure of the dataset
- Cognitive neuroscience theory (decision-making dynamics)
- Practical ML feasibility

---

## 📊 Dataset Overview

- ~156,000 rows (raw trial-level EEG + behavioral data)
- ~39 participants
- Final modelling dataset: **~897 aggregated trials (free-only)**

Each trial includes:

- Option values (`Option1`, `Option2`)
- Reaction time (`ResponseTime`)
- Trial index (`Trial`)
- Risky choice label (`risky-choice`)
- EEG-derived features (e.g., `mean_amp`)

---

## ⚠️ Critical Data Issues Identified

### 1. Forced vs Free Trials
Some trials were **forced choices**, meaning:
- The participant had no real decision
- Including them would corrupt learning

✔️ Solution:
- Filtered dataset → **free trials only**

---

### 2. EEG Duplication Problem
Each trial appeared multiple times due to:
- Channels
- Time windows

✔️ Solution:
- Aggregated to **one row per trial**

---

### 3. Leakage Features
Several columns directly encode the outcome:

- `ChoiceMade`
- `CorrectChoice`
- `CurrentScore`

✔️ These were **removed** to prevent cheating models

---

### 4. Near-Constant / Useless Features
Many columns had:
- Almost no variance
- No predictive value

✔️ Automatically detected and dropped

---

## 🧹 Data Processing Pipeline

### Step 1 — Filter Free Trials
```python
df = df[df["forced and free risk trials"].str.lower() == "free"]
```

---

### Step 2 — Feature Engineering

#### Value-based features
```python
OptionDiff = Option1 - Option2
AbsOptionDiff = abs(Option1 - Option2)
ValueSum = Option1 + Option2
```

#### Behavioral features
```python
RT_log = log(ResponseTime)
RT_zscore = participant-normalized RT
```

#### History features
```python
PrevRisky = previous trial choice
PrevRT = previous reaction time
```

#### Interaction features
```python
OptionDiff_x_RTlog
OptionDiff_x_Trial
```

---

### Step 3 — Aggregation

From EEG-expanded → trial-level:

Grouped by:
- `participant_id`
- `Trial`

---

## 🤖 Modelling Approach

### Cross-validation
- **GroupKFold (n=5)**
- Prevents leakage across participants

---

### Models Used

- Logistic Regression
- Random Forest
- HistGradientBoosting (HGB)

---

## 📈 Feature Selection Strategy

We used multiple approaches:

### 1. ML Feature Advisor
Automatically evaluated:
- Mutual information
- Leakage
- Redundancy

Key findings:

✔️ Strongest signals:
- `OptionDiff`
- `ResponseTime`
- `RT_log`
- `Trial`

---

### 2. Feature Sweeps (Custom Pipeline)

We implemented:

- Single-feature evaluation
- Forward selection
- Custom combinations

---

### 🔍 Best Performing Feature Sets

#### Forward Selection Result
```
['Trial', 'OptionDiff', 'PrevRisky']
```

#### Improved Variant
```
['Trial', 'OptionDiff', 'PrevRisky', 'RT_log']
```

#### Temporal-only Insight
```
['Trial', 'PrevRT']
```

---

## 📊 Final Model Performance

Best results (approx):

| Model | Accuracy | Balanced Acc | ROC AUC |
|------|--------|-------------|--------|
| Logistic Regression | ~0.55 | ~0.55 | ~0.53 |
| Random Forest | ~0.57 | ~0.58 | ~0.56 |
| HGB (Best) | ~0.59–0.60 | ~0.57–0.58 | ~0.56 |

---

## 🧠 Key Insights

### 1. Decision Behaviour is Weakly Predictable
- Performance slightly above random (~0.5 baseline)
- Indicates:
  - High noise
  - Human decision variability

---

### 2. Value Difference Matters
- `OptionDiff` consistently contributes
- Larger differences → more predictable choices

---

### 3. Temporal Effects Exist
- `Trial` index has signal
- Suggests:
  - Learning
  - fatigue
  - adaptation

---

### 4. Behavioural History Helps
- `PrevRisky`, `PrevRT` improve performance
- Indicates sequential dependency

---

### 5. EEG Features Contribute Little
- `mean_amp`, `mean_amp_z` → near zero importance

Possible reasons:
- Too noisy
- Poor feature extraction
- Requires deeper signal processing

---

## ⚠️ Limitations

- Small dataset after aggregation (~897 rows)
- Limited participants (~39)
- No deep EEG feature engineering
- Trial-level randomness dominates signal

---

## 🚀 Future Improvements

### Data Level
- More participants
- Better EEG preprocessing (ERP windows, frequency bands)

### Feature Engineering
- Rolling averages (last N trials)
- Reward prediction error features
- Risk sensitivity metrics

### Modelling
- Sequence models (RNN / Transformer)
- Hierarchical models (participant + trial)

---

## 📁 Project Structure

```
chronotype_ml/
│
├── data/
│   └── merged/
│
├── scripts/
│   ├── ml_feature_advisor.py
│   ├── train_risky_choice.py
│
├── reports/
│   └── eeg_behav_trial_models/
│
└── README.md
```

---

## ✅ Conclusion

This project demonstrates:

- Real-world ML challenges (noise, leakage, weak signals)
- Importance of proper preprocessing
- Value of iterative feature engineering
- Limits of prediction in human decision-making data

> Final takeaway:
Even with strong engineering, **human behaviour is only partially predictable**.
