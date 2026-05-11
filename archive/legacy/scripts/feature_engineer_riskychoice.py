#!/usr/bin/env python3
import pandas as pd
import numpy as np

print("🔄 Loading dataset...")
df = pd.read_csv("data/merged/eeg_behav_trial_aligned.csv")

# Value-based features
df["OptionDiff"] = df["Option1"] - df["Option2"]
df["AbsOptionDiff"] = (df["Option1"] - df["Option2"]).abs()
df["ValueSum"] = df["Option1"] + df["Option2"]

# Behavioral dynamics
df["RT_log"] = np.log(df["ResponseTime"] + 1e-6)
df["RT_zscore"] = df.groupby("participant_id")["ResponseTime"].transform(
    lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1)
)

# EEG-normalized features (optional)
if "mean_amp" in df.columns:
    df["mean_amp_z"] = df.groupby("participant_id")["mean_amp"].transform(
        lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1)
    )

out_path = "data/merged/eeg_behav_trial_aligned_augmented.csv"
df.to_csv(out_path, index=False)
print(f"✅ Saved augmented dataset → {out_path}")