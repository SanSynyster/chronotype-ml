#!/usr/bin/env python3
"""Build a leakage-aware risky-choice modelling table.

The output uses only same-trial pre-choice value/context features plus lagged
history features from previous trials. Same-trial choice, correctness, score,
feedback, response time, and EEG response features are not used as predictors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUTS = [
    Path("data/processed/ml_ready_features.csv"),
    Path("archive/legacy/data_processed/ml_ready_features.csv"),
]

TARGET = "risky-choice"
LEAKAGE_COLUMNS = {
    "ChoiceMade",
    "CorrectChoice",
    "CurrentScore",
    "ResponseTime",
    "behav_valence",
    "feedback-condition",
    "trigger_numeric",
    "trigger_val",
    TARGET,
}


def resolve_input(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for path in DEFAULT_INPUTS:
        if path.exists():
            return path
    raise FileNotFoundError("No default input found. Pass --input explicitly.")


def zscore_safe(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return series * 0
    return (series - series.mean()) / std


def infer_eeg_columns(df: pd.DataFrame) -> list[str]:
    known_suffixes = ("_FRN", "_P300")
    return [c for c in df.columns if c.endswith(known_suffixes)]


def normalize_condition(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def add_value_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Option1", "Option2", "Block", "Trial", "global_trial_index"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["OptionDiff"] = out["Option1"] - out["Option2"]
    out["AbsOptionDiff"] = out["OptionDiff"].abs()
    out["ValueSum"] = out["Option1"] + out["Option2"]
    out["ValueMax"] = out[["Option1", "Option2"]].max(axis=1)
    out["ValueMin"] = out[["Option1", "Option2"]].min(axis=1)
    denom = out["Option2"].replace(0, np.nan)
    out["OptionRatio"] = (out["Option1"] / denom).replace([np.inf, -np.inf], np.nan)
    out["AbsOptionRatio"] = out["OptionRatio"].abs()
    out["IsMixedSigns"] = ((out["Option1"] < 0) != (out["Option2"] < 0)).astype(int)
    out["BothPositive"] = ((out["Option1"] > 0) & (out["Option2"] > 0)).astype(int)
    out["BothNegative"] = ((out["Option1"] < 0) & (out["Option2"] < 0)).astype(int)
    return out


def add_history_features(df: pd.DataFrame, include_prev_eeg: bool) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out = out.sort_values(["participant_id", "global_trial_index", "Block", "Trial"], na_position="last")
    g = out.groupby("participant_id", sort=False)

    out["TrialInParticipant"] = g.cumcount() + 1
    max_trial = g["TrialInParticipant"].transform("max").replace(0, np.nan)
    out["TrialProgress"] = out["TrialInParticipant"] / max_trial

    lag_sources = ["risky_label", "OptionDiff", "AbsOptionDiff", "ValueSum"]
    lag_sources += [c for c in ["ResponseTime", "CorrectChoice", "CurrentScore"] if c in out.columns]

    created = []
    for col in lag_sources:
        if col not in out.columns:
            continue
        new_col = "PrevRisky" if col == "risky_label" else f"Prev{col}"
        out[new_col] = g[col].shift(1)
        created.append(new_col)

    if "ResponseTime" in out.columns:
        out["PrevRTLog"] = np.log(pd.to_numeric(out["PrevResponseTime"], errors="coerce") + 1e-6)
        created.append("PrevRTLog")

    if "CurrentScore" in out.columns:
        prev_score = g["CurrentScore"].shift(1)
        prev_prev_score = g["CurrentScore"].shift(2)
        out["PrevScoreDelta"] = prev_score - prev_prev_score
        created.append("PrevScoreDelta")

    shifted_risky = g["risky_label"].shift(1)
    for window in [3, 5, 10]:
        col = f"RollingRiskyRate{window}"
        out[col] = shifted_risky.groupby(out["participant_id"]).transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        created.append(col)

    if "ResponseTime" in out.columns:
        shifted_rt = g["ResponseTime"].shift(1)
        for window in [3, 5, 10]:
            mean_col = f"RollingRTMean{window}"
            std_col = f"RollingRTStd{window}"
            out[mean_col] = shifted_rt.groupby(out["participant_id"]).transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
            out[std_col] = shifted_rt.groupby(out["participant_id"]).transform(
                lambda s: s.rolling(window=window, min_periods=2).std()
            )
            created.extend([mean_col, std_col])

    if "behav_valence" in out.columns:
        out["PrevFeedbackGain"] = g["behav_valence"].shift(1).astype(str).str.lower().eq("gain").astype(float)
        created.append("PrevFeedbackGain")

    if "feedback-condition" in out.columns:
        prev_condition = g["feedback-condition"].shift(1).map(normalize_condition)
        out["PrevFeedbackLoss"] = prev_condition.str.startswith("loss").astype(float)
        out["PrevFeedbackError"] = prev_condition.str.endswith("error").astype(float)
        out["PrevGainCorrect"] = prev_condition.eq("gain_correct").astype(float)
        out["PrevGainError"] = prev_condition.eq("gain_error").astype(float)
        out["PrevLossCorrect"] = prev_condition.eq("loss_correct").astype(float)
        out["PrevLossError"] = prev_condition.eq("loss_error").astype(float)
        created.extend(["PrevFeedbackLoss", "PrevFeedbackError", "PrevGainCorrect", "PrevGainError", "PrevLossCorrect", "PrevLossError"])

    if include_prev_eeg:
        for col in infer_eeg_columns(out):
            new_col = f"Prev{col}"
            out[new_col] = g[col].shift(1)
            created.append(new_col)

    return out, created


def select_existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def feature_packs(df: pd.DataFrame) -> dict[str, list[str]]:
    id_target = ["participant_id", "Block", "Trial", "global_trial_index", TARGET]
    current_value = select_existing(df, [
        "TrialInParticipant",
        "TrialProgress",
        "Option1",
        "Option2",
        "OptionDiff",
        "AbsOptionDiff",
        "ValueSum",
        "ValueMax",
        "ValueMin",
        "OptionRatio",
        "AbsOptionRatio",
        "IsMixedSigns",
        "BothPositive",
        "BothNegative",
    ])
    history = select_existing(df, [
        "PrevRisky",
        "PrevOptionDiff",
        "PrevAbsOptionDiff",
        "PrevValueSum",
        "PrevResponseTime",
        "PrevRTLog",
        "PrevCorrectChoice",
        "PrevCurrentScore",
        "PrevScoreDelta",
        "PrevFeedbackGain",
        "PrevFeedbackLoss",
        "PrevFeedbackError",
        "PrevGainCorrect",
        "PrevGainError",
        "PrevLossCorrect",
        "PrevLossError",
        "RollingRiskyRate3",
        "RollingRiskyRate5",
        "RollingRiskyRate10",
        "RollingRTMean3",
        "RollingRTStd3",
        "RollingRTMean5",
        "RollingRTStd5",
        "RollingRTMean10",
        "RollingRTStd10",
    ])
    prev_eeg = [c for c in df.columns if c.startswith("Prev") and c.endswith(("_FRN", "_P300"))]
    packs = {
        "value_only": id_target + current_value,
        "history_only": id_target + history,
        "value_history": id_target + current_value + history,
        "prev_eeg": id_target + current_value + history + prev_eeg,
        "all_clean": list(df.columns),
    }
    return {name: list(dict.fromkeys(cols)) for name, cols in packs.items()}


def build_dataset(input_path: Path, include_prev_eeg: bool) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(input_path)
    required = {"participant_id", "Option1", "Option2", TARGET}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "forced and free risk trials" in df.columns:
        df = df[df["forced and free risk trials"].astype(str).str.lower().eq("free")].copy()

    df["risky_label"] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df["risky_label"].isin([0, 1])].copy()

    if "global_trial_index" not in df.columns:
        sort_cols = [c for c in ["participant_id", "Block", "Trial"] if c in df.columns]
        df = df.sort_values(sort_cols)
        df["global_trial_index"] = df.groupby("participant_id").cumcount() + 1

    df = add_value_features(df)
    df, history_cols = add_history_features(df, include_prev_eeg=include_prev_eeg)

    same_trial_features = [
        "participant_id",
        "Block",
        "Trial",
        "global_trial_index",
        "TrialInParticipant",
        "TrialProgress",
        "Option1",
        "Option2",
        "OptionDiff",
        "AbsOptionDiff",
        "ValueSum",
        "ValueMax",
        "ValueMin",
        "OptionRatio",
        "AbsOptionRatio",
        "IsMixedSigns",
        "BothPositive",
        "BothNegative",
    ]
    same_trial_features = [c for c in same_trial_features if c in df.columns]
    feature_cols = same_trial_features + history_cols
    out = df[feature_cols + ["risky_label"]].copy()
    out = out.rename(columns={"risky_label": TARGET})

    leaked = sorted(set(out.columns) & (LEAKAGE_COLUMNS - {TARGET}))
    if leaked:
        raise AssertionError(f"Leakage columns reached output: {leaked}")

    manifest = {
        "input": str(input_path),
        "rows": int(out.shape[0]),
        "columns": int(out.shape[1]),
        "target": TARGET,
        "include_prev_eeg": include_prev_eeg,
        "same_trial_rule": "pre-choice value/context only",
        "excluded_same_trial_columns": sorted(LEAKAGE_COLUMNS - {TARGET}),
        "feature_columns": [c for c in out.columns if c != TARGET],
    }
    return out, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean risky-choice features.")
    parser.add_argument("--input", default=None, help="Input ML-ready trial CSV.")
    parser.add_argument("--out", default="data/clean/risky_choice_prechoice.csv")
    parser.add_argument("--manifest", default="data/clean/risky_choice_prechoice_manifest.json")
    parser.add_argument("--include-prev-eeg", action="store_true", help="Add previous-trial EEG lags as predictors.")
    parser.add_argument("--write-packs", action="store_true", help="Also write clean risky-choice feature-pack CSVs.")
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    out, manifest = build_dataset(input_path, include_prev_eeg=args.include_prev_eeg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.write_packs:
        for pack_name, cols in feature_packs(out).items():
            pack_path = out_path.with_name(f"risky_choice_{pack_name}.csv")
            out[cols].to_csv(pack_path, index=False)
            print(f"Wrote {pack_path} with shape {out[cols].shape}")

    print(f"Wrote {out_path} with shape {out.shape}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
