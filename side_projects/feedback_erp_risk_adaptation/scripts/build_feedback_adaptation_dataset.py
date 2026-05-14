#!/usr/bin/env python3
"""Build trial-to-next-free-choice adaptation datasets.

Each row represents a current feedback trial t and the participant's next
free-choice trial. Predictors are restricted to information available at or
before trial t; the target is whether the next free-choice trial is risky.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/processed/ml_ready_features.csv")
DEFAULT_OUTDIR = Path("side_projects/feedback_erp_risk_adaptation/data")
TARGET = "next_free_choice_risky"
ADAPTATION_TARGETS = [
    TARGET,
    "risk_switch",
    "risk_persistence_given_current_risky",
    "risk_initiation_given_current_safe",
]
NEGATIVE_FEEDBACK_SCENARIOS = {
    "loss_error": "FeedbackLossError == 1",
    "all_loss": "FeedbackLoss == 1",
    "all_error": "FeedbackError == 1",
    "loss_or_error": "FeedbackLoss == 1 or FeedbackError == 1",
}
BASE_ID_COLUMNS = [
    "participant_id",
    "Block",
    "Trial",
    "global_trial_index",
]
METADATA_COLUMNS = ["next_free_global_trial_index", "next_free_gap_trials", "next_free_is_immediate"]


def normalize_condition(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def infer_eeg_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith(("_FRN", "_P300")) and not c.endswith(("_FRN_pz", "_P300_pz", "_FRN_pc", "_P300_pc"))]


def infer_centered_eeg_columns(df: pd.DataFrame) -> list[str]:
    regional = ["FrontalFRN_mean_pz", "PosteriorP300_mean_pz", "FRN_x_LossError", "P300_x_LossError"]
    participant_z = [f"{c}_pz" for c in infer_eeg_columns(df)]
    return select_existing(df, regional + participant_z)


def select_existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def add_value_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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


def add_feedback_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    condition = out["feedback-condition"].map(normalize_condition)
    out["FeedbackLoss"] = condition.str.startswith("loss").astype(int)
    out["FeedbackGain"] = condition.str.startswith("gain").astype(int)
    out["FeedbackError"] = condition.str.endswith("error").astype(int)
    out["FeedbackCorrect"] = condition.str.endswith("correct").astype(int)
    out["FeedbackGainCorrect"] = condition.eq("gain_correct").astype(int)
    out["FeedbackGainError"] = condition.eq("gain_error").astype(int)
    out["FeedbackLossCorrect"] = condition.eq("loss_correct").astype(int)
    out["FeedbackLossError"] = condition.eq("loss_error").astype(int)
    return out


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["participant_id", "global_trial_index", "Block", "Trial"]).reset_index(drop=True)
    group = out.groupby("participant_id", sort=False)

    out["TrialInParticipant"] = group.cumcount() + 1
    max_trial = group["TrialInParticipant"].transform("max").replace(0, np.nan)
    out["TrialProgress"] = out["TrialInParticipant"] / max_trial

    out["CurrentRisky"] = pd.to_numeric(out["risky_label"], errors="coerce")
    out["CurrentRTLog"] = np.log(pd.to_numeric(out["ResponseTime"], errors="coerce") + 1e-6)
    prev_score = group["CurrentScore"].shift(1)
    out["CurrentScoreDelta"] = out["CurrentScore"] - prev_score

    lag_sources = [
        "risky_label",
        "ResponseTime",
        "CorrectChoice",
        "CurrentScore",
        "CurrentScoreDelta",
        "FeedbackLoss",
        "FeedbackError",
        "FeedbackGainCorrect",
        "FeedbackGainError",
        "FeedbackLossCorrect",
        "FeedbackLossError",
    ]
    for col in lag_sources:
        if col not in out.columns:
            continue
        new_col = "PrevRisky" if col == "risky_label" else f"Prev{col}"
        out[new_col] = group[col].shift(1)

    if "PrevResponseTime" in out.columns:
        out["PrevRTLog"] = np.log(pd.to_numeric(out["PrevResponseTime"], errors="coerce") + 1e-6)

    shifted_risky = group["risky_label"].shift(1)
    shifted_rt = group["ResponseTime"].shift(1)
    for window in [3, 5, 10]:
        out[f"RollingRiskyRate{window}"] = shifted_risky.groupby(out["participant_id"]).transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        out[f"RollingRTMean{window}"] = shifted_rt.groupby(out["participant_id"]).transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        out[f"RollingRTStd{window}"] = shifted_rt.groupby(out["participant_id"]).transform(
            lambda s: s.rolling(window=window, min_periods=2).std()
        )

    return out


def add_centered_eeg_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add within-participant ERP features for trial-level models."""
    out = df.copy()
    eeg_cols = infer_eeg_columns(out)
    group = out.groupby("participant_id", sort=False)
    for col in eeg_cols:
        mean = group[col].transform("mean")
        std = group[col].transform("std").replace(0, np.nan)
        out[f"{col}_pc"] = out[col] - mean
        out[f"{col}_pz"] = (out[col] - mean) / std

    frn_cols = select_existing(out, ["Fz_FRN_pz", "FCz_FRN_pz", "Cz_FRN_pz"])
    p300_cols = select_existing(out, ["Pz_P300_pz", "POz_P300_pz", "CPz_P300_pz"])
    if frn_cols:
        out["FrontalFRN_mean_pz"] = out[frn_cols].mean(axis=1)
    if p300_cols:
        out["PosteriorP300_mean_pz"] = out[p300_cols].mean(axis=1)
    if "FrontalFRN_mean_pz" in out.columns and "FeedbackLossError" in out.columns:
        out["FRN_x_LossError"] = out["FrontalFRN_mean_pz"] * out["FeedbackLossError"]
    if "PosteriorP300_mean_pz" in out.columns and "FeedbackLossError" in out.columns:
        out["P300_x_LossError"] = out["PosteriorP300_mean_pz"] * out["FeedbackLossError"]
    return out


def add_next_free_target(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, part in df.groupby("participant_id", sort=False):
        part = part.sort_values("global_trial_index").copy()
        positions = np.arange(len(part))
        free_mask = part["is_free_choice"].to_numpy(dtype=bool)
        free_positions = positions[free_mask]
        risky_values = part["risky_label"].to_numpy()
        global_trials = part["global_trial_index"].to_numpy()

        for pos in positions:
            next_idx = np.searchsorted(free_positions, pos + 1, side="left")
            if next_idx >= len(free_positions):
                continue
            next_pos = free_positions[next_idx]
            target_value = risky_values[next_pos]
            if pd.isna(target_value):
                continue
            row = part.iloc[pos].copy()
            row[TARGET] = int(target_value)
            row["next_free_global_trial_index"] = int(global_trials[next_pos])
            row["next_free_gap_trials"] = int(global_trials[next_pos] - global_trials[pos])
            row["next_free_is_immediate"] = int(global_trials[next_pos] - global_trials[pos] == 1)
            rows.append(row)

    if not rows:
        raise ValueError("No valid current-trial to next-free-choice rows were created.")
    return pd.DataFrame(rows).reset_index(drop=True)


def add_adaptation_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_switch"] = (out[TARGET] != out["CurrentRisky"]).astype(int)
    out["risk_persistence_given_current_risky"] = np.where(out["CurrentRisky"].eq(1), out[TARGET], np.nan)
    out["risk_initiation_given_current_safe"] = np.where(out["CurrentRisky"].eq(0), out[TARGET], np.nan)
    return out


def feature_packs(df: pd.DataFrame, target: str = TARGET) -> dict[str, list[str]]:
    ids = select_existing(df, BASE_ID_COLUMNS + [target])
    value_context = select_existing(df, [
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
        "is_current_free_choice",
    ])
    feedback_context = select_existing(df, [
        "FeedbackLoss",
        "FeedbackGain",
        "FeedbackError",
        "FeedbackCorrect",
        "FeedbackGainCorrect",
        "FeedbackGainError",
        "FeedbackLossCorrect",
        "FeedbackLossError",
        "trigger_behavior_match",
        "good_trial",
    ])
    behavior_state = select_existing(df, [
        "CurrentRisky",
        "ChoiceMade",
        "CorrectChoice",
        "ResponseTime",
        "CurrentRTLog",
        "CurrentScore",
        "CurrentScoreDelta",
        "PrevRisky",
        "PrevResponseTime",
        "PrevRTLog",
        "PrevCorrectChoice",
        "PrevCurrentScore",
        "PrevCurrentScoreDelta",
        "PrevFeedbackLoss",
        "PrevFeedbackError",
        "PrevFeedbackGainCorrect",
        "PrevFeedbackGainError",
        "PrevFeedbackLossCorrect",
        "PrevFeedbackLossError",
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
    lag_history = select_existing(df, [
        "PrevRisky",
        "PrevResponseTime",
        "PrevRTLog",
        "PrevCorrectChoice",
        "PrevCurrentScore",
        "PrevCurrentScoreDelta",
        "PrevFeedbackLoss",
        "PrevFeedbackError",
        "PrevFeedbackGainCorrect",
        "PrevFeedbackGainError",
        "PrevFeedbackLossCorrect",
        "PrevFeedbackLossError",
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
    eeg = infer_eeg_columns(df)
    centered_eeg = infer_centered_eeg_columns(df)

    packs = {
        "context_only": ids + value_context + feedback_context,
        "history_only": ids + behavior_state,
        "eeg_only": ids + eeg,
        "eeg_centered_only": ids + centered_eeg,
        "context_eeg": ids + value_context + feedback_context + eeg,
        "history_context": ids + value_context + feedback_context + behavior_state,
        "history_context_eeg": ids + value_context + feedback_context + behavior_state + eeg,
        "lag_history_context": ids + value_context + feedback_context + lag_history,
        "lag_history_context_eeg_raw": ids + value_context + feedback_context + lag_history + eeg,
        "lag_history_context_eeg_centered": ids + value_context + feedback_context + lag_history + centered_eeg,
    }
    return {name: list(dict.fromkeys(cols)) for name, cols in packs.items()}


def build(input_path: Path) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    df = pd.read_csv(input_path)
    required = {
        "participant_id",
        "Block",
        "Trial",
        "global_trial_index",
        "Option1",
        "Option2",
        "ChoiceMade",
        "CorrectChoice",
        "ResponseTime",
        "CurrentScore",
        "risky-choice",
        "feedback-condition",
        "forced and free risk trials",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "participant_id",
        "Block",
        "Trial",
        "global_trial_index",
        "Option1",
        "Option2",
        "ChoiceMade",
        "CorrectChoice",
        "ResponseTime",
        "CurrentScore",
        "risky-choice",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["participant_id"].notna()].copy()
    df["risky_label"] = pd.to_numeric(df["risky-choice"], errors="coerce")
    df["is_free_choice"] = df["forced and free risk trials"].astype(str).str.lower().eq("free")
    df["is_current_free_choice"] = df["is_free_choice"].astype(int)

    df = add_value_features(df)
    df = add_feedback_features(df)
    df = add_history_features(df)
    df = add_centered_eeg_features(df)
    out = add_next_free_target(df)
    out = add_adaptation_targets(out)

    packs = feature_packs(out, TARGET)
    target_packs = {target: feature_packs(out, target) for target in ADAPTATION_TARGETS}
    scenarios = {}
    for scenario, query in NEGATIVE_FEEDBACK_SCENARIOS.items():
        subset = out.query(query).copy()
        scenarios[scenario] = {
            "query": query,
            "rows": int(subset.shape[0]),
            "participants": int(subset["participant_id"].nunique()),
            "target_counts": subset[TARGET].value_counts(dropna=False).sort_index().astype(int).to_dict(),
            "current_trial_rows_by_type": subset["is_current_free_choice"].value_counts(dropna=False).sort_index().astype(int).to_dict(),
        }

    manifest = {
        "input": str(input_path),
        "target": TARGET,
        "adaptation_targets": {
            target: {
                "rows_non_missing": int(out[target].notna().sum()),
                "target_counts": out[target].dropna().value_counts().sort_index().astype(int).to_dict(),
            }
            for target in ADAPTATION_TARGETS
        },
        "rows": int(out.shape[0]),
        "participants": int(out["participant_id"].nunique()),
        "target_counts": out[TARGET].value_counts(dropna=False).sort_index().astype(int).to_dict(),
        "current_trial_rows_by_type": out["is_current_free_choice"].value_counts(dropna=False).sort_index().astype(int).to_dict(),
        "next_free_gap_trials": out["next_free_gap_trials"].describe().to_dict(),
        "eeg_columns": infer_eeg_columns(out),
        "centered_eeg_columns": infer_centered_eeg_columns(out),
        "metadata_columns_kept_only_in_full_table": select_existing(out, METADATA_COLUMNS),
        "negative_feedback_scenarios": scenarios,
        "feature_packs": {name: {"columns": cols, "n_columns": len(cols), "n_features_excluding_ids_target": len([c for c in cols if c not in BASE_ID_COLUMNS + [TARGET]])} for name, cols in packs.items()},
        "target_feature_packs": {
            target: {name: {"n_columns": len(cols), "n_features_excluding_ids_target": len([c for c in cols if c not in BASE_ID_COLUMNS + [target]])} for name, cols in packs_for_target.items()}
            for target, packs_for_target in target_packs.items()
        },
        "leakage_rule": "Predictors are from current feedback trial t or earlier; all next-free-trial variables except the target/index/gap metadata are excluded.",
        "forbidden_predictors": [
            "next trial option values",
            "next trial response time",
            "next trial feedback",
            "next trial score",
            "next trial EEG",
            "participant_id as model feature",
            "rolling features that include the next free-choice trial",
        ],
    }
    return out, target_packs, manifest


def write_outputs(df: pd.DataFrame, target_packs: dict[str, dict[str, list[str]]], manifest: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "feedback_adaptation_full.csv", index=False)
    packs = target_packs[TARGET]
    for name, cols in packs.items():
        df[cols].to_csv(outdir / f"feedback_adaptation_{name}.csv", index=False)
    for scenario, query in NEGATIVE_FEEDBACK_SCENARIOS.items():
        subset = df.query(query).copy()
        subset.to_csv(outdir / f"feedback_adaptation_{scenario}_full.csv", index=False)
        for name, cols in packs.items():
            subset[cols].to_csv(outdir / f"feedback_adaptation_{scenario}_{name}.csv", index=False)

    for target, packs_for_target in target_packs.items():
        if target == TARGET:
            continue
        for name, cols in packs_for_target.items():
            df[cols].to_csv(outdir / f"feedback_adaptation_{target}_{name}.csv", index=False)
        for scenario, query in NEGATIVE_FEEDBACK_SCENARIOS.items():
            subset = df.query(query).copy()
            for name, cols in packs_for_target.items():
                subset[cols].to_csv(outdir / f"feedback_adaptation_{scenario}_{target}_{name}.csv", index=False)
    (outdir / "feedback_adaptation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feedback-to-next-risk adaptation datasets.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    df, target_packs, manifest = build(Path(args.input))
    write_outputs(df, target_packs, manifest, Path(args.outdir))

    printable = {k: v for k, v in manifest.items() if k != "feature_packs"}
    printable["feature_packs"] = {k: v["n_columns"] for k, v in manifest["feature_packs"].items()}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
