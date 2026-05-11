#!/usr/bin/env python3
"""Build a participant-level chronotype modelling table.

This script aggregates trial-level behaviour and EEG features to one row per
participant. It avoids trial-level row inflation for chronotype prediction.
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


def infer_eeg_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith(("_FRN", "_P300"))]


def normalize_condition(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def mean_numeric(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns or frame.empty:
        return np.nan
    return pd.to_numeric(frame[col], errors="coerce").mean()


def slope_or_nan(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or y[mask].nunique() < 2:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def normalize_chronotype(series: pd.Series) -> pd.Series:
    labels = series.astype(str).str.strip().str.lower()
    labels = labels.replace({
        "e": "Evening",
        "evening": "Evening",
        "eveningness": "Evening",
        "m": "Morning",
        "morning": "Morning",
        "morningness": "Morning",
    })
    labels = labels.where(labels.isin(["Evening", "Morning"]), series)
    return labels


def add_trial_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Option1", "Option2", "ResponseTime", "Trial", "Block", "global_trial_index", "risky-choice"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    sort_cols = [c for c in ["participant_id", "global_trial_index", "Block", "Trial"] if c in out.columns]
    if sort_cols and "feedback-condition" in out.columns:
        out = out.sort_values(sort_cols, na_position="last").copy()
        out["prev_feedback_condition"] = out.groupby("participant_id")["feedback-condition"].shift(1)
    if {"Option1", "Option2"}.issubset(out.columns):
        out["OptionDiff"] = out["Option1"] - out["Option2"]
        out["AbsOptionDiff"] = out["OptionDiff"].abs()
        out["ValueSum"] = out["Option1"] + out["Option2"]
    return out


def build_participant_row(pid: object, g: pd.DataFrame, eeg_cols: list[str]) -> dict:
    row: dict[str, object] = {"participant_id": pid}
    row["Chronotype"] = normalize_chronotype(g["Chronotype"]).dropna().mode().iloc[0]

    if "Age" in g.columns:
        row["Age"] = pd.to_numeric(g["Age"], errors="coerce").dropna().median()
    if "Gender" in g.columns:
        non_missing = g["Gender"].dropna()
        row["Gender"] = non_missing.mode().iloc[0] if not non_missing.empty else np.nan

    row["n_trials"] = int(g.shape[0])
    if "forced and free risk trials" in g.columns:
        free = g[g["forced and free risk trials"].astype(str).str.lower().eq("free")]
    else:
        free = g
    row["n_free_trials"] = int(free.shape[0])

    for frame_name, frame in [("all", g), ("free", free)]:
        if frame.empty:
            continue
        if "risky-choice" in frame.columns:
            row[f"{frame_name}_risky_rate"] = pd.to_numeric(frame["risky-choice"], errors="coerce").mean()
        if "ResponseTime" in frame.columns:
            rt = pd.to_numeric(frame["ResponseTime"], errors="coerce")
            row[f"{frame_name}_rt_mean"] = rt.mean()
            row[f"{frame_name}_rt_std"] = rt.std()
            row[f"{frame_name}_rt_median"] = rt.median()
        for col in ["OptionDiff", "AbsOptionDiff", "ValueSum"]:
            if col in frame.columns:
                vals = pd.to_numeric(frame[col], errors="coerce")
                row[f"{frame_name}_{col}_mean"] = vals.mean()
                row[f"{frame_name}_{col}_std"] = vals.std()

    if "prev_feedback_condition" in g.columns:
        cond = g["prev_feedback_condition"].map(normalize_condition)
        condition_names = ["gain_correct", "gain_error", "loss_correct", "loss_error"]
        free_cond = cond.loc[free.index] if not free.empty else pd.Series(dtype=object)
        for condition in condition_names:
            frame = g.loc[cond.eq(condition)]
            free_frame = free.loc[free_cond.eq(condition)] if not free.empty else free
            row[f"{condition}_n"] = int(frame.shape[0])
            row[f"{condition}_free_n"] = int(free_frame.shape[0])
            row[f"{condition}_risky_rate"] = mean_numeric(free_frame, "risky-choice")
            row[f"{condition}_rt_mean"] = mean_numeric(frame, "ResponseTime")

        gain_error_rt = row.get("gain_error_rt_mean", np.nan)
        loss_error_rt = row.get("loss_error_rt_mean", np.nan)
        gain_correct_rt = row.get("gain_correct_rt_mean", np.nan)
        loss_correct_rt = row.get("loss_correct_rt_mean", np.nan)
        row["post_error_slowing"] = np.nanmean([gain_error_rt, loss_error_rt]) - np.nanmean([gain_correct_rt, loss_correct_rt])

        row["risk_after_error_minus_correct"] = np.nanmean([
            row.get("gain_error_risky_rate", np.nan),
            row.get("loss_error_risky_rate", np.nan),
        ]) - np.nanmean([
            row.get("gain_correct_risky_rate", np.nan),
            row.get("loss_correct_risky_rate", np.nan),
        ])

        row["risk_after_loss_error_minus_gain_correct"] = row.get("loss_error_risky_rate", np.nan) - row.get("gain_correct_risky_rate", np.nan)

    order_col = "global_trial_index" if "global_trial_index" in g.columns else "Trial"
    if order_col in g.columns and "risky-choice" in g.columns:
        ordered = g.sort_values(order_col)
        row["risky_slope"] = slope_or_nan(ordered[order_col], ordered["risky-choice"])
        n = len(ordered)
        if n >= 4:
            early = ordered.iloc[: max(1, n // 3)]
            late = ordered.iloc[-max(1, n // 3):]
            row["risky_late_minus_early"] = pd.to_numeric(late["risky-choice"], errors="coerce").mean() - pd.to_numeric(early["risky-choice"], errors="coerce").mean()

    if order_col in g.columns and "ResponseTime" in g.columns:
        ordered = g.sort_values(order_col)
        row["rt_slope"] = slope_or_nan(ordered[order_col], ordered["ResponseTime"])

    for col in eeg_cols:
        vals = pd.to_numeric(g[col], errors="coerce")
        row[f"{col}_mean"] = vals.mean()
        row[f"{col}_std"] = vals.std()

    if "behav_valence" in g.columns:
        for col in eeg_cols:
            gain = pd.to_numeric(g.loc[g["behav_valence"].astype(str).str.lower().eq("gain"), col], errors="coerce").mean()
            loss = pd.to_numeric(g.loc[g["behav_valence"].astype(str).str.lower().eq("loss"), col], errors="coerce").mean()
            row[f"{col}_gain_mean"] = gain
            row[f"{col}_loss_mean"] = loss
            row[f"{col}_loss_minus_gain"] = loss - gain

    if "feedback-condition" in g.columns:
        cond = g["feedback-condition"].map(normalize_condition)
        for col in eeg_cols:
            for condition in ["gain_correct", "gain_error", "loss_correct", "loss_error"]:
                row[f"{col}_{condition}_mean"] = mean_numeric(g.loc[cond.eq(condition)], col)
            row[f"{col}_error_minus_correct"] = np.nanmean([
                row.get(f"{col}_gain_error_mean", np.nan),
                row.get(f"{col}_loss_error_mean", np.nan),
            ]) - np.nanmean([
                row.get(f"{col}_gain_correct_mean", np.nan),
                row.get(f"{col}_loss_correct_mean", np.nan),
            ])
            row[f"{col}_loss_error_minus_gain_correct"] = row.get(f"{col}_loss_error_mean", np.nan) - row.get(f"{col}_gain_correct_mean", np.nan)

    frn_cols = [c for c in eeg_cols if c.endswith("_FRN")]
    p300_cols = [c for c in eeg_cols if c.endswith("_P300")]
    if frn_cols:
        row["frn_global_mean"] = pd.to_numeric(g[frn_cols].stack(), errors="coerce").mean()
    if p300_cols:
        row["p300_global_mean"] = pd.to_numeric(g[p300_cols].stack(), errors="coerce").mean()

    frontal = [c for c in eeg_cols if c.startswith(("Fz_", "FCz_", "FC1_", "FC2_"))]
    parietal = [c for c in eeg_cols if c.startswith(("Pz_", "POz_", "CPz_"))]
    if frontal:
        row["frontal_eeg_mean"] = pd.to_numeric(g[frontal].stack(), errors="coerce").mean()
    if parietal:
        row["parietal_eeg_mean"] = pd.to_numeric(g[parietal].stack(), errors="coerce").mean()

    return row


def select_existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def feature_packs(df: pd.DataFrame) -> dict[str, list[str]]:
    id_target = ["participant_id", "Chronotype"]
    demo = select_existing(df, ["Age", "Gender"])
    behavior_core = select_existing(df, [
        "free_risky_rate",
        "free_rt_mean",
        "free_rt_std",
        "gain_correct_risky_rate",
        "gain_error_risky_rate",
        "loss_correct_risky_rate",
        "loss_error_risky_rate",
        "gain_correct_rt_mean",
        "gain_error_rt_mean",
        "loss_correct_rt_mean",
        "loss_error_rt_mean",
        "post_error_slowing",
        "risk_after_error_minus_correct",
        "risk_after_loss_error_minus_gain_correct",
        "risky_slope",
        "risky_late_minus_early",
        "rt_slope",
    ])
    frn_core = select_existing(df, [
        "Fz_FRN_mean",
        "FCz_FRN_mean",
        "Cz_FRN_mean",
        "Fz_FRN_loss_minus_gain",
        "FCz_FRN_loss_minus_gain",
        "Cz_FRN_loss_minus_gain",
        "Fz_FRN_error_minus_correct",
        "FCz_FRN_error_minus_correct",
        "Cz_FRN_error_minus_correct",
        "Fz_FRN_loss_error_minus_gain_correct",
        "FCz_FRN_loss_error_minus_gain_correct",
        "Cz_FRN_loss_error_minus_gain_correct",
        "frn_global_mean",
        "frontal_eeg_mean",
    ])
    p300_core = select_existing(df, [
        "Pz_P300_mean",
        "CPz_P300_mean",
        "POz_P300_mean",
        "Pz_P300_loss_minus_gain",
        "CPz_P300_loss_minus_gain",
        "POz_P300_loss_minus_gain",
        "Pz_P300_error_minus_correct",
        "CPz_P300_error_minus_correct",
        "POz_P300_error_minus_correct",
        "Pz_P300_gain_correct_mean",
        "CPz_P300_gain_correct_mean",
        "POz_P300_gain_correct_mean",
        "p300_global_mean",
        "parietal_eeg_mean",
    ])
    packs = {
        "demo_only": id_target + demo,
        "behavior_core": id_target + demo + behavior_core,
        "frn_core": id_target + demo + frn_core,
        "p300_core": id_target + demo + p300_core,
        "compact_combined": id_target + demo + behavior_core + frn_core + p300_core,
        "all_literature": list(df.columns),
    }
    return {name: list(dict.fromkeys(cols)) for name, cols in packs.items()}


def build_dataset(input_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(input_path)
    required = {"participant_id", "Chronotype"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = add_trial_features(df)
    eeg_cols = infer_eeg_columns(df)
    rows = [build_participant_row(pid, g, eeg_cols) for pid, g in df.groupby("participant_id", sort=True)]
    out = pd.DataFrame(rows)
    out = out[out["Chronotype"].isin(["Evening", "Morning"])].reset_index(drop=True)

    manifest = {
        "input": str(input_path),
        "rows": int(out.shape[0]),
        "columns": int(out.shape[1]),
        "target": "Chronotype",
        "unit_of_analysis": "participant",
        "behavior_condition_rule": "Behavioural adaptation features use previous-trial feedback-condition; ERP condition contrasts use current trial feedback-condition.",
        "eeg_columns_detected": eeg_cols,
        "feature_columns": [c for c in out.columns if c != "Chronotype"],
    }
    return out, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean participant-level chronotype features.")
    parser.add_argument("--input", default=None, help="Input ML-ready trial CSV.")
    parser.add_argument("--out", default="data/clean/chronotype_participant.csv")
    parser.add_argument("--manifest", default="data/clean/chronotype_participant_manifest.json")
    parser.add_argument("--write-packs", action="store_true", help="Also write literature-guided compact feature-pack CSVs.")
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    out, manifest = build_dataset(input_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    manifest_path = Path(args.manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.write_packs:
        for pack_name, cols in feature_packs(out).items():
            pack_path = out_path.with_name(f"chronotype_{pack_name}.csv")
            out[cols].to_csv(pack_path, index=False)
            print(f"Wrote {pack_path} with shape {out[cols].shape}")

    print(f"Wrote {out_path} with shape {out.shape}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
