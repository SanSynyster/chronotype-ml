#!/usr/bin/env python3
"""Authoritative rebuild from participant-ID keyed raw files.

This script intentionally does not read data/_outdated_raw, data/processed, or
data/clean inputs. It regenerates the active processed/clean tables from the
participant key, behavioural workbook, and condition-averaged ERP workbooks.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = ["Fz", "FC1", "FC2", "Cz", "Pz", "POz", "FCz"]
CONTRAST_CHANNELS = ["Fz", "FCz", "Cz", "Pz", "POz"]
CONDITIONS = ["gain_correct", "gain_error", "loss_correct", "loss_error"]


def norm_col(value: object) -> str:
    return str(value).strip()


def norm_chronotype(value: object) -> str | float:
    label = str(value).strip().lower()
    if label in {"morning", "m", "morningness"}:
        return "Morning"
    if label in {"evening", "e", "eveningness"}:
        return "Evening"
    if label in {"inter", "intermediate", "i"}:
        return "Intermediate"
    return np.nan


def norm_gender(value: object) -> str | float:
    label = str(value).strip().lower()
    if label in {"m", "male"}:
        return "M"
    if label in {"w", "f", "female", "woman"}:
        return "F"
    return np.nan


def write_conflicts(conflicts: pd.DataFrame, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    conflicts.to_csv(outdir / f"{name}.csv", index=False)
    md = [f"# Integrity Failure: {name}", "", "The authoritative rebuild stopped because a required join/check failed.", ""]
    md.append(conflicts.to_markdown(index=False))
    (outdir / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def fail_if(condition: bool, conflicts: pd.DataFrame, outdir: Path, name: str) -> None:
    if condition:
        write_conflicts(conflicts, outdir, name)
        raise SystemExit(f"Integrity failure: wrote {outdir / (name + '.csv')}")


def load_master(key_path: Path, outdir: Path) -> pd.DataFrame:
    key = pd.read_csv(key_path)
    key.columns = [norm_col(c) for c in key.columns]
    required = {"ERPset", "meq", "MCTQ", "minutes", "choronotype", "gender", "electro data availability", "age", "participant id"}
    missing = sorted(required - set(key.columns))
    fail_if(bool(missing), pd.DataFrame({"missing_column": missing}), outdir, "key_missing_columns")

    key["participant_id"] = pd.to_numeric(key["participant id"], errors="coerce").astype("Int64")
    dup = key[key["participant_id"].duplicated(keep=False) | key["participant_id"].isna()]
    fail_if(not dup.empty, dup, outdir, "key_duplicate_or_missing_ids")

    master = pd.DataFrame({
        "participant_id": key["participant_id"].astype(int),
        "ERPset": key["ERPset"].astype(str).str.strip(),
        "MEQ": pd.to_numeric(key["meq"], errors="coerce"),
        "meq": pd.to_numeric(key["meq"], errors="coerce"),
        "MCTQ": key["MCTQ"].astype(str).str.strip(),
        "mctq_minutes": pd.to_numeric(key["minutes"], errors="coerce"),
        "Chronotype": key["choronotype"].map(norm_chronotype),
        "is_intermediate": key["choronotype"].map(norm_chronotype).eq("Intermediate"),
        "Gender": key["gender"].map(norm_gender),
        "gender_key_raw": key["gender"],
        "Age": pd.to_numeric(key["age"], errors="coerce"),
        "has_eeg": key["electro data availability"].astype(str).str.strip().str.lower().eq("yes"),
    })
    bad = master[master[["MEQ", "Chronotype", "Gender", "Age"]].isna().any(axis=1)]
    fail_if(not bad.empty, bad, outdir, "key_invalid_values")
    return master.sort_values("participant_id").reset_index(drop=True)


def load_behavior(path: Path, master: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    beh = pd.read_excel(path)
    beh.columns = [norm_col(c) for c in beh.columns]
    required = {"UserID", "Block", "Trial", "Option1", "Option2", "ChoiceMade", "CorrectChoice", "ResponseTime", "CurrentScore", "Age", "Gender", "risky choice", "feedback"}
    missing = sorted(required - set(beh.columns))
    fail_if(bool(missing), pd.DataFrame({"missing_column": missing}), outdir, "behavior_missing_columns")

    beh["participant_id"] = pd.to_numeric(beh["UserID"], errors="coerce").astype("Int64")
    bad_id = beh[beh["participant_id"].isna()]
    fail_if(not bad_id.empty, bad_id.head(100), outdir, "behavior_missing_ids")

    beh_ids = pd.DataFrame({"participant_id": sorted(beh["participant_id"].astype(int).unique())})
    missing_key = beh_ids[~beh_ids["participant_id"].isin(master["participant_id"])]
    fail_if(not missing_key.empty, missing_key, outdir, "behavior_ids_missing_from_key")

    analysis_ids = master.loc[master["has_eeg"] & master["Chronotype"].isin(["Morning", "Evening"]), ["participant_id"]]
    missing_beh = analysis_ids[~analysis_ids["participant_id"].isin(beh_ids["participant_id"])]
    fail_if(not missing_beh.empty, missing_beh, outdir, "analysis_ids_missing_from_behavior")

    check = beh.drop_duplicates("participant_id")[["participant_id", "Age", "Gender"]].copy()
    check["behavior_age"] = pd.to_numeric(check["Age"], errors="coerce")
    check["behavior_gender"] = check["Gender"].map(norm_gender)
    check = check.merge(master[["participant_id", "Age", "Gender"]].rename(columns={"Age": "key_age", "Gender": "key_gender"}), on="participant_id", how="left")
    conflicts = check[(check["behavior_age"].ne(check["key_age"])) | (check["behavior_gender"].ne(check["key_gender"]))]
    fail_if(not conflicts.empty, conflicts, outdir, "behavior_key_age_gender_conflicts")

    out = beh.rename(columns={"risky choice": "risky-choice", "feedback": "feedback-condition", "Chronotype": "Chronotype_behavior", "Age": "Age_behavior", "Gender": "Gender_behavior"}).copy()
    out["participant_id"] = out["participant_id"].astype(int)
    out = out.sort_values(["participant_id", "Block", "Trial"]).reset_index(drop=True)
    out["global_trial_index"] = out.groupby("participant_id").cumcount() + 1
    out = out.merge(master[["participant_id", "Chronotype", "MEQ", "meq", "MCTQ", "mctq_minutes", "Gender", "Age", "has_eeg", "is_intermediate"]], on="participant_id", how="inner")
    out["behav_valence"] = out["feedback-condition"].astype(str).str.strip().str.lower().str.split("-").str[0]
    return out


ERP_RE = re.compile(r"^bin\d+_(?P<risk>.+?)_(?P<cond>gain-correct|gain-error|loss-correct|loss-error)_*_(?P<chan>Fz|FC1|FC2|Cz|Pz|POz|FCz)$")


def load_erp(path: Path, suffix: str, master: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    raw.columns = [norm_col(c) for c in raw.columns]
    fail_if("ERPset" not in raw.columns, pd.DataFrame({"column": list(raw.columns)}), outdir, f"{suffix.lower()}_missing_erpset")
    raw["participant_id"] = pd.to_numeric(raw["ERPset"], errors="coerce").astype("Int64")
    bad = raw[raw["participant_id"].isna()]
    fail_if(not bad.empty, bad, outdir, f"{suffix.lower()}_bad_erpset_ids")
    dup = raw[raw["participant_id"].duplicated(keep=False)]
    fail_if(not dup.empty, dup[["participant_id", "ERPset"]], outdir, f"{suffix.lower()}_duplicate_ids")

    erp_ids = pd.DataFrame({"participant_id": sorted(raw["participant_id"].astype(int).unique())})
    missing_key = erp_ids[~erp_ids["participant_id"].isin(master["participant_id"])]
    fail_if(not missing_key.empty, missing_key, outdir, f"{suffix.lower()}_ids_missing_from_key")
    analysis_ids = master.loc[master["has_eeg"] & master["Chronotype"].isin(["Morning", "Evening"]), ["participant_id"]]
    missing_erp = analysis_ids[~analysis_ids["participant_id"].isin(erp_ids["participant_id"])]
    fail_if(not missing_erp.empty, missing_erp, outdir, f"analysis_ids_missing_from_{suffix.lower()}")

    long_rows = []
    for col in raw.columns:
        match = ERP_RE.match(col)
        if not match:
            continue
        risk = match.group("risk").replace("-", "_")
        condition = match.group("cond").replace("-", "_")
        channel = match.group("chan")
        values = pd.to_numeric(raw[col], errors="coerce")
        frame = pd.DataFrame({
            "participant_id": raw["participant_id"].astype(int),
            "component": suffix,
            "risk": risk,
            "condition": condition,
            "channel": channel,
            "value": values,
        })
        long_rows.append(frame)
    long = pd.concat(long_rows, ignore_index=True)

    cond = long.groupby(["participant_id", "channel", "condition"], as_index=False)["value"].mean()
    wide = cond.pivot_table(index="participant_id", columns=["channel", "condition"], values="value", aggfunc="mean")
    wide.columns = [f"{ch}_{suffix}_{condition}_mean" for ch, condition in wide.columns]
    wide = wide.reset_index()

    for ch in CHANNELS:
        cols = [f"{ch}_{suffix}_{c}_mean" for c in CONDITIONS if f"{ch}_{suffix}_{c}_mean" in wide]
        if cols:
            wide[f"{ch}_{suffix}_mean"] = wide[cols].mean(axis=1)
        gain = [f"{ch}_{suffix}_{c}_mean" for c in ["gain_correct", "gain_error"] if f"{ch}_{suffix}_{c}_mean" in wide]
        loss = [f"{ch}_{suffix}_{c}_mean" for c in ["loss_correct", "loss_error"] if f"{ch}_{suffix}_{c}_mean" in wide]
        correct = [f"{ch}_{suffix}_{c}_mean" for c in ["gain_correct", "loss_correct"] if f"{ch}_{suffix}_{c}_mean" in wide]
        error = [f"{ch}_{suffix}_{c}_mean" for c in ["gain_error", "loss_error"] if f"{ch}_{suffix}_{c}_mean" in wide]
        if ch in CONTRAST_CHANNELS:
            wide[f"{ch}_{suffix}_loss_minus_gain"] = wide[loss].mean(axis=1) - wide[gain].mean(axis=1)
            wide[f"{ch}_{suffix}_error_minus_correct"] = wide[error].mean(axis=1) - wide[correct].mean(axis=1)
            wide[f"{ch}_{suffix}_loss_error_minus_gain_correct"] = wide[f"{ch}_{suffix}_loss_error_mean"] - wide[f"{ch}_{suffix}_gain_correct_mean"]
    return wide


def slope_or_nan(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or y[mask].nunique() < 2:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def mean_numeric(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    return pd.to_numeric(frame[col], errors="coerce").mean()


def build_participant_features(behavior: pd.DataFrame, master: pd.DataFrame, erp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    behavior = behavior.copy()
    for col in ["Option1", "Option2", "ResponseTime", "Trial", "Block", "global_trial_index", "risky-choice"]:
        behavior[col] = pd.to_numeric(behavior[col], errors="coerce")
    behavior["OptionDiff"] = behavior["Option1"] - behavior["Option2"]
    behavior["AbsOptionDiff"] = behavior["OptionDiff"].abs()
    behavior["ValueSum"] = behavior["Option1"] + behavior["Option2"]
    behavior = behavior.sort_values(["participant_id", "global_trial_index"])
    behavior["prev_feedback_condition"] = behavior.groupby("participant_id")["feedback-condition"].shift(1).astype(str).str.lower().str.replace("-", "_", regex=False)

    for pid, g in behavior.groupby("participant_id", sort=True):
        meta = master.loc[master["participant_id"].eq(pid)].iloc[0]
        row: dict[str, object] = {
            "participant_id": pid,
            "Chronotype": meta["Chronotype"],
            "Age": meta["Age"],
            "Gender": meta["Gender"],
            "MEQ": meta["MEQ"],
            "meq": meta["meq"],
            "MCTQ": meta["MCTQ"],
            "mctq_minutes": meta["mctq_minutes"],
            "has_eeg": meta["has_eeg"],
            "is_intermediate": meta["is_intermediate"],
            "n_trials": int(g.shape[0]),
            "n_free_trials": int(g.shape[0]),
        }
        for frame_name, frame in [("all", g), ("free", g)]:
            row[f"{frame_name}_risky_rate"] = mean_numeric(frame, "risky-choice")
            rt = pd.to_numeric(frame["ResponseTime"], errors="coerce")
            row[f"{frame_name}_rt_mean"] = rt.mean()
            row[f"{frame_name}_rt_std"] = rt.std()
            row[f"{frame_name}_rt_median"] = rt.median()
            for col in ["OptionDiff", "AbsOptionDiff", "ValueSum"]:
                vals = pd.to_numeric(frame[col], errors="coerce")
                row[f"{frame_name}_{col}_mean"] = vals.mean()
                row[f"{frame_name}_{col}_std"] = vals.std()
        cond = g["prev_feedback_condition"]
        for condition in CONDITIONS:
            frame = g.loc[cond.eq(condition)]
            row[f"{condition}_n"] = int(frame.shape[0])
            row[f"{condition}_free_n"] = int(frame.shape[0])
            row[f"{condition}_risky_rate"] = mean_numeric(frame, "risky-choice")
            row[f"{condition}_rt_mean"] = mean_numeric(frame, "ResponseTime")
        row["post_error_slowing"] = np.nanmean([row.get("gain_error_rt_mean"), row.get("loss_error_rt_mean")]) - np.nanmean([row.get("gain_correct_rt_mean"), row.get("loss_correct_rt_mean")])
        row["risk_after_error_minus_correct"] = np.nanmean([row.get("gain_error_risky_rate"), row.get("loss_error_risky_rate")]) - np.nanmean([row.get("gain_correct_risky_rate"), row.get("loss_correct_risky_rate")])
        row["risk_after_loss_error_minus_gain_correct"] = row.get("loss_error_risky_rate") - row.get("gain_correct_risky_rate")
        ordered = g.sort_values("global_trial_index")
        row["risky_slope"] = slope_or_nan(ordered["global_trial_index"], ordered["risky-choice"])
        row["rt_slope"] = slope_or_nan(ordered["global_trial_index"], ordered["ResponseTime"])
        n = len(ordered)
        early = ordered.iloc[: max(1, n // 3)]
        late = ordered.iloc[-max(1, n // 3):]
        row["risky_late_minus_early"] = mean_numeric(late, "risky-choice") - mean_numeric(early, "risky-choice")
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.merge(erp, on="participant_id", how="left")
    eeg_cols = [c for c in out.columns if c.endswith(("_FRN_mean", "_P300_mean"))]
    frn_cols = [c for c in eeg_cols if "_FRN_" in c]
    p300_cols = [c for c in eeg_cols if "_P300_" in c]
    if frn_cols:
        out["frn_global_mean"] = out[frn_cols].mean(axis=1)
        out["frontal_eeg_mean"] = out[[c for c in frn_cols if c.startswith(("Fz_", "FCz_", "FC1_", "FC2_"))]].mean(axis=1)
    if p300_cols:
        out["p300_global_mean"] = out[p300_cols].mean(axis=1)
        out["parietal_eeg_mean"] = out[[c for c in p300_cols if c.startswith(("Pz_", "POz_"))]].mean(axis=1)
    return out[out["has_eeg"] & out["Chronotype"].isin(["Morning", "Evening"])].sort_values("participant_id").reset_index(drop=True)


def add_risky_prechoice(behavior: pd.DataFrame) -> pd.DataFrame:
    out = behavior[behavior["has_eeg"] & behavior["Chronotype"].isin(["Morning", "Evening"])].copy()
    for col in ["Option1", "Option2", "Block", "Trial", "global_trial_index", "risky-choice", "ResponseTime", "CorrectChoice", "CurrentScore"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["risky-choice"].isin([0, 1])].sort_values(["participant_id", "global_trial_index"])
    out["risky_label"] = out["risky-choice"]
    out["OptionDiff"] = out["Option1"] - out["Option2"]
    out["AbsOptionDiff"] = out["OptionDiff"].abs()
    out["ValueSum"] = out["Option1"] + out["Option2"]
    out["ValueMax"] = out[["Option1", "Option2"]].max(axis=1)
    out["ValueMin"] = out[["Option1", "Option2"]].min(axis=1)
    out["OptionRatio"] = (out["Option1"] / out["Option2"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["AbsOptionRatio"] = out["OptionRatio"].abs()
    out["IsMixedSigns"] = ((out["Option1"] < 0) != (out["Option2"] < 0)).astype(int)
    out["BothPositive"] = ((out["Option1"] > 0) & (out["Option2"] > 0)).astype(int)
    out["BothNegative"] = ((out["Option1"] < 0) & (out["Option2"] < 0)).astype(int)
    g = out.groupby("participant_id", sort=False)
    out["TrialInParticipant"] = g.cumcount() + 1
    out["TrialProgress"] = out["TrialInParticipant"] / g["TrialInParticipant"].transform("max")
    for col in ["risky_label", "OptionDiff", "AbsOptionDiff", "ValueSum", "ResponseTime", "CorrectChoice", "CurrentScore"]:
        out["PrevRisky" if col == "risky_label" else f"Prev{col}"] = g[col].shift(1)
    out["PrevRTLog"] = np.log(pd.to_numeric(out["PrevResponseTime"], errors="coerce") + 1e-6)
    prev_score = g["CurrentScore"].shift(1)
    prev_prev_score = g["CurrentScore"].shift(2)
    out["PrevScoreDelta"] = prev_score - prev_prev_score
    shifted_risky = g["risky_label"].shift(1)
    for window in [3, 5, 10]:
        out[f"RollingRiskyRate{window}"] = shifted_risky.groupby(out["participant_id"]).transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    shifted_rt = g["ResponseTime"].shift(1)
    for window in [3, 5, 10]:
        out[f"RollingRTMean{window}"] = shifted_rt.groupby(out["participant_id"]).transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        out[f"RollingRTStd{window}"] = shifted_rt.groupby(out["participant_id"]).transform(lambda s: s.rolling(window=window, min_periods=2).std())
    prev_condition = g["feedback-condition"].shift(1).astype(str).str.lower().str.replace("-", "_", regex=False)
    out["PrevFeedbackLoss"] = prev_condition.str.startswith("loss").astype(float)
    out["PrevFeedbackError"] = prev_condition.str.endswith("error").astype(float)
    out["PrevGainCorrect"] = prev_condition.eq("gain_correct").astype(float)
    out["PrevGainError"] = prev_condition.eq("gain_error").astype(float)
    out["PrevLossCorrect"] = prev_condition.eq("loss_correct").astype(float)
    out["PrevLossError"] = prev_condition.eq("loss_error").astype(float)
    return out


def write_feature_packs(participant: pd.DataFrame, risky: pd.DataFrame, clean_dir: Path) -> None:
    compact = ["participant_id", "Chronotype", "post_error_slowing", "rt_slope", "risky_late_minus_early", "risk_after_loss_error_minus_gain_correct", "gain_correct_risky_rate", "loss_error_risky_rate", "Fz_FRN_error_minus_correct", "FCz_FRN_error_minus_correct", "Fz_FRN_loss_error_minus_gain_correct", "POz_P300_loss_minus_gain", "Pz_P300_loss_minus_gain", "POz_P300_error_minus_correct"]
    participant[[c for c in compact if c in participant.columns]].to_csv(clean_dir / "chronotype_compact_12.csv", index=False)
    packs = {
        "chronotype_demo_only.csv": ["participant_id", "Chronotype", "Age", "Gender"],
        "chronotype_behavior_core.csv": ["participant_id", "Chronotype", "Age", "Gender", "free_risky_rate", "free_rt_mean", "free_rt_std", "gain_correct_risky_rate", "gain_error_risky_rate", "loss_correct_risky_rate", "loss_error_risky_rate", "post_error_slowing", "risk_after_error_minus_correct", "risk_after_loss_error_minus_gain_correct", "risky_slope", "risky_late_minus_early", "rt_slope"],
        "chronotype_frn_core.csv": ["participant_id", "Chronotype", "Age", "Gender", "Fz_FRN_mean", "FCz_FRN_mean", "Cz_FRN_mean", "Fz_FRN_loss_minus_gain", "FCz_FRN_loss_minus_gain", "Cz_FRN_loss_minus_gain", "Fz_FRN_error_minus_correct", "FCz_FRN_error_minus_correct", "Cz_FRN_error_minus_correct", "Fz_FRN_loss_error_minus_gain_correct", "FCz_FRN_loss_error_minus_gain_correct", "Cz_FRN_loss_error_minus_gain_correct", "frn_global_mean", "frontal_eeg_mean"],
        "chronotype_p300_core.csv": ["participant_id", "Chronotype", "Age", "Gender", "Pz_P300_mean", "POz_P300_mean", "Pz_P300_loss_minus_gain", "POz_P300_loss_minus_gain", "Pz_P300_error_minus_correct", "POz_P300_error_minus_correct", "Pz_P300_gain_correct_mean", "POz_P300_gain_correct_mean", "p300_global_mean", "parietal_eeg_mean"],
        "chronotype_compact_combined.csv": list(participant.columns),
        "chronotype_all_literature.csv": list(participant.columns),
    }
    for name, cols in packs.items():
        cols = [c for c in cols if c in participant.columns]
        participant[cols].to_csv(clean_dir / name, index=False)
    risky.to_csv(clean_dir / "risky_choice_prechoice.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild clean data from authoritative participant-ID keyed files.")
    parser.add_argument("--key", default="data/raw/meq mctq scores - Sheet1.csv")
    parser.add_argument("--behavior", default="data/raw/all behavioral-2.xlsx")
    parser.add_argument("--frn", default="data/raw/frn_all_25-_350.xlsx")
    parser.add_argument("--p300", default="data/raw/p300_all_350_450.xlsx")
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--report-dir", default="reports/clean/rebuild")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    processed_dir = Path(args.processed_dir)
    report_dir = Path(args.report_dir)
    for d in [clean_dir, processed_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    master = load_master(Path(args.key), report_dir)
    behavior = load_behavior(Path(args.behavior), master, report_dir)
    frn = load_erp(Path(args.frn), "FRN", master, report_dir)
    p300 = load_erp(Path(args.p300), "P300", master, report_dir)
    erp = frn.merge(p300, on="participant_id", how="inner")

    participant = build_participant_features(behavior, master, erp)
    risky = add_risky_prechoice(behavior)

    master.to_csv(clean_dir / "participant_master.csv", index=False)
    master.to_csv(processed_dir / "participant_metadata.csv", index=False)
    master[["participant_id", "meq"]].rename(columns={"participant_id": "UserID"}).to_csv(processed_dir / "participant_meq_scores.csv", index=False)
    behavior.to_csv(processed_dir / "ml_ready_features.csv", index=False)
    participant.to_csv(clean_dir / "chronotype_participant.csv", index=False)
    write_feature_packs(participant, risky, clean_dir)

    summary = {
        "key_rows": int(master.shape[0]),
        "key_chronotype_counts": master["Chronotype"].value_counts(dropna=False).to_dict(),
        "has_eeg_counts": master["has_eeg"].value_counts(dropna=False).to_dict(),
        "analysis_sample_n": int(participant.shape[0]),
        "analysis_sample_counts": participant["Chronotype"].value_counts().to_dict(),
        "behavior_rows": int(behavior.shape[0]),
        "behavior_participants": int(behavior["participant_id"].nunique()),
        "erp_participants": int(erp["participant_id"].nunique()),
        "integrity_conflicts": [],
    }
    (report_dir / "rebuild_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
