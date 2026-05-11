#!/usr/bin/env python3
"""Link participant-level raw metadata sheets to behavioural UserID values.

The summary workbooks do not contain an explicit UserID. This script derives a
stable mapping by matching participant_summary behavioural aggregates to metrics
recomputed from raw_behavioral_trials.xlsx. The matching columns in
participant_summary are previous-trial feedback summaries over the full 384
behavioural rows per participant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "mean response time",
    "response time after gain correct",
    "response time after gain error",
    "response time after loss correct",
    "response time after loss error",
    "mean risk taking only free tials",
    "risk taking after gain correct only free trials",
    "risk taking after gain error only free trials",
    "risk taking after loss correct only free trials",
    "risk taking after loss error only free trials",
]


def normalize_chronotype(value: object) -> str | float:
    label = str(value).strip().lower()
    if label in {"e", "evening", "eveningness"}:
        return "Evening"
    if label in {"m", "morning", "morningness"}:
        return "Morning"
    return np.nan


def clean_erpset(value: object) -> str:
    return str(value).strip().lower()


def summarize_previous_feedback(behavior: pd.DataFrame) -> pd.DataFrame:
    df = behavior.sort_values(["UserID", "Block", "Trial"]).copy()
    df["prev_feedback_condition"] = df.groupby("UserID")["feedback-condition"].shift(1)
    rows: list[dict[str, object]] = []

    for user_id, group in df.groupby("UserID", sort=True):
        free = group[group["forced and free risk trials"].astype(str).str.lower().eq("free")]
        row: dict[str, object] = {
            "UserID": int(user_id),
            "behavior_chronotype": normalize_chronotype(group["Chronotype"].dropna().iloc[0]),
            "mean response time": pd.to_numeric(group["ResponseTime"], errors="coerce").mean(),
            "mean risk taking only free tials": pd.to_numeric(free["risky-choice"], errors="coerce").mean(),
        }
        for condition, label in [
            ("Gain-Correct", "gain correct"),
            ("Gain-Error", "gain error"),
            ("Loss-Correct", "loss correct"),
            ("Loss-Error", "loss error"),
        ]:
            row[f"response time after {label}"] = pd.to_numeric(
                group.loc[group["prev_feedback_condition"].eq(condition), "ResponseTime"],
                errors="coerce",
            ).mean()
            row[f"risk taking after {label} only free trials"] = pd.to_numeric(
                free.loc[free["prev_feedback_condition"].eq(condition), "risky-choice"],
                errors="coerce",
            ).mean()
        rows.append(row)

    return pd.DataFrame(rows)


def match_summary_to_behavior(summary: pd.DataFrame, derived: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    left = summary[SUMMARY_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    right = derived[SUMMARY_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    scale = np.nanstd(np.vstack([left, right]), axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    distances = np.sqrt(np.nanmean(((left[:, None, :] - right[None, :, :]) / scale) ** 2, axis=2))
    best = distances.argmin(axis=1)

    rows: list[dict[str, object]] = []
    for idx, match_idx in enumerate(best):
        diffs = np.abs(left[idx] - right[match_idx])
        rows.append({
            "summary_row": idx + 1,
            "UserID": int(derived.iloc[match_idx]["UserID"]),
            "match_distance": float(distances[idx, match_idx]),
            "match_max_abs_diff": float(np.nanmax(diffs)),
            "summary_chronotype": normalize_chronotype(summary.iloc[idx].get("chronotype")),
            "summary_erpset": summary.iloc[idx].get("    ERPset"),
        })

    mapping = pd.DataFrame(rows)
    qc = {
        "rows": int(mapping.shape[0]),
        "unique_user_ids": int(mapping["UserID"].nunique()),
        "mean_match_distance": float(mapping["match_distance"].mean()),
        "max_match_distance": float(mapping["match_distance"].max()),
        "max_abs_diff": float(mapping["match_max_abs_diff"].max()),
    }
    return mapping, qc


def build_metadata(mapping: pd.DataFrame, derived: pd.DataFrame, final_data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    linked_summary = mapping.merge(derived[["UserID", "behavior_chronotype"]], on="UserID", how="left")
    linked_summary["summary_erpset_key"] = linked_summary["summary_erpset"].map(clean_erpset)

    final = final_data.copy()
    final["summary_erpset_key"] = final["    ERPset"].map(clean_erpset)
    final = final.rename(columns={
        "chronotype": "final_chronotype_raw",
        "ERPset": "metadata_name",
    })
    final_cols = [
        "summary_erpset_key",
        "metadata_name",
        "final_chronotype_raw",
    ]
    meta = linked_summary.merge(final[final_cols], on="summary_erpset_key", how="left")
    meta["primary_chronotype"] = meta["final_chronotype_raw"].map(normalize_chronotype)
    meta["summary_chronotype"] = meta["summary_chronotype"].map(normalize_chronotype)
    meta["chronotype_source"] = "all final data.xlsx chronotype via ERPset link"
    meta["chronotype_conflict_behavior"] = meta["primary_chronotype"] != meta["behavior_chronotype"]
    meta["chronotype_conflict_summary"] = meta["primary_chronotype"] != meta["summary_chronotype"]

    columns = [
        "UserID",
        "primary_chronotype",
        "behavior_chronotype",
        "summary_chronotype",
        "final_chronotype_raw",
        "chronotype_source",
        "chronotype_conflict_behavior",
        "chronotype_conflict_summary",
        "MEQ_MCTQ_status",
        "summary_erpset",
        "metadata_name",
        "summary_row",
        "match_distance",
        "match_max_abs_diff",
    ]
    meta["MEQ_MCTQ_status"] = "not_exported_side_by_side_table_order_unvalidated"
    meta = meta[columns].sort_values("UserID").reset_index(drop=True)
    qc = {
        "rows": int(meta.shape[0]),
        "unique_user_ids": int(meta["UserID"].nunique()),
        "primary_chronotype_counts": meta["primary_chronotype"].value_counts(dropna=False).to_dict(),
        "behavior_label_conflicts": meta.loc[meta["chronotype_conflict_behavior"], ["UserID", "primary_chronotype", "behavior_chronotype"]].to_dict("records"),
        "summary_label_conflicts": meta.loc[meta["chronotype_conflict_summary"], ["UserID", "primary_chronotype", "summary_chronotype"]].to_dict("records"),
        "missing_final_links": int(meta["primary_chronotype"].isna().sum()),
        "meq_mctq_status": "not_exported_side_by_side_table_order_unvalidated",
    }
    return meta, qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Link raw participant metadata to UserID.")
    parser.add_argument("--behavior", default="data/raw/raw_behavioral_trials.xlsx")
    parser.add_argument("--summary", default="data/raw/participant_summary.xlsx")
    parser.add_argument("--final", default="data/raw/all final data.xlsx")
    parser.add_argument("--out", default="data/processed/participant_metadata.csv")
    parser.add_argument("--mapping-out", default="reports/clean/metadata/participant_summary_mapping.csv")
    parser.add_argument("--qc", default="reports/clean/metadata/metadata_link_qc.json")
    args = parser.parse_args()

    behavior = pd.read_excel(args.behavior)
    summary = pd.read_excel(args.summary)
    final_data = pd.read_excel(args.final)
    derived = summarize_previous_feedback(behavior)
    mapping, match_qc = match_summary_to_behavior(summary, derived)
    metadata, metadata_qc = build_metadata(mapping, derived, final_data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(out_path, index=False)

    mapping_path = Path(args.mapping_out)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_path, index=False)

    qc = {"summary_matching": match_qc, "metadata": metadata_qc}
    qc_path = Path(args.qc)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with shape {metadata.shape}")
    print(f"Wrote {mapping_path}")
    print(f"Wrote {qc_path}")


if __name__ == "__main__":
    main()
