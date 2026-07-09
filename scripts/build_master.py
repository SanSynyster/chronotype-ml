#!/usr/bin/env python3
"""Build the corrected participant master table from the authoritative key.

SINGLE SOURCE OF TRUTH: data/raw/meq mctq scores - Sheet1.csv, joined by the integer
`participant id`. NEVER re-link statistically. Chronotype/MEQ/MCTQ/gender/age/EEG
availability all come from this file. Behaviour is used only to (a) confirm the join
via gender and (b) list which participants have trial data.

Writes data/clean/participant_master.csv and a short integrity report.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

KEY = "data/raw/meq mctq scores - Sheet1.csv"
BEH = "data/raw/all behavioral-2.xlsx"
OUT = Path("data/clean/participant_master.csv")
REP = Path("reports/clean/master")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True); REP.mkdir(parents=True, exist_ok=True)
    k = pd.read_csv(KEY)
    k.columns = [c.strip() for c in k.columns]
    k = k.rename(columns={"participant id": "pid", "choronotype": "chronotype",
                          "electro data availability": "has_eeg"})
    k["pid"] = k["pid"].astype(int)
    k["chronotype"] = k["chronotype"].str.strip().str.capitalize()      # Morning/Evening/Inter
    k["is_intermediate"] = k["chronotype"].str.startswith("Inter")
    k["chrono_binary"] = k["chronotype"].where(k["chronotype"].isin(["Morning", "Evening"]))
    k["gender"] = k["gender"].str.strip().str.lower().map({"w": "F", "m": "M", "f": "F"})
    k["has_eeg"] = k["has_eeg"].astype(str).str.strip().str.lower().eq("yes")

    # behaviour: which pids have trials + their recorded gender (integrity cross-check)
    b = pd.read_excel(BEH)
    bu = b.groupby("UserID").agg(beh_gender=("Gender", "first")).reset_index()
    bu["beh_gender"] = bu["beh_gender"].astype(str).str.upper().str[0]
    bu["has_behaviour"] = True
    m = k.merge(bu, left_on="pid", right_on="UserID", how="left")
    m["has_behaviour"] = m["has_behaviour"].fillna(False)

    # INTEGRITY: gender must agree wherever behaviour exists (identity-critical)
    both = m[m["has_behaviour"]]
    gconf = both[both["gender"] != both["beh_gender"]]
    assert len(gconf) == 0, f"GENDER CONFLICT (join suspect): {gconf['pid'].tolist()}"

    keep = ["pid", "ERPset", "chronotype", "chrono_binary", "is_intermediate",
            "meq", "MCTQ", "gender", "age", "has_eeg", "has_behaviour"]
    m[keep].to_csv(OUT, index=False)

    n_eeg = int(m["has_eeg"].sum())
    n_bin_eeg = int((m["chrono_binary"].notna() & m["has_eeg"]).sum())
    lines = [
        "# Participant master — integrity", "",
        f"- screened in key: {len(m)}",
        f"- chronotype: " + ", ".join(f"{k}={v}" for k, v in m['chronotype'].value_counts().to_dict().items()),
        f"- has EEG: {n_eeg}",
        f"- has behaviour: {int(m['has_behaviour'].sum())}",
        f"- decisive (Morning/Evening) AND EEG: {n_bin_eeg}",
        f"- gender join conflicts: {len(gconf)} (must be 0) ✓",
        "",
        "Sample for chronotype-agnostic work (decoding, choice prediction) = all with "
        f"EEG and/or behaviour ({n_eeg} EEG). Chronotype is a null side-variable.",
    ]
    (REP / "integrity.md").write_text("\n".join(lines))
    print("\n".join(lines)); print("\nwrote", OUT)


if __name__ == "__main__":
    main()
