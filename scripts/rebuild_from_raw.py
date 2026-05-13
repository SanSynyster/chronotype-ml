#!/usr/bin/env python3
"""Orchestrate active raw-to-clean reconstruction and write provenance metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], execute: bool) -> dict:
    record = {"command": cmd, "executed": execute, "returncode": None}
    print("$ " + " ".join(cmd), flush=True)
    if execute:
        proc = subprocess.run(cmd, cwd=ROOT)
        record["returncode"] = proc.returncode
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
    return record


def file_meta(path: Path) -> dict:
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)
    meta = {"path": display_path, "exists": path.exists()}
    if not path.exists():
        return meta
    meta["bytes"] = path.stat().st_size
    if path.suffix.lower() == ".csv":
        try:
            with path.open("rb") as f:
                meta["rows"] = max(sum(1 for _ in f) - 1, 0)
            meta["columns"] = len(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            meta["inspect_error"] = str(exc)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path, sheet_name=0, nrows=0)
            meta["columns"] = len(df.columns)
        except Exception as exc:
            meta["inspect_error"] = str(exc)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild processed and clean feature tables from raw local data.")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Without this, only writes a dry-run provenance plan.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", default="docs/data_provenance.md")
    parser.add_argument("--json", default="reports/clean/data_provenance.json")
    args = parser.parse_args()

    raw_inputs = [
        ROOT / "data/raw/_singletrial_means",
        ROOT / "data/raw/raw_behavioral_trials.xlsx",
        ROOT / "data/raw/participant_summary.xlsx",
        ROOT / "data/raw/all final data.xlsx",
    ]
    outputs = [
        ROOT / "data/processed/participant_metadata.csv",
        ROOT / "data/processed/ml_ready_features.csv",
        ROOT / "data/clean/risky_choice_prechoice.csv",
        ROOT / "data/clean/chronotype_participant.csv",
        ROOT / "data/clean/chronotype_compact_12.csv",
        ROOT / "data/clean/chronotype_compact_performance.csv",
        ROOT / "data/clean/sensitivity/manifest.json",
    ]

    commands = [
        [args.python, "scripts/link_raw_metadata.py"],
        [args.python, "scripts/build_ml_ready_from_raw.py"],
        [args.python, "scripts/build_clean_risky_choice.py", "--include-prev-eeg", "--write-packs"],
        [args.python, "scripts/build_clean_chronotype.py", "--write-packs"],
        [args.python, "scripts/build_compact_chronotype.py"],
        [args.python, "scripts/build_compact_performance_chronotype.py"],
        [args.python, "scripts/build_chronotype_sensitivity.py"],
    ]

    command_records = [run(cmd, args.execute) for cmd in commands]
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "executed": args.execute,
        "inputs": [file_meta(p) for p in raw_inputs],
        "outputs": [file_meta(p) for p in outputs],
        "commands": command_records,
        "notes": [
            "Raw data are local and intentionally not committed.",
            "Participant_summary is linked to UserID by matching previous-feedback behavioural aggregates recomputed from raw behaviour.",
            "All final data is linked through the ERPset column shared with participant_summary.",
            "Primary chronotype labels come from participant_summary / all final data metadata, not the raw behavioural Chronotype column.",
            "MEQ/MCTQ fields are not exported because their side-by-side workbook table order is unvalidated.",
            "The performance-informed compact model is exploratory and kept separate from the a priori theory-driven compact model.",
            "Active scripts build leakage-aware modelling datasets from the raw-derived ML-ready feature table.",
        ],
    }

    json_path = ROOT / args.json
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    lines = [
        "# Data Provenance",
        "",
        f"Generated: `{provenance['created_at']}`",
        f"Executed commands: `{args.execute}`",
        "",
        "## Raw Inputs",
        "",
        "| path | exists | bytes | columns |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in provenance["inputs"]:
        lines.append(f"| `{item['path']}` | {item.get('exists')} | {item.get('bytes', '')} | {item.get('columns', '')} |")
    lines += ["", "## Commands", ""]
    for idx, item in enumerate(provenance["commands"], start=1):
        lines.append(f"{idx}. `{' '.join(item['command'])}`")
    lines += ["", "## Outputs", "", "| path | exists | rows | columns | bytes |", "| --- | ---: | ---: | ---: | ---: |"]
    for item in provenance["outputs"]:
        lines.append(f"| `{item['path']}` | {item.get('exists')} | {item.get('rows', '')} | {item.get('columns', '')} | {item.get('bytes', '')} |")
    lines += ["", "## Notes", ""] + [f"- {note}" for note in provenance["notes"]]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
