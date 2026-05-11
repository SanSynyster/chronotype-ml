#!/usr/bin/env python3
"""Build and evaluate literature-guided clean feature packs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_leaderboard(outdir: Path) -> None:
    rows = []
    for summary_path in sorted(outdir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        dataset = summary_path.parent.name
        task = "chronotype" if dataset.startswith("chronotype_") else "risky-choice"
        pack = dataset.replace("chronotype_", "").replace("risky_choice_", "")
        for model, metrics in summary.get("by_model_mean", {}).items():
            rows.append({
                "task": task,
                "pack": pack,
                "model": model,
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "roc_auc": metrics.get("roc_auc"),
                "n_rows": summary.get("rows"),
                "n_features": summary.get("features", {}).get("n_total"),
                "cv": summary.get("cv"),
            })
    if not rows:
        return

    leaderboard = pd.DataFrame(rows).sort_values(["task", "balanced_accuracy"], ascending=[True, False])
    leaderboard.to_csv(outdir / "leaderboard.csv", index=False)

    def table_to_markdown(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in frame.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.4f}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    markdown = ["# Literature Feature-Pack Leaderboard", ""]
    for task, task_df in leaderboard.groupby("task", sort=False):
        markdown.append(f"## {task}")
        markdown.append("")
        markdown.append(table_to_markdown(task_df.reset_index(drop=True)))
        markdown.append("")
    (outdir / "leaderboard.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Wrote {outdir / 'leaderboard.csv'}")
    print(f"Wrote {outdir / 'leaderboard.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate literature-guided feature packs.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--outdir", default="reports/clean/literature_packs")
    args = parser.parse_args()

    if not args.skip_build:
        run([args.python, "scripts/build_clean_risky_choice.py", "--include-prev-eeg", "--write-packs"])
        run([args.python, "scripts/build_clean_chronotype.py", "--write-packs"])

    risky_packs = sorted(Path("data/clean").glob("risky_choice_*.csv"))
    chronotype_packs = sorted(Path("data/clean").glob("chronotype_*.csv"))

    for path in risky_packs:
        if path.name == "risky_choice_prechoice.csv":
            continue
        run([
            args.python,
            "scripts/train_clean_baseline.py",
            "--data",
            str(path),
            "--target",
            "risky-choice",
            "--group-col",
            "participant_id",
            "--outdir",
            args.outdir,
        ])

    for path in chronotype_packs:
        if path.name == "chronotype_participant.csv":
            continue
        run([
            args.python,
            "scripts/train_clean_baseline.py",
            "--data",
            str(path),
            "--target",
            "Chronotype",
            "--group-col",
            "",
            "--outdir",
            args.outdir,
        ])

    write_leaderboard(Path(args.outdir))


if __name__ == "__main__":
    main()
