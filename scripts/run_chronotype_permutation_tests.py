#!/usr/bin/env python3
"""Run permutation tests for chronotype literature feature packs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_PACKS = [
    "chronotype_demo_only.csv",
    "chronotype_behavior_core.csv",
    "chronotype_frn_core.csv",
    "chronotype_p300_core.csv",
    "chronotype_compact_combined.csv",
]


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_leaderboard(outdir: Path) -> None:
    rows = []
    for summary_path in sorted(outdir.glob("chronotype_*/logreg/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "pack": summary_path.parent.parent.name.replace("chronotype_", ""),
            "model": summary["model"],
            "observed_balanced_accuracy": summary["observed_balanced_accuracy"],
            "null_mean": summary["null_mean"],
            "null_std": summary["null_std"],
            "null_p95": summary["null_p95"],
            "p_value": summary["p_value"],
            "n_features": summary["n_features"],
            "n_permutations": summary["n_permutations"],
        })
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("observed_balanced_accuracy", ascending=False)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "leaderboard.csv", index=False)

    cols = list(df.columns)
    lines = ["# Chronotype Permutation Tests", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    (outdir / "leaderboard.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {outdir / 'leaderboard.csv'}")
    print(f"Wrote {outdir / 'leaderboard.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chronotype permutation tests.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--outdir", default="reports/clean/permutation_tests")
    parser.add_argument("--packs", default=",".join(DEFAULT_PACKS), help="Comma-separated CSV filenames under data/clean.")
    args = parser.parse_args()

    for pack_name in [p.strip() for p in args.packs.split(",") if p.strip()]:
        path = Path("data/clean") / pack_name
        if not path.exists():
            raise FileNotFoundError(path)
        run([
            args.python,
            "scripts/permutation_test_clean.py",
            "--data",
            str(path),
            "--target",
            "Chronotype",
            "--group-col",
            "",
            "--model",
            "logreg",
            "--permutations",
            str(args.permutations),
            "--outdir",
            args.outdir,
        ])

    write_leaderboard(Path(args.outdir))


if __name__ == "__main__":
    main()
