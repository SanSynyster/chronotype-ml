#!/usr/bin/env python3
"""Run permutation tests for chronotype literature feature packs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def add_fdr(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    """Add Benjamini-Hochberg FDR-adjusted p-values across the feature packs.

    Each pack is one hypothesis test in a family, so the leaderboard p-values
    must be corrected for the number of packs evaluated before any pack's
    significance is interpreted.
    """
    df = df.copy()
    p = df[p_col].to_numpy(dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adjusted = p * n / ranks
    # Enforce monotonicity of BH-adjusted values (step-up), then clip to 1.
    adj_sorted = adjusted[order]
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj_sorted, 0, 1)
    df["p_value_fdr_bh"] = out
    df["significant_fdr_05"] = df["p_value_fdr_bh"] < 0.05
    return df


def write_leaderboard(outdir: Path, pack_stems: list[str]) -> None:
    # Restrict the FDR family to exactly the packs run in this invocation.
    # Globbing the whole directory would fold in stale results and sensitivity
    # variants of the same model, which are not independent hypotheses and would
    # inflate the multiple-comparison family.
    rows = []
    for stem in pack_stems:
        summary_path = outdir / stem / "logreg" / "summary.json"
        if not summary_path.exists():
            continue
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
    df = add_fdr(df, p_col="p_value")
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

    pack_stems = []
    for pack_name in [p.strip() for p in args.packs.split(",") if p.strip()]:
        path = Path("data/clean") / pack_name
        if not path.exists():
            raise FileNotFoundError(path)
        pack_stems.append(path.stem)
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

    write_leaderboard(Path(args.outdir), pack_stems)


if __name__ == "__main__":
    main()
