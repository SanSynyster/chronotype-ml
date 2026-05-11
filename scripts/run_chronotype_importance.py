#!/usr/bin/env python3
"""Run the focused feature-importance analysis for chronotype."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chronotype compact-combined feature importance.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    cmd = [
        args.python,
        "scripts/feature_importance_clean.py",
        "--data",
        "data/clean/chronotype_compact_combined.csv",
        "--target",
        "Chronotype",
        "--group-col",
        "",
        "--model",
        "logreg",
        "--repeats",
        str(args.repeats),
        "--outdir",
        "reports/clean/feature_importance",
    ]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
