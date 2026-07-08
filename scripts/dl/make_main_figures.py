#!/usr/bin/env python3
"""Consolidate manuscript figures into five main figure panels.

Run from repo root:
    env/bin/python scripts/dl/make_main_figures.py

Dependencies: matplotlib and numpy. Seed is fixed at 0 for deterministic layout;
the script only combines existing figures and does not recompute statistics.
Outputs vector PDF and 300-DPI PNG files named `docs/figures/fig_main_*`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEED = 0
FIGDIR = Path("docs/figures")
REPORTS = Path("reports/clean")


def image(path: Path) -> np.ndarray:
    return plt.imread(path)


def panel(ax: plt.Axes, path: Path, title: str) -> None:
    ax.imshow(image(path))
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.axis("off")


def save(fig: plt.Figure, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(FIGDIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def two_panel(stem: str, left: Path, left_title: str, right: Path, right_title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    panel(axes[0], left, left_title)
    panel(axes[1], right, right_title)
    save(fig, stem)


def main() -> None:
    np.random.seed(SEED)

    spec_curve = REPORTS / "spec_curve/curve.png"
    if not spec_curve.exists():
        raise SystemExit("Missing reports/clean/spec_curve/curve.png; run scripts/dl/p300_spec_curve.py first.")

    two_panel(
        "fig_main_1_p300_spec",
        FIGDIR / "fig1_p300_by_chronotype.png",
        "A. Posterior P300 group contrast",
        spec_curve,
        "B. P300 specification curve",
    )
    shutil.copy2(spec_curve, FIGDIR / "fig_s1_full_spec_curve.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    panel(axes[0], FIGDIR / "fig5_meq_continuous_p300.png", "A. Posterior P300 versus MEQ")
    panel(axes[1], FIGDIR / "fig9_chronotype_from_dynamics.png", "B. MEQ-tracking predictive scores")
    save(fig, "fig_main_2_continuous_meq")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    panel(axes[0], FIGDIR / "fig9_chronotype_from_dynamics.png", "A. Behavioural dynamics decoding")
    panel(axes[1], FIGDIR / "fig3_feature_importance.png", "B. Validated ERP features")
    save(fig, "fig_main_3_fusion")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    panel(axes[0], FIGDIR / "fig10_p300_risk_coupling.png", "A. P300 to next-choice coupling")
    panel(axes[1], FIGDIR / "fig2_sensitivity_forest.png", "B. Trait-level robustness context")
    save(fig, "fig_main_4_single_trial_coupling")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panel(axes[0], FIGDIR / "fig6_ml_pipeline.png", "A. Leakage-safe pipeline")
    panel(axes[1], FIGDIR / "fig7_roc.png", "B. ROC curve")
    panel(axes[2], FIGDIR / "fig8_confusion_matrix.png", "C. Confusion matrix")
    save(fig, "fig_main_5_roc_pipeline")

    print("Wrote consolidated main figures to docs/figures/fig_main_*.[pdf,png]")


if __name__ == "__main__":
    main()
