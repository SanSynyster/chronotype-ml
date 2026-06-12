#!/usr/bin/env python3
"""Generate manuscript figures from the tracked clean data and reports.

Outputs publication-ready vector PDFs plus PNG previews to docs/figures/.

Figures:
  1. P300 loss-minus-gain by chronotype (primary finding).
  2. Sensitivity forest: P300 effect size and classifier permutation p across
     participant-exclusion scenarios.
  3. Held-out permutation importance for the compact_12 classifier.
  4. Risky-choice balanced accuracy by feature pack vs naive baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVENING_C = "#4C72B0"
MORNING_C = "#DD8452"
ACCENT = "#C44E52"
GREY = "#555555"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png")
    plt.close(fig)
    print(f"Wrote {outdir / name}.pdf / .png")


def jitter(n: int, width: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n) - 0.5) * width


def fig1_p300(participant_csv: Path, group_stats_csv: Path, outdir: Path) -> None:
    df = pd.read_csv(participant_csv)
    stats = pd.read_csv(group_stats_csv).set_index("feature")
    features = ["Pz_P300_loss_minus_gain", "POz_P300_loss_minus_gain"]
    titles = ["Pz P300 (loss - gain)", "POz P300 (loss - gain)"]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4), sharey=False)
    for ax, feat, title in zip(axes, features, titles):
        groups = {"Evening": (EVENING_C, 1), "Morning": (MORNING_C, 2)}
        for label, (color, x) in groups.items():
            vals = pd.to_numeric(df.loc[df["Chronotype"].eq(label), feat], errors="coerce").dropna().to_numpy()
            ax.scatter(np.full(len(vals), x) + jitter(len(vals), 0.18, 7), vals,
                       color=color, alpha=0.7, s=28, zorder=3, edgecolor="white", linewidth=0.5)
            mean = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(len(vals))
            ax.plot([x - 0.22, x + 0.22], [mean, mean], color=color, lw=2.5, zorder=4)
            ax.errorbar(x, mean, yerr=1.96 * se, color=color, lw=2, capsize=4, zorder=4)
        ax.axhline(0, color=GREY, lw=0.8, ls=":", zorder=1)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Evening", "Morning"])
        ax.set_xlim(0.5, 2.5)
        ax.set_title(title)
        d = stats.loc[feat, "cohens_d_evening_minus_morning"]
        p = stats.loc[feat, "welch_p"]
        fdr = stats.loc[feat, "welch_p_fdr"]
        ax.annotate(f"d = {d:.2f}\np = {p:.3f}\nFDR p = {fdr:.3f}",
                    xy=(0.03, 0.03), xycoords="axes fraction", fontsize=9,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, alpha=0.8))
    axes[0].set_ylabel("Single-trial-mean amplitude contrast (z)")
    fig.suptitle("Feedback-related posterior P300 differs by chronotype", fontweight="bold")
    save(fig, outdir, "fig1_p300_by_chronotype")


def fig2_sensitivity(p300_csv: Path, classifier_csv: Path, outdir: Path) -> None:
    p300 = pd.read_csv(p300_csv)
    clf = pd.read_csv(classifier_csv)
    order = ["full", "exclude_1013", "exclude_label_conflicts", "exclude_all_flagged"]
    labels = ["Full (n=39)", "- 1013 (n=38)", "- conflicts (n=37)", "- all flagged (n=36)"]
    pos = {s: i for i, s in enumerate(order)}

    fig, (ax_d, ax_p) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})

    feats = {"Pz_P300_loss_minus_gain": (ACCENT, -0.12, "Pz"), "POz_P300_loss_minus_gain": ("#8172B3", 0.12, "POz")}
    for feat, (color, off, short) in feats.items():
        sub = p300[p300["feature"].eq(feat)]
        ys, ds, los, his = [], [], [], []
        for _, r in sub.iterrows():
            ys.append(pos[r["scenario"]] + off)
            ds.append(r["cohens_d"])
            los.append(r["cohens_d"] - r["cohens_d_ci95_low"])
            his.append(r["cohens_d_ci95_high"] - r["cohens_d"])
        ax_d.errorbar(ds, ys, xerr=[los, his], fmt="o", color=color, capsize=3,
                      label=short, lw=1.8, ms=6)
    ax_d.axvline(0, color=GREY, lw=0.8, ls=":")
    ax_d.set_yticks(range(len(order)))
    ax_d.set_yticklabels(labels)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Cohen's d (Evening - Morning), 95% CI")
    ax_d.set_title("Primary: P300 group difference")
    ax_d.legend(title="Electrode", loc="upper right", frameon=True, framealpha=0.85)

    cp = clf.set_index("scenario").loc[order]
    yvals = range(len(order))
    bars = ax_p.barh(list(yvals), cp["perm_p_value"].to_numpy(), color="#999999")
    for y, p in zip(yvals, cp["perm_p_value"].to_numpy()):
        ax_p.text(p + 0.01, y, f"{p:.3f}", va="center", fontsize=9)
    ax_p.axvline(0.05, color=ACCENT, lw=1.2, ls="--", label="p = 0.05")
    ax_p.set_yticks(list(yvals))
    ax_p.set_yticklabels([])
    ax_p.invert_yaxis()
    ax_p.set_xlim(0, max(0.45, cp["perm_p_value"].max() * 1.2))
    ax_p.set_xlabel("Permutation p")
    ax_p.set_title("Secondary: classifier")
    ax_p.legend(loc="lower right", frameon=False)

    fig.suptitle("Neural effect is robust; classifier is fragile under exclusions", fontweight="bold")
    save(fig, outdir, "fig2_sensitivity_forest")


def fig3_importance(importance_csv: Path, outdir: Path) -> None:
    df = pd.read_csv(importance_csv).sort_values("mean_importance")
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    colors = [ACCENT if v > 0 else GREY for v in df["mean_importance"]]
    ax.barh(df["feature"], df["mean_importance"], xerr=df["std_importance"],
            color=colors, alpha=0.85, error_kw={"lw": 1, "alpha": 0.5})
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("Mean balanced-accuracy drop (held-out permutation importance)")
    ax.set_title("compact_12 logistic regression: feature importance", fontweight="bold")
    save(fig, outdir, "fig3_feature_importance")


def fig4_risky(leaderboard_csv: Path, baseline_json: Path, outdir: Path) -> None:
    df = pd.read_csv(leaderboard_csv)
    df = df[df["task"].eq("risky-choice")].copy()
    best = df.sort_values("balanced_accuracy", ascending=False).groupby("pack", as_index=False).first()
    best = best.sort_values("balanced_accuracy", ascending=True)
    base = json.loads(Path(baseline_json).read_text())["baselines"]

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.barh(best["pack"] + " (" + best["model"] + ")", best["balanced_accuracy"], color=EVENING_C, alpha=0.85)
    refs = [
        ("Majority (0.50)", base["majority_class"]["balanced_accuracy"], GREY, ":"),
        ("Persistence (0.55)", base["persistence_prev_choice"]["balanced_accuracy"], "#666666", "--"),
        ("Participant-mean oracle (0.60)", base["participant_mean_oracle"]["balanced_accuracy"], ACCENT, "-."),
    ]
    for name, val, color, ls in refs:
        ax.axvline(val, color=color, ls=ls, lw=1.4, label=name)
    ax.set_xlim(0.45, 0.66)
    ax.set_xlabel("Balanced accuracy (participant-grouped CV)")
    ax.set_title("Risky choice: leakage-safe models vs naive baselines", fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    save(fig, outdir, "fig4_risky_choice_baselines")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript figures.")
    parser.add_argument("--outdir", default="docs/figures")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig1_p300(Path("data/clean/chronotype_participant.csv"),
              Path("reports/clean/group_stats/chronotype_group_stats.csv"), outdir)
    fig2_sensitivity(Path("reports/clean/sensitivity_matrix/p300_sensitivity.csv"),
                     Path("reports/clean/sensitivity_matrix/classifier_sensitivity.csv"), outdir)
    fig3_importance(Path("reports/clean/feature_importance/chronotype_compact_12/logreg/importance_summary.csv"), outdir)
    fig4_risky(Path("reports/clean/literature_packs/leaderboard.csv"),
               Path("reports/clean/risky_choice_baseline/summary.json"), outdir)


if __name__ == "__main__":
    main()
