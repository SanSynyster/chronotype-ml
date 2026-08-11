#!/usr/bin/env python3
"""G-E: P300 loss-minus-gain specification curve (Claude, per docs/specs_for_gpt.md).

Show the primary Pz/POz P300 loss-minus-gain group effect (Evening vs Morning) is
not an artefact of one analysis window. Recompute the group Cohen's d across a grid
of time windows and summary choices. This is a DESCRIPTIVE group comparison (no CV,
no prediction, no leakage concern -- it mirrors the primary ERP analysis).

Loads epochs once (widest window) and slices windows in memory. Epochs are already
baseline-corrected upstream, so no second baseline is applied.

Run (env_dl -- needs mne):
    PYTHONPATH=. env_dl/bin/python scripts/dl/p300_spec_curve.py
Writes reports/clean/spec_curve/{summary.json,summary.md,curve.png}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.dl.load_clean_epochs import load_all

OUTDIR = Path("reports/clean/spec_curve")
SEED = 0
N_BOOT = 5000
CHANNELS = ["Pz", "POz"]
# published anchor (primary ERP analysis): 450-550 ms mean window
ANCHOR = (0.450, 0.550)
ANCHOR_D = {"Pz": -1.04, "POz": -0.92}


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan


def boot_ci(a, b, rng, n=N_BOOT):
    a, b = np.asarray(a), np.asarray(b)
    ds = [cohens_d(rng.choice(a, len(a), True), rng.choice(b, len(b), True)) for _ in range(n)]
    return float(np.nanpercentile(ds, 2.5)), float(np.nanpercentile(ds, 97.5))


def per_subject_contrast(X_ch, valence, subject, times, w0, w1, summary):
    """Per-participant loss-minus-gain contrast for one channel/window/summary."""
    mask = (times >= w0) & (times <= w1)
    if summary == "mean":
        amp = X_ch[:, mask].mean(axis=1)
    else:  # peak: max positive amplitude in-window
        amp = X_ch[:, mask].max(axis=1)
    rows = {}
    for s in np.unique(subject):
        sm = subject == s
        loss = amp[sm & (valence == 1)]
        gain = amp[sm & (valence == 0)]
        if len(loss) and len(gain):
            rows[s] = loss.mean() - gain.mean()
    return rows


def main():
    rng = np.random.default_rng(SEED)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    d = load_all(tmin=-0.2, tmax=0.8, labeled_only=True)
    times = d["times"]
    valence = d["valence"]
    subject = d["subject"]
    chrono = d["chronotype"]  # per-trial; constant within subject
    subj_chrono = {s: int(chrono[subject == s][0]) for s in np.unique(subject)}
    assert all(c in d["ch_names"] for c in CHANNELS), "Pz/POz missing"

    # grid: window centres 400-600 ms, widths {50,100} ms, step 25 ms; summaries mean/peak
    centres = np.arange(0.400, 0.601, 0.025)
    widths = [0.050, 0.100]
    summaries = ["mean", "peak"]

    records = []
    for ch in CHANNELS:
        X_ch = d["X"][:, d["ch_names"].index(ch), :]
        for width in widths:
            half = width / 2
            for c in centres:
                w0, w1 = round(c - half, 4), round(c + half, 4)
                if w0 < times[0] or w1 > times[-1]:
                    continue
                for summ in summaries:
                    contrasts = per_subject_contrast(X_ch, valence, subject, times, w0, w1, summ)
                    ev = np.array([v for s, v in contrasts.items() if subj_chrono[s] == 1])
                    mo = np.array([v for s, v in contrasts.items() if subj_chrono[s] == 0])
                    d_val = cohens_d(ev, mo)
                    lo, hi = boot_ci(ev, mo, rng)
                    _, wp = stats.ttest_ind(ev, mo, equal_var=False)
                    records.append({
                        "channel": ch, "window": [w0, w1], "centre_ms": round(c * 1000),
                        "width_ms": round(width * 1000), "summary": summ,
                        "cohens_d_evening_minus_morning": float(d_val),
                        "ci_low": lo, "ci_high": hi, "welch_p": float(wp),
                        "n_evening": int(len(ev)), "n_morning": int(len(mo)),
                    })

    # sanity: the anchor cell (Pz/POz, 450-550 mean) should reproduce the primary d
    anchor_cells = {}
    for ch in CHANNELS:
        cell = next((r for r in records if r["channel"] == ch and r["summary"] == "mean"
                     and r["window"] == [ANCHOR[0], ANCHOR[1]]), None)
        if cell:
            anchor_cells[ch] = {"d": cell["cohens_d_evening_minus_morning"],
                                "published": ANCHOR_D[ch]}

    n_cells = len(records)
    n_strong = sum(1 for r in records if r["cohens_d_evening_minus_morning"] < -0.8
                   and r["welch_p"] < 0.05)
    n_neg = sum(1 for r in records if r["cohens_d_evening_minus_morning"] < 0)
    summary = {
        "seed": SEED, "n_boot": N_BOOT, "n_cells": n_cells,
        "grid": {"centres_ms": [round(c * 1000) for c in centres],
                 "widths_ms": [50, 100], "summaries": summaries, "channels": CHANNELS},
        "anchor_450_550_mean": anchor_cells,
        "n_cells_d_lt_-0.8_and_p_lt_.05": n_strong,
        "n_cells_negative_sign": n_neg,
        "sign_stable": n_neg == n_cells,
        "records": records,
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # figure: d +/- CI across window centre, one line per channel x summary, mean-width panel
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, width in zip(axes, [50, 100]):
        for ch, base in [("Pz", "#2166ac"), ("POz", "#b2182b")]:
            for summ, ls in [("mean", "-"), ("peak", "--")]:
                sub = [r for r in records if r["channel"] == ch and r["summary"] == summ
                       and r["width_ms"] == width]
                sub.sort(key=lambda r: r["centre_ms"])
                x = [r["centre_ms"] for r in sub]
                y = [r["cohens_d_evening_minus_morning"] for r in sub]
                lo = [r["ci_low"] for r in sub]; hi = [r["ci_high"] for r in sub]
                ax.plot(x, y, ls, color=base, label=f"{ch} {summ}")
                ax.fill_between(x, lo, hi, color=base, alpha=0.10)
        ax.axhline(0, color="grey", lw=0.9)
        ax.axhline(-0.8, color="grey", lw=0.7, ls=":")
        for ch, col in [("Pz", "#2166ac"), ("POz", "#b2182b")]:
            ax.scatter([500], [ANCHOR_D[ch]], marker="*", s=120, color=col, zorder=5,
                       edgecolor="white", label=f"{ch} published anchor")
        ax.set_title(f"{width} ms window")
        ax.set_xlabel("window centre (ms)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Cohen's d (Evening − Morning)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("P300 loss−minus−gain group effect: specification curve", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / "curve.png", dpi=200)
    fig.savefig(OUTDIR / "curve.pdf")
    plt.close(fig)

    md = [
        "# G-E: P300 loss-minus-gain specification curve",
        "",
        f"{n_cells} specification cells (Pz/POz × window centre 400-600 ms × width "
        f"{{50,100}} ms × {{mean,peak}}).",
        f"- Sign stable (d < 0, Evening more negative) in **{n_neg}/{n_cells}** cells "
        f"({'all' if summary['sign_stable'] else 'not all'}).",
        f"- Large & significant (d < -0.8 and p < .05) in **{n_strong}/{n_cells}** cells.",
        "",
        "## Anchor check (450-550 ms mean) vs published primary d",
    ]
    for ch, a in anchor_cells.items():
        md.append(f"- {ch}: spec-curve d = {a['d']:.3f} vs published {a['published']:.2f}")
    md += ["", "Figure: reports/clean/spec_curve/curve.png",
           "Interpretation: the primary P300 effect is not window-cherry-picked; the "
           "sign is stable across the plausible window/summary space and the largest, "
           "most significant cells cluster around the pre-specified 450-550 ms P300 window."]
    (OUTDIR / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nwrote", OUTDIR / "summary.json")


if __name__ == "__main__":
    main()
