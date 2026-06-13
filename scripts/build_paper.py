#!/usr/bin/env python3
"""Render the manuscript markdown into a single self-contained paper.html.

The output embeds the figures as base64 so the file is fully portable: it opens
directly in Word (File > Open) and imports into Google Docs, where co-authors can
revise with tracked changes and comments. No external assets or network needed.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import markdown

FIGURES = [
    ("fig1_p300_by_chronotype.png", "Figure 1. Posterior P300 loss-minus-gain by chronotype (primary finding)."),
    ("fig2_sensitivity_forest.png", "Figure 2. Robustness across participant exclusions: neural effect stable, classifier fragile."),
    ("fig3_feature_importance.png", "Figure 3. Held-out permutation importance for the compact_12 classifier."),
    ("fig4_risky_choice_baselines.png", "Figure 4. Risky-choice balanced accuracy vs naive baselines."),
    ("fig5_meq_continuous_p300.png", "Figure 5. Posterior P300 vs the continuous MEQ score (intermediate band shaded)."),
    ("fig6_ml_pipeline.png", "Figure 6. Leakage-aware nested cross-validation machine-learning pipeline."),
    ("fig7_roc.png", "Figure 7. Chronotype classification ROC (out-of-fold, nested CV)."),
    ("fig8_confusion_matrix.png", "Figure 8. Out-of-fold confusion matrix for the primary classifier."),
]

STYLE = """
body { font-family: Georgia, 'Times New Roman', serif; max-width: 820px;
  margin: 40px auto; line-height: 1.55; color: #1a1a1a; padding: 0 24px; }
h1 { font-size: 1.7em; line-height: 1.25; }
h2 { font-size: 1.25em; margin-top: 1.8em; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
h3 { font-size: 1.08em; margin-top: 1.4em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.92em; }
th, td { border: 1px solid #bbb; padding: 5px 9px; text-align: left; }
th { background: #f2f2f2; }
blockquote { background: #fff8e1; border-left: 4px solid #e0b400; margin: 1.2em 0;
  padding: 10px 16px; font-family: -apple-system, Arial, sans-serif; font-size: 0.92em; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
figure { margin: 1.6em 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid #e0e0e0; }
figcaption { font-size: 0.9em; color: #444; margin-top: 6px; font-family: -apple-system, Arial, sans-serif; }
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained paper.html from the manuscript markdown.")
    parser.add_argument("--md", default="docs/manuscript_draft.md")
    parser.add_argument("--figdir", default="docs/figures")
    parser.add_argument("--out", default="docs/paper.html")
    args = parser.parse_args()

    text = Path(args.md).read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])

    figdir = Path(args.figdir)
    fig_html = ["<h2>Figures</h2>"]
    for fname, caption in FIGURES:
        fpath = figdir / fname
        if not fpath.exists():
            continue
        b64 = base64.b64encode(fpath.read_bytes()).decode("ascii")
        fig_html.append(
            f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f"<figcaption>{caption}</figcaption></figure>"
        )

    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Chronotype and Feedback Processing</title>"
        f"<style>{STYLE}</style></head><body>"
        f"{body}\n{''.join(fig_html)}"
        "</body></html>"
    )
    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({kb:.0f} KB, {len(FIGURES)} figures embedded)")


if __name__ == "__main__":
    main()
