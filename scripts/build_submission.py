#!/usr/bin/env python3
"""Build a clean submission DOCX from docs/manuscript_draft.md.

The working draft carries co-author scaffolding (cover note, integration map,
[AUTHOR INPUT] markers, an author-input checklist, and inline repo script paths)
and uses the old 1-10 figure numbering. This produces a submission-clean
`docs/paper_submission.docx` that:

  * strips all scaffolding and repo artifacts,
  * renumbers figure callouts to the final 5 main figures + supplementary S1-S4,
  * places each main figure inline right after its first mention (not dumped at
    the end), with a caption,
  * renders through Pandoc with a title block and no duplicate bibliography.

Run:  env/bin/python scripts/build_submission.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

MD = Path("docs/manuscript_draft.md")
OUT = Path("docs/paper_submission.docx")
FIGDIR = Path("docs/figures")
BUILD = Path("docs/.build")

# final main figures (consolidated) -> file + caption
FIGS = {
    1: ("fig_main_1_p300_spec.png",
        "Figure 1. Feedback-locked posterior P300 (loss minus gain) by chronotype "
        "at Pz and POz, with the window specification curve confirming robustness."),
    2: ("fig_main_2_continuous_meq.png",
        "Figure 2. Posterior P300 loss-minus-gain amplitude versus the continuous "
        "MEQ score."),
    3: ("fig_main_3_fusion.png",
        "Figure 3. Chronotype decoding from behavioural choice dynamics (GRU), from "
        "validated ERP contrasts, and their super-additive fusion."),
    4: ("fig_main_4_single_trial_coupling.png",
        "Figure 4. Single-trial feedback P300 to next-trial risk coupling by "
        "chronotype (overall and valence-resolved)."),
    5: ("fig_main_5_roc_pipeline.png",
        "Figure 5. Leakage-safe, permutation-clean evaluation pipeline and "
        "out-of-fold classifier ROC."),
}

# fallback to the original per-analysis figures if the consolidated set is absent
FALLBACK = {
    1: "fig1_p300_by_chronotype.png", 2: "fig5_meq_continuous_p300.png",
    3: "fig9_chronotype_from_dynamics.png", 4: "fig10_p300_risk_coupling.png",
    5: "fig7_roc.png",
}


def _remap_figures(text: str) -> str:
    """Old 1-10 callouts -> final 5 main + supplementary S1-S4, via placeholders."""
    order = [  # most specific first; two-pass to avoid collisions
        ("Figures 3, 6-8", "@@RANGE@@"),
        ("Figure 10", "@@F4@@"),   # single-trial coupling -> main 4
        ("Figure 1", "@@F1@@"),    # posterior P300 -> main 1
        ("Figure 2", "@@S1@@"),    # sensitivity forest -> S1
        ("Figure 3", "@@S2@@"),    # feature importance -> S2
        ("Figure 4", "@@S3@@"),    # risky-choice baselines -> S3
        ("Figure 5", "@@F2@@"),    # continuous MEQ -> main 2
        ("Figure 6", "@@F5@@"),    # ML pipeline -> main 5
        ("Figure 7", "@@F5@@"),    # ROC -> main 5
        ("Figure 8", "@@S4@@"),    # confusion matrix -> S4
        ("Figure 9", "@@F3@@"),    # dynamics/fusion -> main 3
    ]
    for old, ph in order:
        text = text.replace(old, ph)
    finals = {
        "@@F1@@": "Figure 1", "@@F2@@": "Figure 2", "@@F3@@": "Figure 3",
        "@@F4@@": "Figure 4", "@@F5@@": "Figure 5",
        "@@S1@@": "Supplementary Figure S1", "@@S2@@": "Supplementary Figure S2",
        "@@S3@@": "Supplementary Figure S3", "@@S4@@": "Supplementary Figure S4",
        "@@RANGE@@": "Figure 5; see also Supplementary Figures S2 and S4",
    }
    for ph, fin in finals.items():
        text = text.replace(ph, fin)
    return text


def _strip_scaffolding(text: str) -> tuple[str, str, str]:
    # cut the author-input checklist (and anything after it)
    text = re.split(r"\n#+ Author-input checklist", text)[0]
    # drop [AUTHOR INPUT: ...] markers (non-nested)
    text = re.sub(r"\s*\[AUTHOR INPUT:[^\]]*\]", "", text, flags=re.S)
    # replace the internal Reproducibility section with a clean availability statement
    text = re.sub(
        r"#+ 6\. Reproducibility and data availability.*?(?=\n#+ References)",
        "## Data and code availability\n\nAnalysis code and a de-identified "
        "participant-level dataset will be made openly available in a public "
        "repository upon publication; raw EEG data are available from the "
        "corresponding author on reasonable request.\n\n",
        text, flags=re.S)
    # remove the References editorial note (italic paragraph)
    text = re.sub(r"\*APA-7 list, DOI-verified.*?\*", "", text, flags=re.S)
    # drop blockquotes (cover note + integration map) and the target-journal line
    lines = [l for l in text.splitlines()
             if not l.lstrip().startswith(">") and not l.startswith("Target journal:")]
    text = "\n".join(lines)
    # pull out title + authors, then remove those lines
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), "Manuscript")
    authors = next((l.replace("Authors:", "").strip().rstrip(".")
                    for l in lines if l.startswith("Authors:")), "")
    text = re.sub(r"^# .*\n", "", text, count=1)
    text = re.sub(r"\nAuthors:.*\n", "\n", text, count=1)
    # strip backtick code spans that are repo paths/scripts
    text = re.sub(r"`[^`]*(?:/|\.py|scripts|reports/|docs/|data/)[^`]*`", "", text)
    # tidy punctuation left by removals
    text = re.sub(r"\(\s*[;,]?\s*\)", "", text)          # empty ()
    text = re.sub(r"\(\s*;\s*", "(", text)               # "( ; x)"
    text = re.sub(r"\s*;\s*\)", ")", text)               # "(x ; )"
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r"\n---\n", "\n\n", text)              # drop horizontal rules
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), title, authors


def _insert_figures_inline(text: str) -> str:
    paras = text.split("\n\n")
    placed: set[int] = set()
    out: list[str] = []
    for p in paras:
        out.append(p)
        for n in range(1, 6):
            if n in placed:
                continue
            if re.search(rf"\bFigure {n}\b", p):
                name, cap = FIGS[n]
                path = FIGDIR / name
                if not path.exists():
                    path = FIGDIR / FALLBACK[n]
                if path.exists():
                    out.append(f"![{cap}]({path.as_posix()})")
                placed.add(n)
    return "\n\n".join(out)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    raw = MD.read_text(encoding="utf-8")
    # The working draft is already numbered with the final 5 main figures +
    # Supplementary S-items, so no callout remapping is applied (running the old
    # 1-10 -> 1-5 remap here would corrupt the already-correct numbering).
    body, title, authors = _strip_scaffolding(raw)
    body = _insert_figures_inline(body)

    meta = f"---\ntitle: |\n  {title}\nauthor: |\n  {authors}\n---\n\n"
    src = BUILD / "submission.md"
    src.write_text(meta + body, encoding="utf-8")

    subprocess.run([
        "pandoc", src.as_posix(),
        "--from", "markdown+pipe_tables+tex_math_dollars",
        "--to", "docx",
        "--resource-path", ".:docs:docs/figures",
        "--output", OUT.as_posix(),
    ], check=True)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
