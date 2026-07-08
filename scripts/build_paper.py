#!/usr/bin/env python3
"""Build the manuscript DOCX with Pandoc, APA-7 CSL, references, and figures.

Make-style one-liner: `env/bin/python scripts/build_paper.py --format docx`.

Dependencies: Python standard library plus Pandoc 3.10 on PATH. The default CSL is
the Zotero APA style URL; pass `--csl path/to/style.csl` for fully offline builds.
The manuscript source is read-only to this script; a temporary markdown file with
figure links and bibliography metadata is generated under `docs/.build/`.
"""

from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
from pathlib import Path


APA_CSL = "https://www.zotero.org/styles/apa"

MAIN_FIGURES = [
    ("fig_main_1_p300_spec.png", "Figure 1. Posterior P300 by chronotype and specification curve."),
    ("fig_main_2_continuous_meq.png", "Figure 2. Continuous MEQ associations."),
    ("fig_main_3_fusion.png", "Figure 3. Behaviour and ERP fusion."),
    ("fig_main_4_single_trial_coupling.png", "Figure 4. Single-trial P300 to next-choice coupling."),
    ("fig_main_5_roc_pipeline.png", "Figure 5. ROC and leakage-safe analysis pipeline."),
]

FALLBACK_FIGURES = [
    ("fig1_p300_by_chronotype.png", "Figure 1. Posterior P300 loss-minus-gain by chronotype."),
    ("fig5_meq_continuous_p300.png", "Figure 2. Posterior P300 versus continuous MEQ."),
    ("fig9_chronotype_from_dynamics.png", "Figure 3. Chronotype decoding from behaviour and fusion."),
    ("fig10_p300_risk_coupling.png", "Figure 4. Single-trial P300 to next-choice coupling."),
    ("fig7_roc.png", "Figure 5. Chronotype classifier ROC."),
]


def _figure_block(figdir: Path) -> str:
    figures = MAIN_FIGURES if all((figdir / name).exists() for name, _ in MAIN_FIGURES) else FALLBACK_FIGURES
    blocks = ["", "# Figures", ""]
    for name, caption in figures:
        path = figdir / name
        if path.exists():
            blocks += [f"![{caption}]({path.as_posix()})", ""]
    return "\n".join(blocks)


def _prepare_markdown(md: Path, figdir: Path, build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    text = md.read_text(encoding="utf-8")
    meta = "---\nnocite: '@*'\n---\n\n"
    tmp = build_dir / "paper_with_figures.md"
    tmp.write_text(meta + text + _figure_block(figdir) + "\n# References\n\n", encoding="utf-8")
    return tmp


def _run_pandoc(src: Path, out: Path, bib: Path, csl: str) -> None:
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc is not on PATH; install Pandoc 3.10 or pass through the correct environment.")
    cmd = [
        "pandoc",
        src.as_posix(),
        "--from",
        "markdown+pipe_tables+tex_math_dollars",
        "--to",
        "docx",
        "--citeproc",
        "--bibliography",
        bib.as_posix(),
        "--csl",
        csl,
        "--resource-path",
        ".:docs:docs/figures",
        "--output",
        out.as_posix(),
    ]
    subprocess.run(cmd, check=True)


def _build_html(md: Path, figdir: Path, out: Path) -> None:
    text = md.read_text(encoding="utf-8")
    figures = []
    for name, caption in FALLBACK_FIGURES:
        path = figdir / name
        if not path.exists():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        figures.append(f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    html = "<html><body><pre>" + text.replace("&", "&amp;").replace("<", "&lt;") + "</pre>" + "\n".join(figures) + "</body></html>"
    out.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build docs/paper.docx from docs/manuscript_draft.md.")
    parser.add_argument("--md", type=Path, default=Path("docs/manuscript_draft.md"))
    parser.add_argument("--figdir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--bib", type=Path, default=Path("references.bib"))
    parser.add_argument("--csl", default=APA_CSL)
    parser.add_argument("--out", type=Path, default=Path("docs/paper.docx"))
    parser.add_argument("--format", choices=["docx", "html"], default="docx")
    args = parser.parse_args()

    if args.format == "html":
        _build_html(args.md, args.figdir, args.out.with_suffix(".html"))
        print(f"Wrote {args.out.with_suffix('.html')}")
        return

    tmp = _prepare_markdown(args.md, args.figdir, Path("docs/.build"))
    _run_pandoc(tmp, args.out, args.bib, args.csl)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
