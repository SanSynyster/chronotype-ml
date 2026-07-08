#!/usr/bin/env python3
"""Check manuscript statistics against reports/clean artifacts.

Run from repo root:
    env/bin/python scripts/dl/statcheck.py

Dependencies: Python standard library only. The script is read-only with respect to
`docs/manuscript_draft.md` and writes `reports/clean/statcheck/report.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEED = 0


@dataclass
class Check:
    label: str
    source: str
    report_value: float
    manuscript_patterns: list[str]
    digits: int = 2


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def row_by_feature(summary: dict, feature: str) -> dict:
    for row in summary["rows"]:
        if row["feature"] == feature:
            return row
    raise KeyError(feature)


def meq_row(rows: list[dict], feature: str) -> dict:
    for row in rows:
        if row["feature"] == feature:
            return row
    raise KeyError(feature)


def fmt(value: float, digits: int) -> str:
    value = float(value)
    if math.isclose(value, 0.0, abs_tol=10 ** (-(digits + 1))):
        value = 0.0
    return f"{value:.{digits}f}"


def pattern_found(text: str, patterns: list[str]) -> tuple[bool, str]:
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE):
            return True, pat
    return False, ""


def build_checks(root: Path) -> list[Check]:
    group = load_json(root / "reports/clean/group_stats/summary.json")
    meq = load_json(root / "reports/clean/meq_p300/summary.json")
    fusion = load_json(root / "reports/clean/fusion_gru_p300/metrics.json")
    continuous = load_json(root / "reports/clean/continuous_meq/metrics.json")
    robustness = load_json(root / "reports/clean/robustness/metrics.json")
    spec = load_json(root / "reports/clean/spec_curve/summary.json")
    rl = load_json(root / "reports/clean/rl_hier/summary.json")
    eeg_feedback = load_json(root / "reports/clean/eegnet_feedback/metrics.json")
    eeg_chrono = load_json(root / "reports/clean/eeg_chronotype/metrics.json")
    coupling = load_json(root / "reports/clean/p300_coupling/summary.json")
    bayes = load_json(root / "reports/clean/bayes_nulls/summary.json")
    tost = load_json(root / "reports/clean/tost/summary.json")
    ml = load_json(root / "reports/clean/ml_full/summary.json")

    pz = row_by_feature(group, "Pz_P300_loss_minus_gain")
    poz = row_by_feature(group, "POz_P300_loss_minus_gain")
    pz_meq = meq_row(meq, "Pz_P300_loss_minus_gain")
    poz_meq = meq_row(meq, "POz_P300_loss_minus_gain")
    p300 = coupling["electrodes"]["Pz_P300"]

    fc_bf = next(row for row in bayes["frn_group_difference"] if row["contrast"] == "FCz_FRN_error_minus_correct")
    cz_bf = next(row for row in bayes["frn_group_difference"] if row["contrast"] == "Cz_FRN_error_minus_correct")
    fz_tost = next(row for row in tost["contrasts"] if row["contrast"] == "Fz_FRN_loss_minus_gain")

    return [
        Check("Pz P300 Cohen d", "group_stats", pz["cohens_d_evening_minus_morning"], [r"Pz[^\n]{0,180}d\s*=\s*-1\.04"], 2),
        Check("Pz P300 Welch p", "group_stats", pz["welch_p"], [r"Pz[\s\S]{0,260}Welch p\s*=\s*0\.0028"], 4),
        Check("POz P300 Cohen d", "group_stats", poz["cohens_d_evening_minus_morning"], [r"POz[^\n]{0,180}d\s*=\s*-0\.92"], 2),
        Check("POz P300 Welch p", "group_stats", poz["welch_p"], [r"POz[\s\S]{0,260}Welch p\s*=\s*0\.0076"], 4),
        Check("Pz MEQ Pearson r", "meq_p300", pz_meq["pearson_r"], [r"Pz Pearson r\s*=\s*0\.29"], 2),
        Check("POz MEQ Pearson r", "meq_p300", poz_meq["pearson_r"], [r"POz\s+r\s*=\s*0\.24"], 2),
        Check("Spec cells", "spec_curve", spec["n_cells"], [r"72-cell specification curve"], 0),
        Check("Spec negative cells", "spec_curve", spec["n_cells_negative_sign"], [r"same sign[\s\S]{0,120}64/72"], 0),
        Check("Spec significant large cells", "spec_curve", spec["n_cells_d_lt_-0.8_and_p_lt_.05"], [r"19/72"], 0),
        Check("ML primary balanced accuracy", "ml_full", ml["permutation"]["observed_ba"], [r"balanced accuracy\s+0\.717"], 3),
        Check("ML permutation p", "ml_full", ml["permutation"]["p_value"], [r"p\s*=\s*0\.020"], 3),
        Check("FRN FCz BF01", "bayes_nulls", fc_bf["bf01_null"], [r"FCz\s+BF01\s*=\s*2\.46"], 2),
        Check("FRN Cz BF01", "bayes_nulls", cz_bf["bf01_null"], [r"Cz BF01\s*=\s*3\.17"], 2),
        Check("FRN Fz TOST p", "tost", fz_tost["tost_p"], [r"Fz TOST p\s*=\s*0\.43"], 2),
        Check("GRU AUC", "fusion_gru_p300", fusion["gru_behavioral"]["nested_loo_roc_auc"], [r"ROC AUC\s+0\.713", r"AUC\s+0\.713"], 3),
        Check("GRU perm p", "fusion_gru_p300", fusion["gru_behavioral"]["perm_p_value"], [r"p\s*=\s*0\.027"], 3),
        Check("ERP-only AUC", "fusion_gru_p300", fusion["erp_p300"]["nested_loo_roc_auc"], [r"AUC\s+0\.668"], 3),
        Check("Fused AUC", "fusion_gru_p300", fusion["fused"]["nested_loo_roc_auc"], [r"AUC\s+0\.797"], 3),
        Check("Fused perm p", "fusion_gru_p300", fusion["fused"]["perm_p_value"], [r"p\s*=\s*0\.004"], 3),
        Check("Fused bootstrap CI low", "robustness", robustness["full"]["bootstrap_auc_95ci"][0], [r"\[0\.639,\s*0\.924\]", r"\[0\.64,\s*0\.92\]"], 3),
        Check("Fused MEQ prediction r", "continuous_meq", continuous["fused_gru_erp"]["loo_pred_actual_r"], [r"r\s*=\s*0\.344"], 3),
        Check("Hierarchical max R-hat", "rl_hier", rl["max_rhat"], [r"max\s+R-hat\s+1\.02"], 2),
        Check("Hier alpha_gain contrast", "rl_hier", rl["parameters"]["alpha_gain"]["contrast_evening_minus_morning"], [r"Evening . Morning .?\s*0\.00", r"alpha_gain contrast collapsed[^\n]+zero"], 2),
        Check("EEGNet chronotype mean AUC", "eeg_chronotype", eeg_chrono["embeddings"]["mean"]["nested_loo_roc_auc"], [r"mean-pooled AUC\s+0\.426"], 3),
        Check("EEGNet chronotype contrast AUC", "eeg_chronotype", eeg_chrono["embeddings"]["contrast"]["nested_loo_roc_auc"], [r"contrast embedding\s+AUC\s+0\.389"], 3),
        Check("EEGNet feedback valence AUC established", "eeg_chronotype auxiliary_valence_auc", eeg_chrono["auxiliary_valence_auc"], [r"AUC\s+0\.641", r"AUC\s+0\.64"], 3),
        Check("EEGNet feedback valence AUC fresh short run", "eegnet_feedback", eeg_feedback["out_of_fold"]["roc_auc"], [r"AUC\s+0\.594", r"AUC\s+0\.59"], 3),
        Check("Coupling overall d", "p300_coupling", p300["two_stage"]["overall"]["cohens_d_evening_minus_morning"], [r"d\s*=\s*-0\.36"], 2),
        Check("Coupling overall p", "p300_coupling", p300["two_stage"]["overall"]["mannwhitney_p"], [r"p\s*=\s*0\.22"], 2),
        Check("Coupling valence d", "p300_coupling", p300["two_stage"]["valence"]["cohens_d_evening_minus_morning"], [r"d\s*=\s*-0\.08"], 2),
        Check("Coupling GLMM p", "p300_coupling", p300["glmm"]["interaction_p"], [r"interaction[^\n]{0,80}p\s*=\s*0\.12"], 2),
    ]


def extract_stats(text: str) -> list[str]:
    patterns = [
        r"\bd\s*=\s*-?\d+\.\d+",
        r"\bp\s*[<=>]\s*\d+\.\d+",
        r"\bAUC\s+\d+\.\d+",
        r"\br\s*=\s*-?\d+\.\d+",
        r"\bCI\s*\[[^\]]+\]",
        r"\bBF01\s*=\s*\d+\.\d+",
    ]
    out = []
    for pat in patterns:
        out.extend(re.findall(pat, text, flags=re.IGNORECASE))
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=Path("docs/manuscript_draft.md"))
    parser.add_argument("--out", type=Path, default=Path("reports/clean/statcheck/report.md"))
    args = parser.parse_args()

    text = args.manuscript.read_text(encoding="utf-8")
    checks = build_checks(Path("."))
    rows = []
    for check in checks:
        found, pat = pattern_found(text, check.manuscript_patterns)
        status = "MATCH" if found else "MISMATCH / NOT FOUND"
        rows.append((status, check, pat))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manuscript Statcheck",
        "",
        f"Seed: {SEED}",
        "",
        "This report compares statistics in `docs/manuscript_draft.md` against local `reports/clean/*` artifacts. The manuscript was read only.",
        "",
        "## Summary",
        "",
        f"- Matches: {sum(1 for status, _, _ in rows if status == 'MATCH')}",
        f"- Mismatches / not found: {sum(1 for status, _, _ in rows if status != 'MATCH')}",
        "",
        "## Checks",
        "",
        "| Status | Statistic | Report source | Report value | Matched manuscript pattern |",
        "|---|---|---|---:|---|",
    ]
    for status, check, pat in rows:
        lines.append(f"| {status} | {check.label} | {check.source} | {fmt(check.report_value, check.digits)} | `{pat or 'not found'}` |")

    lines += [
        "",
        "## Parsed In-Text Statistics",
        "",
    ]
    lines.extend(f"- `{item}`" for item in extract_stats(text))
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
