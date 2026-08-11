#!/usr/bin/env python3
"""Assemble docs/supplementary.md from reports/clean artifacts.

Run from repo root:
    env/bin/python scripts/dl/make_supplementary.py

Dependencies: Python standard library only. Seed is fixed at 0 in all consumed
analysis summaries; this script does not recompute analysis statistics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SEED = 0
ROOT = Path(".")


def j(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def f(value: Any, digits: int = 3) -> str:
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    spec = j("reports/clean/spec_curve/summary.json")
    rl = j("reports/clean/rl_hier/summary.json")
    robustness = j("reports/clean/robustness/metrics.json")
    risky = j("reports/clean/risky_choice_baseline/summary.json")
    ml = j("reports/clean/ml_full/summary.json")
    bayes = j("reports/clean/bayes_nulls/summary.json")
    tost = j("reports/clean/tost/summary.json")
    fusion = j("reports/clean/fusion_gru_p300/metrics.json")
    multimodal = j("reports/clean/multimodal_chronotype/metrics.json")
    eeg = j("reports/clean/eeg_chronotype/metrics.json")
    continuous = j("reports/clean/continuous_meq/metrics.json")
    coupling = j("reports/clean/p300_coupling/summary.json")
    importance = read_csv("reports/clean/feature_importance/chronotype_compact_12/logreg/importance_summary.csv")

    lines = [
        "# Supplementary Materials",
        "",
        "All values are derived from `reports/clean/*` artifacts with fixed seed 0. Raw `reports/clean` files are intentionally untracked; this document is the committed derived supplement.",
        "",
        "## Table S1. Full P300 Specification Curve",
        "",
        f"Grid: {spec['n_cells']} cells; {spec['n_cells_negative_sign']}/{spec['n_cells']} cells had the expected negative Evening-minus-Morning sign; {spec['n_cells_d_lt_-0.8_and_p_lt_.05']}/{spec['n_cells']} had d < -0.8 and p < .05.",
        "",
    ]
    spec_rows = []
    for row in spec["records"]:
        spec_rows.append([
            row["channel"], row["centre_ms"], row["width_ms"], row["summary"],
            f(row["cohens_d_evening_minus_morning"]), f(row["ci_low"]), f(row["ci_high"]), f(row["welch_p"], 4),
        ])
    lines += table(["Channel", "Centre ms", "Width ms", "Summary", "d", "CI low", "CI high", "Welch p"], spec_rows)

    lines += ["", "## Table S2. Hierarchical RL Contrasts and Shrinkage", ""]
    rl_rows = []
    for name, row in rl["parameters"].items():
        rl_rows.append([name, f(row.get("evening_mean")), f(row.get("morning_mean")), f(row["contrast_evening_minus_morning"]), f(row["contrast_hdi94"][0]), f(row["contrast_hdi94"][1]), f(row["P(contrast>0)"])])
    lines += table(["Parameter", "Evening mean", "Morning mean", "E-M", "94% HDI low", "94% HDI high", "P(E-M > 0)"], rl_rows)
    lines += ["", f"Sampler diagnostics: max R-hat = {f(rl['max_rhat'], 2)}; divergences = {rl['n_divergences']}.", ""]
    lines += table(["Parameter", "MLE vs hierarchical subject-estimate r"], [[k, f(v)] for k, v in rl["hier_vs_mle_corr"].items()])

    lines += ["", "## Table S3. Robustness Battery", ""]
    robust_rows = [["full", robustness["full"]["n"], f(robustness["full"]["nested_loo_roc_auc"]), "NA", f(robustness["full"]["perm_p_value"])]]
    for name, row in robustness["exclusions"].items():
        robust_rows.append([name, row["n"], f(row["nested_loo_roc_auc"]), f(row["balanced_accuracy"]), "NA"])
    lines += table(["Scenario", "n", "AUC", "Balanced accuracy", "Permutation p"], robust_rows)
    infl = robustness["loo_influence"]
    lines += ["", f"Leave-one-subject-out influence: AUC range {f(infl['auc_min'])}-{f(infl['auc_max'])}; most influential participant {infl['most_influential_participant']}; AUC without this participant {f(infl['auc_without_most_influential'])}."]

    lines += ["", "## Table S4. Risky-Choice Baselines", ""]
    lines += table(["Model", "Accuracy", "Balanced accuracy", "Note"], [[name, f(row.get("accuracy")), f(row.get("balanced_accuracy")), row.get("note", "")] for name, row in risky["baselines"].items()])

    lines += ["", "## Table S5. Chronotype Classifier Confusion Matrix", ""]
    labels = ml["confusion_matrix_labels"]
    cm = ml["confusion_matrix"]
    lines += table(["Actual", f"Pred {labels[0]}", f"Pred {labels[1]}"], [[labels[0], cm[0][0], cm[0][1]], [labels[1], cm[1][0], cm[1][1]]])
    lines += ["", f"Primary nested-CV balanced accuracy = {f(ml['permutation']['observed_ba'], 4)}; permutation p = {f(ml['permutation']['p_value'], 4)}."]

    lines += ["", "## Table S6. Feature Importance", ""]
    imp_rows = []
    for row in importance[:12]:
        imp_rows.append([row.get("feature", row.get("Feature", "")), row.get("mean_importance", row.get("importance_mean", "")), row.get("std_importance", row.get("importance_sd", ""))])
    lines += table(["Feature", "Mean importance", "SD"], imp_rows)

    lines += ["", "## Table S7. Bayes Factors for Planned Nulls", ""]
    lines += table(["Contrast", "Evening mean", "Morning mean", "Welch p", "BF01"], [[row["contrast"], f(row["evening_mean"]), f(row["morning_mean"]), f(row["welch_p"], 4), f(row["bf01_null"])] for row in bayes["frn_group_difference"]])
    eeg_bf_rows = []
    for name, row in bayes["eegnet_chronotype"]["chronotype_embeddings"].items():
        eeg_bf_rows.append([name, f(row["observed_auc"]), f(row["null_auc_mean"]), f(row["permutation_p_value"]), f(row["bf01_density_ratio"])])
    lines += ["", "EEGNet chronotype embedding nulls:", ""]
    lines += table(["Embedding", "Observed AUC", "Null mean", "Permutation p", "BF01"], eeg_bf_rows)

    lines += ["", "## Table S8. Equivalence Tests", ""]
    lines += table(["Contrast", "d", "TOST p", "Equivalent at alpha .05"], [[row["contrast"], f(row["cohens_d_evening_minus_morning"]), f(row["tost_p"], 4), row["equivalent_at_alpha"]] for row in tost["contrasts"]])

    lines += ["", "## Table S9. Deep-Learning and Fusion Decoding", ""]
    model_rows = []
    for name, row in fusion.items():
        model_rows.append([name, row["n_features"], f(row["nested_loo_roc_auc"]), f(row["nested_loo_balanced_accuracy"]), f(row["perm_p_value"]), f(row["meq_pearson_r"])])
    for name, row in multimodal.items():
        model_rows.append([name, row["n_features"], f(row["nested_loo_roc_auc"]), f(row["nested_loo_balanced_accuracy"]), f(row["perm_p_value"]), f(row["meq_pearson_r"])])
    lines += table(["Model", "n features", "AUC", "Balanced accuracy", "Permutation p", "MEQ r"], model_rows)
    lines += ["", f"Continuous MEQ fused Ridge prediction: r = {f(continuous['fused_gru_erp']['loo_pred_actual_r'])}, permutation p = {f(continuous['fused_gru_erp']['perm_p_value'])}."]

    lines += ["", "## Figure S1. Full Specification Curve", "", "See `reports/clean/spec_curve/curve.png`; the committed main figure script incorporates this panel into `docs/figures/fig_main_1_p300_spec.*`."]
    lines += ["", "## Figure S2. Single-Trial Coupling", ""]
    primary = coupling["electrodes"]["Pz_P300"]
    lines += [f"Pz overall coupling: d = {f(primary['two_stage']['overall']['cohens_d_evening_minus_morning'])}, p = {f(primary['two_stage']['overall']['mannwhitney_p'])}; valence interaction d = {f(primary['two_stage']['valence']['cohens_d_evening_minus_morning'])}, p = {f(primary['two_stage']['valence']['mannwhitney_p'])}; GLMM interaction p = {f(primary['glmm']['interaction_p'])}."]
    lines += ["", f"EEGNet auxiliary valence AUC recorded with chronotype embeddings = {f(eeg['auxiliary_valence_auc'])}; see `reports/clean/statcheck/auc_reconciliation.md` for the positive-control reconciliation."]

    out = ROOT / "docs/supplementary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
