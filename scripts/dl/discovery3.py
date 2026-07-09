#!/usr/bin/env python3
"""Risk/agency modulation and individual-difference layer for model-based EEG.

Run with:
    env/bin/python scripts/dl/discovery3.py

Inputs are the corrected participant key, averaged ERP workbooks, behavioural trials,
hierarchical RL posterior means, and the model-based single-trial EEG/RPE table from
scripts/dl/model_based_eeg.py. Writes reports/clean/discovery3/.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


SEED = 0
N_BOOT = 5000
N_PERM = 5000
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/clean/discovery3"
MASTER = ROOT / "data/clean/participant_master.csv"
FRN_XLSX = ROOT / "data/raw/frn_all_25-_350.xlsx"
P300_XLSX = ROOT / "data/raw/p300_all_350_450.xlsx"
MODEL = ROOT / "reports/clean/model_based"


def _fdr_bh(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    prev = 1.0
    m = len(p)
    for rank, idx in enumerate(order[::-1], start=1):
        k = m - rank + 1
        prev = min(prev, p[idx] * m / k)
        q[idx] = prev
    return q.tolist()


def _ci(x: np.ndarray, rng: np.random.Generator, stat_fn=np.mean) -> list[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    vals = [float(stat_fn(rng.choice(x, len(x), replace=True))) for _ in range(N_BOOT)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _paired_perm_p(diff: np.ndarray, rng: np.random.Generator) -> float:
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    obs = abs(float(diff.mean()))
    null = []
    for _ in range(N_PERM):
        signs = rng.choice([-1.0, 1.0], size=len(diff), replace=True)
        null.append(abs(float((diff * signs).mean())))
    return float((np.sum(np.asarray(null) >= obs) + 1) / (N_PERM + 1))


def _corr_perm_p(x: np.ndarray, y: np.ndarray, method: str, rng: np.random.Generator) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    corr_fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    obs = abs(float(corr_fn(x, y).statistic))
    null = []
    for _ in range(N_PERM):
        null.append(abs(float(corr_fn(x, rng.permutation(y)).statistic)))
    return float((np.sum(np.asarray(null) >= obs) + 1) / (N_PERM + 1))


def _corr_ci(x: np.ndarray, y: np.ndarray, method: str, rng: np.random.Generator) -> list[float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    corr_fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    vals = []
    idx = np.arange(len(x))
    for _ in range(N_BOOT):
        b = rng.choice(idx, len(idx), replace=True)
        xb, yb = x[b], y[b]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        vals.append(float(corr_fn(xb, yb).statistic))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _clean_erp(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df["participant_id"] = df["ERPset"].astype(str).str.strip().astype(int)
    return df


def _erp_col(df: pd.DataFrame, bin_id: str, channel: str) -> str:
    pat = re.compile(rf"^{bin_id}.*_+{re.escape(channel)}$", re.IGNORECASE)
    hits = [c for c in df.columns if pat.search(c)]
    if not hits:
        pat = re.compile(rf"^{bin_id}.*{re.escape(channel)}$", re.IGNORECASE)
        hits = [c for c in df.columns if pat.search(c)]
    if len(hits) != 1:
        raise ValueError(f"expected one column for {bin_id}/{channel}, got {hits}")
    return hits[0]


def _mean_bins(df: pd.DataFrame, bins: list[str], channels: list[str]) -> pd.Series:
    cols = [_erp_col(df, b, ch) for b in bins for ch in channels]
    return df[cols].mean(axis=1)


def _paired_result(name: str, a: pd.Series, b: pd.Series, rng: np.random.Generator) -> dict:
    d = (a - b).to_numpy(float)
    d = d[np.isfinite(d)]
    dz = float(d.mean() / d.std(ddof=1))
    t = stats.ttest_rel(a, b, nan_policy="omit")
    w = stats.wilcoxon(d, zero_method="wilcox") if np.any(d != 0) else None
    return {
        "contrast": name,
        "n": int(len(d)),
        "mean_diff": float(d.mean()),
        "cohens_dz": dz,
        "dz_ci95": _ci(d / d.std(ddof=1), rng),
        "mean_diff_ci95": _ci(d, rng),
        "paired_t_p": float(t.pvalue),
        "wilcoxon_p": float(w.pvalue) if w is not None else 1.0,
        "permutation_p": _paired_perm_p(d, rng),
    }


def load_erp_measures(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    master = pd.read_csv(MASTER)
    keep = master[master["has_eeg"].astype(bool) & master["has_behaviour"].astype(bool)].copy()
    key = keep[["pid"]].rename(columns={"pid": "participant_id"}).copy()
    out = key.copy()

    results = {"risk_outcome": [], "sanity": []}
    specs = {
        "frn": (FRN_XLSX, ["FCz", "Cz"]),
        "p300": (P300_XLSX, ["Pz", "POz"]),
    }
    for feat, (path, chans) in specs.items():
        df = key.merge(_clean_erp(path), on="participant_id", how="inner")
        if len(df) != len(key):
            raise ValueError(f"{feat}: ERP join lost rows {len(df)} != {len(key)}")

        vals = {
            "low_gain_correct": _mean_bins(df, ["bin01"], chans),
            "low_gain_error": _mean_bins(df, ["bin02"], chans),
            "low_loss_correct": _mean_bins(df, ["bin03"], chans),
            "low_loss_error": _mean_bins(df, ["bin04"], chans),
            "high_gain_correct": _mean_bins(df, ["bin05"], chans),
            "high_loss_error": _mean_bins(df, ["bin06"], chans),
        }
        for k, v in vals.items():
            out[f"{feat}_{k}"] = v.to_numpy(float)
        out[f"{feat}_loss_minus_gain_all"] = _mean_bins(df, ["bin10"], chans).to_numpy(float) - _mean_bins(df, ["bin09"], chans).to_numpy(float)
        out[f"{feat}_error_minus_correct_low"] = ((vals["low_gain_error"] + vals["low_loss_error"]) / 2 - (vals["low_gain_correct"] + vals["low_loss_correct"]) / 2).to_numpy(float)
        out[f"{feat}_riskmod_gain_correct"] = (vals["high_gain_correct"] - vals["low_gain_correct"]).to_numpy(float)
        out[f"{feat}_riskmod_loss_error"] = (vals["high_loss_error"] - vals["low_loss_error"]).to_numpy(float)

        results["risk_outcome"].append(_paired_result(f"{feat}_gain_correct_high_minus_low", vals["high_gain_correct"], vals["low_gain_correct"], rng))
        results["risk_outcome"].append(_paired_result(f"{feat}_loss_error_high_minus_low", vals["high_loss_error"], vals["low_loss_error"], rng))
        gain_low = (vals["low_gain_correct"] + vals["low_gain_error"]) / 2
        loss_low = (vals["low_loss_correct"] + vals["low_loss_error"]) / 2
        corr_low = (vals["low_gain_correct"] + vals["low_loss_correct"]) / 2
        err_low = (vals["low_gain_error"] + vals["low_loss_error"]) / 2
        results["sanity"].append(_paired_result(f"{feat}_lowrisk_loss_minus_gain", loss_low, gain_low, rng))
        results["sanity"].append(_paired_result(f"{feat}_lowrisk_error_minus_correct", err_low, corr_low, rng))

    pvals = [r["permutation_p"] for r in results["risk_outcome"]]
    for r, q in zip(results["risk_outcome"], _fdr_bh(pvals)):
        r["fdr_q"] = q
    pvals = [r["permutation_p"] for r in results["sanity"]]
    for r, q in zip(results["sanity"], _fdr_bh(pvals)):
        r["fdr_q"] = q
    return out, results


def _slope(g: pd.DataFrame, feature: str) -> float:
    covars = ["signed_rpe", "abs_rpe", "outcome_valence", "outcome_magnitude_z", "trial_z"]
    d = g[[feature] + covars].dropna()
    if len(d) < 20 or d["signed_rpe"].nunique() < 3:
        return math.nan
    x = sm.add_constant(d[["signed_rpe", "abs_rpe", "outcome_valence", "outcome_magnitude_z", "trial_z"]], has_constant="add")
    return float(sm.OLS(d[feature], x).fit().params["signed_rpe"])


def risk_modulated_rpe(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(MODEL / "trial_regressor_table_hier.csv")
    data = data[data["is_free"]].copy()
    data["trial_z"] = data.groupby("participant_id")["trial_in_subject"].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    data["outcome_magnitude_z"] = data.groupby("participant_id")["outcome_magnitude"].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    features = {"frn": "frn_z", "theta": "theta_z", "p300": "p300_z"}
    rows = []
    for pid, g in data.groupby("participant_id"):
        row = {"participant_id": int(pid)}
        for name, col in features.items():
            safe = g[g["action"].astype(int) == 0]
            risky = g[g["action"].astype(int) == 1]
            row[f"{name}_rpe_slope_safe"] = _slope(safe, col)
            row[f"{name}_rpe_slope_risky"] = _slope(risky, col)
            row[f"{name}_rpe_slope_risky_minus_safe"] = row[f"{name}_rpe_slope_risky"] - row[f"{name}_rpe_slope_safe"]
        rows.append(row)
    slopes = pd.DataFrame(rows)
    results = {}
    for name in features:
        diff = slopes[f"{name}_rpe_slope_risky_minus_safe"].to_numpy(float)
        diff = diff[np.isfinite(diff)]
        w = stats.wilcoxon(diff, zero_method="wilcox") if np.any(diff != 0) else None
        results[name] = {
            "n": int(len(diff)),
            "mean_slope_diff_risky_minus_safe": float(diff.mean()),
            "median_slope_diff_risky_minus_safe": float(np.median(diff)),
            "bootstrap_ci95": _ci(diff, rng),
            "wilcoxon_p": float(w.pvalue) if w is not None else 1.0,
            "permutation_p": _paired_perm_p(diff, rng),
        }
    for r, q in zip(results.values(), _fdr_bh([v["permutation_p"] for v in results.values()])):
        r["fdr_q"] = q
    return slopes, results


def behavior_measures() -> pd.DataFrame:
    b = pd.read_csv(MODEL / "behavior_trials.csv")
    b = b[b["is_free"]].copy().sort_values(["participant_id", "trial_in_subject"])
    rows = []
    for pid, g in b.groupby("participant_id"):
        a = g["action"].astype(int).to_numpy()
        outcome = g["signed_outcome"].to_numpy(float)
        rt = g["ResponseTime"].to_numpy(float)
        stay = a[1:] == a[:-1]
        shift = a[1:] != a[:-1]
        prev_win = outcome[:-1] > 0
        prev_loss = outcome[:-1] < 0
        n = len(g)
        early = g.iloc[: n // 2]["action"].mean()
        late = g.iloc[n // 2 :]["action"].mean()
        prev_error = g["feedback"].astype(str).str.contains("Error", case=False).to_numpy()[:-1]
        prev_correct = g["feedback"].astype(str).str.contains("Correct", case=False).to_numpy()[:-1]
        rt_next = rt[1:]
        rows.append({
            "participant_id": int(pid),
            "risky_rate": float(np.mean(a)),
            "mean_rt": float(np.nanmean(rt)),
            "post_error_slowing": float(np.nanmean(rt_next[prev_error]) - np.nanmean(rt_next[prev_correct])),
            "win_stay": float(np.mean(stay[prev_win])) if np.any(prev_win) else math.nan,
            "lose_shift": float(np.mean(shift[prev_loss])) if np.any(prev_loss) else math.nan,
            "late_minus_early_risk": float(late - early),
        })
    return pd.DataFrame(rows)


def individual_differences(erp: pd.DataFrame, slopes: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    beh = behavior_measures()
    rl = pd.read_csv(MODEL / "rl_hier_params.csv")
    rl["lr_asymmetry"] = rl["alpha_loss"] - rl["alpha_gain"]
    subj = erp.merge(slopes, on="participant_id").merge(beh, on="participant_id").merge(rl, on="participant_id")
    subj.to_csv(OUT / "participant_measures.csv", index=False)

    neural = [
        "frn_loss_minus_gain_all", "frn_error_minus_correct_low", "frn_riskmod_gain_correct", "frn_riskmod_loss_error",
        "p300_loss_minus_gain_all", "p300_error_minus_correct_low", "p300_riskmod_gain_correct", "p300_riskmod_loss_error",
        "frn_rpe_slope_safe", "frn_rpe_slope_risky", "frn_rpe_slope_risky_minus_safe",
        "theta_rpe_slope_safe", "theta_rpe_slope_risky", "theta_rpe_slope_risky_minus_safe",
    ]
    behavior = ["risky_rate", "mean_rt", "post_error_slowing", "win_stay", "lose_shift", "late_minus_early_risk"]
    rl_vars = ["alpha_gain", "alpha_loss", "lr_asymmetry", "beta", "bias"]
    families = {"neural_behavior": behavior, "neural_rl": rl_vars}
    out: dict[str, list[dict]] = {}
    for fam, targets in families.items():
        rows = []
        for xname in neural:
            for yname in targets:
                x = subj[xname].to_numpy(float)
                y = subj[yname].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 8 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                    continue
                pear = stats.pearsonr(x[ok], y[ok])
                spear = stats.spearmanr(x[ok], y[ok])
                rows.append({
                    "x": xname,
                    "y": yname,
                    "n": int(ok.sum()),
                    "pearson_r": float(pear.statistic),
                    "pearson_ci95": _corr_ci(x, y, "pearson", rng),
                    "pearson_perm_p": _corr_perm_p(x, y, "pearson", rng),
                    "spearman_rho": float(spear.statistic),
                    "spearman_ci95": _corr_ci(x, y, "spearman", rng),
                    "spearman_perm_p": _corr_perm_p(x, y, "spearman", rng),
                })
        for key in ["pearson_perm_p", "spearman_perm_p"]:
            q = _fdr_bh([r[key] for r in rows])
            for r, qi in zip(rows, q):
                r[key.replace("perm_p", "fdr_q")] = qi
        out[fam] = rows
    return subj, out


def _verdict(p: float, q: float | None = None, weak_p: float = 0.05) -> str:
    if q is not None and q < 0.05:
        return "real"
    if p < weak_p:
        return "weak"
    return "null"


def write_findings(summary: dict) -> None:
    lines = [
        "# Discovery3: Risk/Agency and Individual Differences",
        "",
        "Chronotype ignored. All joins use integer participant id; averaged ERP workbooks use the whitespace-stripped ERPset column parsed as participant id. Primary inference is subject-level with bootstrap CIs, sign-flip/permutation p-values, and FDR within families.",
        "",
        "## P1 Risk x Outcome ERP Effects",
    ]
    for r in summary["p1"]["risk_outcome"]:
        lines.append(f"- {r['contrast']}: dz={r['cohens_dz']:.3f}, dz 95% CI [{r['dz_ci95'][0]:.3f}, {r['dz_ci95'][1]:.3f}], mean diff={r['mean_diff']:.3f}, mean-diff 95% CI [{r['mean_diff_ci95'][0]:.3f}, {r['mean_diff_ci95'][1]:.3f}], Wilcoxon p={r['wilcoxon_p']:.4g}, permutation p={r['permutation_p']:.4g}, FDR q={r['fdr_q']:.4g}; verdict={_verdict(r['permutation_p'], r['fdr_q'])}.")
    lines += ["", "Sanity checks:"]
    for r in summary["p1"]["sanity"]:
        lines.append(f"- {r['contrast']}: dz={r['cohens_dz']:.3f}, permutation p={r['permutation_p']:.4g}, FDR q={r['fdr_q']:.4g}; verdict={_verdict(r['permutation_p'], r['fdr_q'])}.")

    lines += ["", "## P2 Risk-Modulated RPE Encoding"]
    for name in ["frn", "theta", "p300"]:
        r = summary["p2"][name]
        lines.append(f"- {name.upper()} risky-minus-safe signed-RPE slope: mean diff={r['mean_slope_diff_risky_minus_safe']:.4f}, median diff={r['median_slope_diff_risky_minus_safe']:.4f}, 95% bootstrap CI [{r['bootstrap_ci95'][0]:.4f}, {r['bootstrap_ci95'][1]:.4f}], Wilcoxon p={r['wilcoxon_p']:.4g}, permutation p={r['permutation_p']:.4g}, FDR q={r['fdr_q']:.4g}; verdict={_verdict(r['permutation_p'], r['fdr_q'])}.")

    lines += ["", "## P3 Individual Differences", "N=52 gives roughly 80% power only for correlations around |r| >= 0.38, so CIs and corrected p-values matter more than nominal hits."]
    for fam in ["neural_behavior", "neural_rl"]:
        rows = sorted(summary["p3"][fam], key=lambda r: r["pearson_fdr_q"])
        lines.append(f"",)
        lines.append(f"Strongest {fam.replace('_', ' ')} correlations by Pearson FDR:")
        for r in rows[:8]:
            lines.append(f"- {r['x']} vs {r['y']}: Pearson r={r['pearson_r']:.3f}, 95% CI [{r['pearson_ci95'][0]:.3f}, {r['pearson_ci95'][1]:.3f}], perm p={r['pearson_perm_p']:.4g}, FDR q={r['pearson_fdr_q']:.4g}; Spearman rho={r['spearman_rho']:.3f}, FDR q={r['spearman_fdr_q']:.4g}; verdict={_verdict(r['pearson_perm_p'], r['pearson_fdr_q'])}.")

    any_p2 = any(summary["p2"][k]["fdr_q"] < 0.05 for k in ["frn", "theta"])
    any_p3 = any(r["pearson_fdr_q"] < 0.05 or r["spearman_fdr_q"] < 0.05 for fam in summary["p3"].values() for r in fam)
    bottom = "yes" if any_p2 else ("weak" if any_p3 else "no")
    lines += [
        "",
        "## Frank Assessment",
        f"Novel layer verdict beyond the established signed-RPE foundation: {bottom}.",
    ]
    if any_p2:
        lines.append("Risk/agency significantly modulates the RPE coupling and adds a mechanistic extension.")
    else:
        lines.append("Risk/agency modulation of RPE coupling is null after subject-level permutation/FDR; individual-difference results are treated as exploratory unless they survive correction.")
    (OUT / "findings.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    erp, p1 = load_erp_measures(rng)
    slopes, p2 = risk_modulated_rpe(rng)
    subj, p3 = individual_differences(erp, slopes, rng)
    p1_df = pd.DataFrame(p1["risk_outcome"] + p1["sanity"])
    p2_df = slopes
    p1_df.to_csv(OUT / "p1_erp_tests.csv", index=False)
    p2_df.to_csv(OUT / "p2_risk_modulated_rpe_slopes.csv", index=False)
    for fam, rows in p3.items():
        pd.DataFrame(rows).to_csv(OUT / f"p3_{fam}_correlations.csv", index=False)
    summary = {
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "n_permutations": N_PERM,
        "n_subjects": int(subj["participant_id"].nunique()),
        "p1": p1,
        "p2": p2,
        "p3": p3,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    write_findings(summary)
    print(f"wrote {OUT / 'summary.json'} and findings.md")


if __name__ == "__main__":
    main()
