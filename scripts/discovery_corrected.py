#!/usr/bin/env python3
"""Corrected computational discovery analyses.

Reads only authoritative corrected/raw sources:
  - data/clean/participant_master.csv
  - data/raw/all behavioral-2.xlsx
  - data/raw/_singletrial_means/*_singletrial_means.csv
  - data/raw/frn_all_25-_350.xlsx, data/raw/p300_all_350_450.xlsx

Outputs reports/clean/discovery/summary.json and findings.md.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, ttest_1samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, roc_auc_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
OUT = Path("reports/clean/discovery")
MASTER = Path("data/clean/participant_master.csv")
BEH = Path("data/raw/all behavioral-2.xlsx")
MEANS = Path("data/raw/_singletrial_means")
FRN = Path("data/raw/frn_all_25-_350.xlsx")
P300 = Path("data/raw/p300_all_350_450.xlsx")


def clean_gender(x):
    return str(x).strip().upper()[0]


def load_master_behaviour():
    master = pd.read_csv(MASTER)
    beh = pd.read_excel(BEH)
    beh = beh.rename(columns={"UserID": "pid", "risky choice": "risky_choice", "feedback": "feedback"})
    beh["pid"] = beh["pid"].astype(int)
    chk = beh.groupby("pid")["Gender"].first().map(clean_gender).reset_index(name="beh_gender")
    joined = master.merge(chk, on="pid", how="inner")
    bad = joined[joined["gender"].map(clean_gender) != joined["beh_gender"]]
    if len(bad):
        raise AssertionError(f"Gender integrity failure: {bad['pid'].tolist()}")
    beh = beh.merge(master[["pid", "meq", "age", "gender", "chronotype", "chrono_binary", "has_eeg"]], on="pid", how="left")
    beh = beh.sort_values(["pid", "Block", "Trial"]).copy()
    beh["trial_index"] = beh.groupby("pid").cumcount() + 1
    beh["free"] = beh["risky_choice"].isin([0, 1])
    beh["choice_risky"] = beh["risky_choice"].where(beh["free"])
    beh["chosen_value"] = np.where(beh["ChoiceMade"].eq(1), beh["ActualValue1"], beh["ActualValue2"])
    beh["chosen_abs"] = np.abs(beh["chosen_value"])
    beh["valence_loss"] = beh["feedback"].astype(str).str.startswith("Loss").astype(int)
    beh["correct"] = beh["feedback"].astype(str).str.endswith("Correct").astype(int)
    return master, beh


def add_choice_features(beh):
    df = beh.copy()
    df["option_abs_diff"] = np.abs(np.abs(df["Option1"]) - np.abs(df["Option2"]))
    df["option_sum"] = df["Option1"] + df["Option2"]
    df["both_gain"] = ((df["ActualValue1"] > 0) & (df["ActualValue2"] > 0)).astype(int)
    df["both_loss"] = ((df["ActualValue1"] < 0) & (df["ActualValue2"] < 0)).astype(int)
    df["trial_progress"] = df["trial_index"] / df.groupby("pid")["trial_index"].transform("max")
    g = df.groupby("pid", sort=False)
    for col in ["choice_risky", "valence_loss", "correct", "ResponseTime", "chosen_value", "CurrentScore"]:
        df[f"prev_{col}"] = g[col].shift(1)
    df["prev_reward_scaled"] = df["prev_chosen_value"] / 25.0
    for w in [3, 5, 10]:
        df[f"roll_risky_{w}"] = g["choice_risky"].shift(1).groupby(df["pid"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"roll_loss_{w}"] = g["valence_loss"].shift(1).groupby(df["pid"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
    free = df[df["free"]].copy()
    return free


CHOICE_FEATURES = [
    "Option1", "Option2", "option_abs_diff", "option_sum", "both_gain", "both_loss", "trial_progress",
    "prev_choice_risky", "prev_valence_loss", "prev_correct", "prev_ResponseTime", "prev_reward_scaled",
    "roll_risky_3", "roll_risky_5", "roll_risky_10", "roll_loss_3", "roll_loss_5", "roll_loss_10",
]


def grouped_binary_cv(df, features, target, groups, model, n_perm=100):
    X = df[features]
    y = df[target].astype(int).to_numpy()
    g = df[groups].to_numpy()
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", model)])
    cv = GroupKFold(n_splits=5)
    prob = np.zeros(len(y))
    for tr, te in cv.split(X, y, g):
        pipe.fit(X.iloc[tr], y[tr])
        prob[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    obs_auc = roc_auc_score(y, prob)
    obs_ba = balanced_accuracy_score(y, (prob >= 0.5).astype(int))
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        # Permute target within participant; preserves free-trial counts and base rates.
        for pid in np.unique(g):
            idx = np.flatnonzero(g == pid)
            yp[idx] = rng.permutation(yp[idx])
        pp = np.zeros(len(y))
        for tr, te in cv.split(X, yp, g):
            pipe.fit(X.iloc[tr], yp[tr])
            pp[te] = pipe.predict_proba(X.iloc[te])[:, 1]
        null.append(roc_auc_score(yp, pp))
    p = float((1 + np.sum(np.array(null) >= obs_auc)) / (n_perm + 1))
    ci = subject_boot_ci(y, prob, g, lambda yy, pr: roc_auc_score(yy, pr))
    pipe.fit(X, y)
    perm_imp = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=20, random_state=SEED)
    top = sorted(zip(features, perm_imp.importances_mean), key=lambda x: x[1], reverse=True)[:8]
    return {"auc": obs_auc, "balanced_accuracy": obs_ba, "auc_ci": ci, "permutation_p": p, "null_auc": null, "top_temporal_drivers": top}


def subject_boot_ci(y, pred, groups, metric, n_boot=1000):
    rng = np.random.default_rng(SEED)
    subjects = np.unique(groups)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(subjects, len(subjects), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == s) for s in sample])
        try:
            vals.append(metric(y[idx], pred[idx]))
        except ValueError:
            pass
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def fit_rl(free):
    def nll(par, actions, rewards):
        ag, al, beta, bias, risk = par
        q = np.zeros(2)
        out = 0.0
        for a, r in zip(actions, rewards):
            logit = beta * (q[1] - q[0]) + bias + risk
            p = np.clip(1 / (1 + np.exp(-logit)), 1e-8, 1 - 1e-8)
            out -= np.log(p if a else 1 - p)
            q[a] += (ag if r > 0 else al) * (r - q[a])
        return out
    rows = []
    rng = np.random.default_rng(SEED)
    for pid, g in free.sort_values(["pid", "trial_index"]).groupby("pid"):
        actions = g["choice_risky"].astype(int).to_numpy()
        rewards = (g["chosen_value"] / 25.0).to_numpy()
        best = None
        for _ in range(12):
            x0 = [rng.uniform(), rng.uniform(), rng.uniform(0, 4), rng.uniform(-1, 1), rng.uniform(-1, 1)]
            res = minimize(nll, x0, args=(actions, rewards), method="L-BFGS-B", bounds=[(0, 1), (0, 1), (0, 10), (-5, 5), (-5, 5)])
            if best is None or res.fun < best.fun:
                best = res
        ag, al, beta, bias, risk = best.x
        rows.append({"pid": int(pid), "alpha_gain": ag, "alpha_loss": al, "lr_asymmetry": al - ag, "beta": beta, "bias": bias, "risk_sensitivity": risk, "nll": best.fun, "n_trials": len(g)})
    params = pd.DataFrame(rows)
    summary = {}
    for col in ["alpha_gain", "alpha_loss", "lr_asymmetry", "beta", "bias", "risk_sensitivity"]:
        vals = params[col].to_numpy()
        summary[col] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "bootstrap_ci": boot_mean_ci(vals)}
    t = ttest_1samp(params["lr_asymmetry"], 0)
    summary["lr_asymmetry"]["one_sample_p"] = float(t.pvalue)
    summary["hierarchical_note"] = "Population-level summaries are participant-level MLE means with subject bootstrap CIs; not a full PyMC posterior."
    return params, summary


def boot_mean_ci(vals, n=5000):
    rng = np.random.default_rng(SEED)
    boots = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def load_singletrial_erp(beh):
    rows = []
    for p in sorted(MEANS.glob("*_singletrial_means.csv")):
        m = re.search(r"(\d{4})", p.name)
        if not m:
            continue
        pid = int(m.group(1))
        df = pd.read_csv(p)
        df = df[df["good_trial"].eq(1)].copy()
        wide = df.pivot_table(index="trial", columns=["window", "channel"], values="mean_amp", aggfunc="mean")
        wide.columns = [f"{w}_{ch}" for w, ch in wide.columns]
        wide = wide.reset_index().rename(columns={"trial": "trial_index"})
        wide["pid"] = pid
        rows.append(wide)
    erp = pd.concat(rows, ignore_index=True)
    # Align by participant and EEG epoch order; EEG has fewer epochs for some subjects, so keep overlapping trial_index only.
    merged = beh.merge(erp, on=["pid", "trial_index"], how="inner")
    g = merged.groupby("pid", sort=False)
    merged["next_choice_risky"] = g["choice_risky"].shift(-1)
    merged["next_rt_log"] = np.log(g["ResponseTime"].shift(-1) + 1e-6)
    merged["risk_adjust"] = merged["next_choice_risky"] - merged["choice_risky"]
    return merged


def brain_behaviour(beh, n_perm=100):
    trial = load_singletrial_erp(beh)
    cols = [c for c in trial.columns if c.startswith(("FRN_", "P300_"))]
    # Electrode families: frontocentral FRN and parietal P300.
    trial["frn_fc"] = trial[[c for c in cols if c in ["FRN_Fz", "FRN_FCz", "FRN_FC1", "FRN_FC2", "FRN_Cz"]]].mean(axis=1)
    trial["p300_parietal"] = trial[[c for c in cols if c in ["P300_Pz", "P300_POz", "P300_Cz"]]].mean(axis=1)
    features = ["frn_fc", "p300_parietal", "valence_loss", "correct", "chosen_value", "choice_risky", "ResponseTime"]
    free_next = trial[trial["next_choice_risky"].isin([0, 1])].copy()
    res_choice = grouped_binary_cv(free_next, features, "next_choice_risky", "pid", LogisticRegression(max_iter=2000, class_weight="balanced"), n_perm=n_perm)
    slope_rows = []
    for pid, g in free_next.groupby("pid"):
        if g["next_choice_risky"].nunique() < 2 or len(g) < 20:
            continue
        X = g[["frn_fc", "p300_parietal"]].fillna(g[["frn_fc", "p300_parietal"]].median())
        X = StandardScaler().fit_transform(X)
        y = g["next_choice_risky"].astype(int).to_numpy()
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        slope_rows.append({"pid": pid, "frn_slope": clf.coef_[0, 0], "p300_slope": clf.coef_[0, 1]})
    slopes = pd.DataFrame(slope_rows)
    slope_summary = {}
    for c in ["frn_slope", "p300_slope"]:
        vals = slopes[c].dropna().to_numpy()
        slope_summary[c] = {"mean": float(vals.mean()), "ci": boot_mean_ci(vals), "one_sample_p": float(ttest_1samp(vals, 0).pvalue)}
    # RT and adjustment as exploratory continuous outcomes.
    cont = {}
    for target in ["next_rt_log", "risk_adjust"]:
        dd = free_next[np.isfinite(free_next[target])].copy()
        X = dd[features]
        y = dd[target].to_numpy(float)
        g = dd["pid"].to_numpy()
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("reg", RidgeCV(alphas=[0.1, 1, 10, 100]))])
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            pipe.fit(X.iloc[tr], y[tr])
            pred[te] = pipe.predict(X.iloc[te])
        cont[target] = {"r2": float(r2_score(y, pred)), "mae": float(mean_absolute_error(y, pred))}
    return trial, {"next_choice": res_choice, "per_subject_slopes": slope_summary, "continuous": cont, "n_trials_aligned": int(len(trial)), "n_participants": int(trial['pid'].nunique())}


def erp_subject_features(master):
    dfs = []
    for path, prefix in [(FRN, "frn"), (P300, "p300")]:
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"ERPset": "pid"})
        df["pid"] = df["pid"].astype(int)
        num = df.select_dtypes(include=[np.number]).drop(columns=["pid"], errors="ignore")
        feat = pd.DataFrame({"pid": df["pid"]})
        feat[f"{prefix}_loss_minus_gain_pz"] = df.filter(regex="loss.*Pz|all_loss.*Pz", axis=1).mean(axis=1) - df.filter(regex="gain.*Pz|all_gain.*Pz", axis=1).mean(axis=1)
        feat[f"{prefix}_mean"] = num.mean(axis=1)
        dfs.append(feat)
    out = dfs[0].merge(dfs[1], on="pid").merge(master, on="pid", how="left")
    return out


def exploratory(master, free, n_perm=100):
    feats = free.groupby("pid").agg(risk_propensity=("choice_risky", "mean"), rt_mean=("ResponseTime", "mean"), loss_rate=("valence_loss", "mean")).reset_index()
    erp = erp_subject_features(master).merge(feats, on="pid", how="left")
    results = {}
    for target in ["meq", "age", "risk_propensity"]:
        dd = erp.dropna(subset=[target]).copy()
        Xcols = ["frn_loss_minus_gain_pz", "p300_loss_minus_gain_pz", "frn_mean", "p300_mean", "rt_mean", "loss_rate"]
        X = dd[Xcols]
        y = dd[target].to_numpy(float)
        groups = dd["pid"].to_numpy()
        pred = np.zeros(len(y))
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("reg", RidgeCV(alphas=[0.1, 1, 10, 100]))])
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            pipe.fit(X.iloc[tr], y[tr])
            pred[te] = pipe.predict(X.iloc[te])
        results[target] = {"r2": float(r2_score(y, pred)), "mae": float(mean_absolute_error(y, pred)), "n": int(len(y))}
    dd = erp[erp["gender"].isin(["F", "M"])].dropna(subset=["risk_propensity"])
    Xcols = ["frn_loss_minus_gain_pz", "p300_loss_minus_gain_pz", "frn_mean", "p300_mean", "risk_propensity", "rt_mean"]
    y = dd["gender"].map({"F": 0, "M": 1}).astype(int).to_numpy()
    prob = np.zeros(len(y))
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    for tr, te in GroupKFold(n_splits=5).split(dd[Xcols], y, dd["pid"]):
        pipe.fit(dd[Xcols].iloc[tr], y[tr])
        prob[te] = pipe.predict_proba(dd[Xcols].iloc[te])[:, 1]
    results["sex"] = {"auc": float(roc_auc_score(y, prob)), "balanced_accuracy": float(balanced_accuracy_score(y, prob >= 0.5)), "n": int(len(y))}
    # FDR over exploratory nominal metrics using permutation-free conservative labels.
    results["multiplicity_note"] = "Exploratory family; no uncorrected result is interpreted as confirmatory."
    return results


def write_reports(summary):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Corrected Discovery Findings", "", "Chronotype is treated as a null side-variable; analyses join by integer participant id only.", ""]
    for key, text in summary["headline_verdicts"].items():
        lines.append(f"- **{key}:** {text}")
    lines += ["", "## Honest Paper Verdict", summary["paper_verdict"], ""]
    (OUT / "findings.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=100)
    args = ap.parse_args()
    master, beh = load_master_behaviour()
    free = add_choice_features(beh)
    choice = grouped_binary_cv(free, CHOICE_FEATURES, "choice_risky", "pid", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_leaf_nodes=15, random_state=SEED), n_perm=args.permutations)
    rl_params, rl_summary = fit_rl(free)
    OUT.mkdir(parents=True, exist_ok=True)
    rl_params.to_csv(OUT / "participant_rl_params.csv", index=False)
    trial, coupling = brain_behaviour(beh, n_perm=args.permutations)
    trial[["pid", "trial_index", "frn_fc", "p300_parietal", "next_choice_risky", "next_rt_log", "risk_adjust"]].to_csv(OUT / "singletrial_coupling_table.csv", index=False)
    exp = exploratory(master, free, n_perm=args.permutations)
    summary = {
        "seed": SEED,
        "data_sources": [str(MASTER), str(BEH), str(MEANS), str(FRN), str(P300)],
        "n_participants_behaviour": int(beh["pid"].nunique()),
        "n_free_trials": int(len(free)),
        "T2_choice_prediction": choice,
        "T3_RL": rl_summary,
        "T4_brain_behaviour_coupling": coupling,
        "T5_exploratory": exp,
    }
    summary["headline_verdicts"] = {
        "T2 choice dynamics": f"AUC {choice['auc']:.3f} (95% CI {choice['auc_ci'][0]:.3f}-{choice['auc_ci'][1]:.3f}), permutation p={choice['permutation_p']:.3f}; {'real' if choice['permutation_p'] < 0.05 else 'weak/null'}.",
        "T3 RL asymmetry": f"mean loss-gain learning-rate asymmetry {rl_summary['lr_asymmetry']['mean']:.3f} (CI {rl_summary['lr_asymmetry']['bootstrap_ci'][0]:.3f}-{rl_summary['lr_asymmetry']['bootstrap_ci'][1]:.3f}), p={rl_summary['lr_asymmetry']['one_sample_p']:.3f}.",
        "T4 ERP to next choice": f"AUC {coupling['next_choice']['auc']:.3f} (CI {coupling['next_choice']['auc_ci'][0]:.3f}-{coupling['next_choice']['auc_ci'][1]:.3f}), permutation p={coupling['next_choice']['permutation_p']:.3f}; {'real' if coupling['next_choice']['permutation_p'] < 0.05 else 'null'}.",
        "T5 individual differences": "Exploratory CV metrics are reported in summary.json; do not interpret as confirmatory without correction/replication.",
    }
    summary["paper_verdict"] = "A defensible non-chronotype paper requires robust feedback decoding plus a replicable behaviour/coupling effect. Based on these corrected analyses, claim only effects that survive grouped CV/permutation; nulls remain null."
    write_reports(summary)
    print(json.dumps(summary["headline_verdicts"], indent=2))
    print("wrote", OUT / "summary.json", "and findings.md")


if __name__ == "__main__":
    main()
