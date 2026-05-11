#!/usr/bin/env python3
import argparse, json
import numpy as np
import warnings
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, GridSearchCV, StratifiedKFold

# Tidy up sklearn warnings about tiny bins and auto-adjusted quantiles
warnings.filterwarnings(
    "ignore",
    message="Bins whose width are too small",
    category=UserWarning,
    module="sklearn.preprocessing._discretization"
)
warnings.filterwarnings(
    "ignore",
    message="n_quantiles (.*) is greater than the total number of samples",
    category=UserWarning,
    module="sklearn.preprocessing._data"
)

NUMERIC_CANDIDATES = [
    # base engineered
    "OptionDiff","AbsOptionDiff","ValueSum",
    "ResponseTime","RT_log","RT_zscore",
    "mean_amp","mean_amp_z",
    # additional engineered
    "OptionRatio","AbsOptionRatio","IsMixedSigns",
    "ValueMax","ValueMin",
    # simple lags (per participant)
    "PrevRisky","PrevRT","PrevOptionDiff","PrevValueSum",
    # interactions
    "OptionDiff_x_RTlog","OptionDiff_x_Trial"
]
CATEGORICAL_CANDIDATES = [
    # keep empty unless you re-add any low-card categoricals you trust
]

# Non-linear basis candidates (we'll add discretized bins + quantile map)

BASIS_COLS = ["OptionDiff", "RT_log"]

# Manual selection from advisor report (fixed feature set)
KEEP_FEATURES = [
    "OptionDiff",
    "RT_log",
    "ResponseTime",
    "Trial",
]

# --- Hyperparameter grid parsing helpers ---

def _coerce_token(s: str):
    s = s.strip()
    if s.lower() in ("none", "null"):  # allow None in CLI grids
        return None
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        # prefer int when possible
        if s.replace(".", "", 1).isdigit() and s.count('.') <= 1:
            return float(s) if '.' in s else int(s)
    except Exception:
        pass
    return s


def parse_grid(grid_str: str) -> dict:
    """Parse a compact grid string like
    "max_depth=2|3|None;learning_rate=0.03|0.05;max_iter=400|800"
    into a dict suitable for GridSearchCV (without the 'clf__' prefix).
    """
    if not grid_str:
        return {}
    grid = {}
    for chunk in grid_str.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '=' not in chunk:
            continue
        key, vals = chunk.split('=', 1)
        options = [v for v in (x.strip() for x in vals.split('|')) if v != ""]
        if options:
            grid[key.strip()] = [_coerce_token(v) for v in options]
    return grid

def _split_cols(arg: str) -> list[str]:
    if not arg:
        return []
    return [c.strip() for c in arg.split(",") if c.strip()]

def _make_preprocessor_and_X(df: pd.DataFrame, selected_cols: list[str]):
    basis_cols_all = [c for c in BASIS_COLS if c in selected_cols]
    basis_cols = [c for c in basis_cols_all if df[c].nunique(dropna=True) >= 3]
    num_cols_no_basis = [c for c in selected_cols if c not in basis_cols]
    cat_cols = []

    # Full basis (for linear models)
    basis_pipe_full = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("quantile", QuantileTransformer(output_distribution="normal", n_quantiles=256, random_state=42)),
        ("kbins", KBinsDiscretizer(n_bins=3, encode="onehot", strategy="uniform", subsample=None)),
    ])
    # Raw basis (for tree models)
    basis_pipe_trees = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre_full = ColumnTransformer([
        ("num", num_pipe, num_cols_no_basis),
        ("basis", basis_pipe_full, basis_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    pre_trees = ColumnTransformer([
        ("num", num_pipe, num_cols_no_basis),
        ("basis", basis_pipe_trees, basis_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    X = df.loc[:, num_cols_no_basis + basis_cols].copy()
    return pre_full, pre_trees, X, basis_cols, num_cols_no_basis, cat_cols

def _eval_models(X, y, groups, pre_full, pre_trees, selected_cols, outdir: Path, trees_raw_basis: bool = False):
    models = {
        "logreg": LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs"),
        "rf": RandomForestClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=42, class_weight="balanced_subsample",
        ),
        "hgb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400, random_state=42),
    }
    gkf = GroupKFold(n_splits=5)
    step_summary = {}
    for name, clf in models.items():
        pre_used = pre_trees if (trees_raw_basis and name in ("rf","hgb")) else pre_full
        pipe = Pipeline([("pre", pre_used), ("clf", clf)])
        fold_rows = []
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
            n_q = int(min(256, max(16, len(tr))))
            try:
                pipe.set_params(pre__basis__quantile__n_quantiles=n_q)
            except ValueError:
                # no quantile step in raw-basis preprocessor
                pass
            if name == "logreg":
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                gs = GridSearchCV(pipe, param_grid={"clf__C": [0.1, 0.3, 1.0, 3.0]},
                                  cv=inner, scoring="balanced_accuracy", n_jobs=-1)
                gs.fit(X.iloc[tr], y[tr])
                cur = gs.best_estimator_
            else:
                pipe.fit(X.iloc[tr], y[tr]); cur = pipe

            y_pred = cur.predict(X.iloc[te])
            y_prob = None
            if hasattr(cur.named_steps["clf"], "predict_proba"):
                y_prob = cur.predict_proba(X.iloc[te])[:, 1]
            elif hasattr(cur.named_steps["clf"], "decision_function"):
                y_prob = cur.decision_function(X.iloc[te])

            if y_prob is not None:
                thrs = np.linspace(0.3, 0.7, 41)
                bal_accs = [balanced_accuracy_score(y[te], (y_prob >= t).astype(int)) for t in thrs]
                best_thr = float(thrs[int(np.argmax(bal_accs))])
                y_pred = (y_prob >= best_thr).astype(int)
            else:
                best_thr = 0.5

            m = fold_metrics(y[te], y_pred, y_prob)
            m["fold"] = fold; m["best_thr"] = best_thr
            fold_rows.append(m)

        fold_df = pd.DataFrame(fold_rows)
        step_summary[name] = fold_df.mean(numeric_only=True).to_dict()

    avg_bal = float(np.mean([v.get("bal_acc", np.nan) for v in step_summary.values()]))
    row = {"features": ",".join(selected_cols), "n_features": len(selected_cols), "score_bal_acc_avg": avg_bal}
    for name, metrics in step_summary.items():
        row.update({f"{name}_{k}": metrics.get(k, np.nan) for k in ("acc","bal_acc","f1_macro","roc_auc")})
    return row


def pick_existing(df, cols):
    return [c for c in cols if c in df.columns]

# --- Feature engineering (leakage-safe with GroupKFold by participant) ---

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Base value features (idempotent: re-creating if missing)
    if "OptionDiff" not in df.columns and all(c in df.columns for c in ["Option1","Option2"]):
        df["OptionDiff"] = df["Option1"] - df["Option2"]
    if "AbsOptionDiff" not in df.columns and "OptionDiff" in df.columns:
        df["AbsOptionDiff"] = df["OptionDiff"].abs()
    if "ValueSum" not in df.columns and all(c in df.columns for c in ["Option1","Option2"]):
        df["ValueSum"] = df[["Option1","Option2"]].sum(axis=1)

    # Additional value context
    if all(c in df.columns for c in ["Option1","Option2"]):
        denom = (df["Option1"].abs() + df["Option2"].abs()).replace(0, np.nan)
        df["OptionRatio"] = df["Option1"] / denom
        df["AbsOptionRatio"] = df["OptionRatio"].abs()
        df["IsMixedSigns"] = ((df["Option1"] * df["Option2"]) < 0).astype(int)
        df["ValueMax"] = df[["Option1","Option2"]].max(axis=1)
        df["ValueMin"] = df[["Option1","Option2"]].min(axis=1)

    # Behavioral transforms
    if "ResponseTime" in df.columns:
        df["RT_log"] = np.log(df["ResponseTime"].astype(float) + 1e-6)
        if "participant_id" in df.columns:
            grp = df.groupby("participant_id")["ResponseTime"]
            df["RT_zscore"] = (df["ResponseTime"] - grp.transform("mean")) / (grp.transform("std") + 1e-9)

    # EEG normalization (if present)
    if "mean_amp" in df.columns and "participant_id" in df.columns:
        grp = df.groupby("participant_id")["mean_amp"]
        df["mean_amp_z"] = (df["mean_amp"] - grp.transform("mean")) / (grp.transform("std") + 1e-9)

    # Lags per participant (safe because GroupKFold keeps participants in single folds)
    if "participant_id" in df.columns:
        sort_keys = [c for c in ["participant_id","Trial"] if c in df.columns]
        if sort_keys:
            df = df.sort_values(sort_keys)
            def _lag(col):
                return df.groupby("participant_id")[col].shift(1)
            if "risky-choice" in df.columns:
                df["PrevRisky"] = _lag("risky-choice").fillna(0).astype(float)
            if "ResponseTime" in df.columns:
                df["PrevRT"] = _lag("ResponseTime").fillna(df["ResponseTime"].median())
            if "OptionDiff" in df.columns:
                df["PrevOptionDiff"] = _lag("OptionDiff").fillna(0)
            if "ValueSum" in df.columns:
                df["PrevValueSum"] = _lag("ValueSum").fillna(0)

    # Interactions
    if all(c in df.columns for c in ["OptionDiff","RT_log"]):
        df["OptionDiff_x_RTlog"] = df["OptionDiff"] * df["RT_log"]
    if all(c in df.columns for c in ["OptionDiff","Trial"]):
        df["OptionDiff_x_Trial"] = df["OptionDiff"] * df["Trial"].astype(float)
    if all(c in df.columns for c in ["OptionDiff","PrevRT"]):
        df["OptionDiff_x_PrevRT"] = df["OptionDiff"] * df["PrevRT"]

    # Clean infinities from ratios/logs
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def fold_metrics(y_true, y_pred, y_prob=None):
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/merged/eeg_behav_trial_aggregated_free.csv")
    ap.add_argument("--outdir", default="reports/eeg_behav_trial_models")
    ap.add_argument("--group", default="participant_id")
    ap.add_argument(
        "--features",
        default="advisor",
        choices=[
            "advisor",
            "value_only",
            "rt_only",
            "trial_only",
            "rt_trial",
            "history_min",
            "fw_best",
            "fw_plus_rtlog",
            "fw_plus_prevrt",
            "fw_temporal",
            "custom",
        ],
        help=(
            "Feature preset to use: "
            "advisor (OptionDiff, RT_log, ResponseTime, Trial); "
            "value_only (OptionDiff); "
            "rt_only (RT_log, ResponseTime); "
            "trial_only (Trial); "
            "rt_trial (RT features + Trial); "
            "history_min (RT features + Trial + PrevRisky); "
            "fw_best (Trial, OptionDiff, PrevRisky as found by forward sweep); "
            "fw_plus_rtlog (fw_best + RT_log); "
            "fw_plus_prevrt (fw_best + PrevRT); "
            "fw_temporal (Trial, PrevRT); "
            "custom (use --custom-cols)."
        ),
    )
    ap.add_argument(
        "--custom-cols",
        default="",
        help=("Comma-separated list of feature columns to use when --features custom is selected.")
    )
    ap.add_argument(
        "--sweep-cols",
        default="",
        help=(
            "Comma-separated list of candidate columns to sweep "
            "(e.g., 'OptionDiff,RT_log,ResponseTime,Trial,PrevRisky')."
        ),
    )
    ap.add_argument(
        "--sweep-mode",
        default="",
        choices=["", "single", "cumulative", "forward"],
        help=(
            "If set, runs a feature sweep: 'single' = each column alone; "
            "'cumulative' = add columns left-to-right; "
            "'forward' = greedy forward selection."
        ),
    )
    ap.add_argument(
        "--sweep-maxk",
        type=int,
        default=10,
        help="Maximum features to select in forward mode.",
    )
    ap.add_argument(
        "--hgb-grid", default="",
        help=(
            "Hyperparameter grid for HistGradientBoosting in the format "
            "'max_depth=2|3|None;learning_rate=0.03|0.05;max_iter=400|800;max_leaf_nodes=15|31'. "
            "Values must be pipe-separated and the whole string quoted to protect '|' from the shell."
        ),
    )
    ap.add_argument(
        "--rf-grid", default="",
        help=(
            "Hyperparameter grid for RandomForest in the format "
            "'n_estimators=400|800;max_depth=None|10;min_samples_leaf=1|2'. "
            "Values must be pipe-separated and the whole string quoted to protect '|' from the shell."
        ),
    )
    ap.add_argument(
        "--trees-raw-basis",
        action="store_true",
        help=("If set, tree models (rf, hgb) will use a 'raw-basis' preprocessor: "
              "median-impute only for basis columns (no quantile mapping or k-bins). "
              "Linear models still use the full quantile+KBins basis."),
    )
    args = ap.parse_args()

    # Parse optional grids and prefix for pipeline usage
    hgb_grid_raw = parse_grid(args.hgb_grid)
    rf_grid_raw = parse_grid(args.rf_grid)
    hgb_grid = {f"clf__{k}": v for k, v in hgb_grid_raw.items()} if hgb_grid_raw else {}
    rf_grid = {f"clf__{k}": v for k, v in rf_grid_raw.items()} if rf_grid_raw else {}

    df = pd.read_csv(args.data)
    assert "risky-choice" in df.columns, "Missing target 'risky-choice'"
    assert args.group in df.columns, f"Missing group column '{args.group}'"

    # Build engineered features
    df = build_features(df)

    # -------------------------------
    # Target and groups
    # -------------------------------
    y = df["risky-choice"].astype(int).values
    groups = df[args.group].values

    # -------------------------------
    # Feature sweep (if requested)
    # -------------------------------
    sweep_cols = _split_cols(args.sweep_cols)
    if args.sweep_mode and sweep_cols:
        outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
        results = []

        if args.sweep_mode == "single":
            for c in sweep_cols:
                cols = [c] if c in df.columns else []
                if not cols:
                    continue
                pre_full, pre_trees, X, basis_cols, num_cols_no_basis, cat_cols = _make_preprocessor_and_X(df, cols)
                row = _eval_models(X, y, groups, pre_full, pre_trees, cols, outdir, trees_raw_basis=args.trees_raw_basis); row["mode"] = "single"
                results.append(row)

        elif args.sweep_mode == "cumulative":
            cols = []
            for c in sweep_cols:
                if c not in df.columns:
                    continue
                cols = cols + [c]
                pre_full, pre_trees, X, basis_cols, num_cols_no_basis, cat_cols = _make_preprocessor_and_X(df, cols)
                row = _eval_models(X, y, groups, pre_full, pre_trees, cols, outdir, trees_raw_basis=args.trees_raw_basis); row["mode"] = "cumulative"
                results.append(row)

        elif args.sweep_mode == "forward":
            remaining = [c for c in sweep_cols if c in df.columns]
            chosen = []
            while remaining and len(chosen) < args.sweep_maxk:
                best = None; best_row = None
                for c in remaining:
                    trial = chosen + [c]
                    pre_full, pre_trees, X, basis_cols, num_cols_no_basis, cat_cols = _make_preprocessor_and_X(df, trial)
                    row = _eval_models(X, y, groups, pre_full, pre_trees, trial, outdir, trees_raw_basis=args.trees_raw_basis)
                    if (best is None) or (row["score_bal_acc_avg"] > best):
                        best = row["score_bal_acc_avg"]; best_row = (c, row)
                if best_row is None:
                    break
                c_best, row = best_row
                chosen.append(c_best); remaining.remove(c_best)
                row["mode"] = "forward"; results.append(row)

        res_df = pd.DataFrame(results)
        res_path = Path(args.outdir) / "feature_sweep_results.csv"
        res_df.to_csv(res_path, index=False)
        print(f"✅ Wrote sweep results -> {res_path}")
        if not res_df.empty:
            best_idx = int(res_df["score_bal_acc_avg"].argmax())
            best_row = res_df.iloc[best_idx]
            print("🔎 Best (by avg balanced accuracy):")
            print(f"  mode={best_row['mode']}  n_features={best_row['n_features']}  features={best_row['features']}")
            print(f"  avg_bal_acc={best_row['score_bal_acc_avg']:.3f}")
        return

    # -------------------------------
    # Feature set selection based on --features argument
    # -------------------------------
    if args.features == "advisor":
        selected_cols = [c for c in KEEP_FEATURES if c in df.columns]
    elif args.features == "value_only":
        selected_cols = [c for c in ["OptionDiff"] if c in df.columns]
    elif args.features == "rt_only":
        selected_cols = [c for c in ["RT_log", "ResponseTime"] if c in df.columns]
    elif args.features == "trial_only":
        selected_cols = [c for c in ["Trial"] if c in df.columns]
    elif args.features == "rt_trial":
        selected_cols = [c for c in ["RT_log", "ResponseTime", "Trial"] if c in df.columns]
    elif args.features == "history_min":
        selected_cols = [c for c in ["RT_log", "ResponseTime", "Trial", "PrevRisky"] if c in df.columns]
    elif args.features == "fw_best":
        selected_cols = [c for c in ["Trial", "OptionDiff", "PrevRisky"] if c in df.columns]
    elif args.features == "fw_plus_rtlog":
        selected_cols = [c for c in ["Trial", "OptionDiff", "PrevRisky", "RT_log"] if c in df.columns]
    elif args.features == "fw_plus_prevrt":
        selected_cols = [c for c in ["Trial", "OptionDiff", "PrevRisky", "PrevRT"] if c in df.columns]
    elif args.features == "fw_temporal":
        selected_cols = [c for c in ["Trial", "PrevRT"] if c in df.columns]
    elif args.features == "custom":
        custom_list = [c.strip() for c in (args.custom_cols or "").split(',') if c.strip()]
        selected_cols = [c for c in custom_list if c in df.columns]
    else:
        selected_cols = [c for c in KEEP_FEATURES if c in df.columns]
    print(f"Using features: {selected_cols}")

    pre_full, pre_trees, X, basis_cols, num_cols_no_basis, cat_cols = _make_preprocessor_and_X(df, selected_cols)

    y = df["risky-choice"].astype(int).values
    groups = df[args.group].values
    selected_cols = num_cols_no_basis + basis_cols  # cat_cols is empty here
    # X already built above

    models = {
        "logreg": LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs"),
        "rf": RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",
        ),
        "hgb": HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=400,
            random_state=42,
        ),
    }

    gkf = GroupKFold(n_splits=5)
    summary = {}

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    for name, clf in models.items():
        pre_used = pre_trees if (args.trees_raw_basis and name in ("rf","hgb")) else pre_full
        pipe = Pipeline([("pre", pre_used), ("clf", clf)])

        fold_rows = []
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
            n_q = int(min(256, max(16, len(tr))))
            try:
                pipe.set_params(pre__basis__quantile__n_quantiles=n_q)
            except ValueError:
                # no quantile step in raw-basis preprocessor
                pass
            if name == "logreg":
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                gs = GridSearchCV(pipe, param_grid={"clf__C": [0.1, 0.3, 1.0, 3.0]},
                                  cv=inner, scoring="balanced_accuracy", n_jobs=-1)
                gs.fit(X.iloc[tr], y[tr])
                cur = gs.best_estimator_
                best_params = gs.best_params_
            elif name == "hgb" and hgb_grid:
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                gs = GridSearchCV(pipe, param_grid=hgb_grid, cv=inner,
                                  scoring="balanced_accuracy", n_jobs=-1)
                gs.fit(X.iloc[tr], y[tr])
                cur = gs.best_estimator_
                best_params = gs.best_params_
            elif name == "rf" and rf_grid:
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                gs = GridSearchCV(pipe, param_grid=rf_grid, cv=inner,
                                  scoring="balanced_accuracy", n_jobs=-1)
                gs.fit(X.iloc[tr], y[tr])
                cur = gs.best_estimator_
                best_params = gs.best_params_
            else:
                pipe.fit(X.iloc[tr], y[tr])
                cur = pipe
                best_params = {}

            y_pred = cur.predict(X.iloc[te])
            y_prob = None
            if hasattr(cur.named_steps["clf"], "predict_proba"):
                y_prob = cur.predict_proba(X.iloc[te])[:, 1]
            elif hasattr(cur.named_steps["clf"], "decision_function"):
                y_prob = cur.decision_function(X.iloc[te])

            if y_prob is not None:
                thrs = np.linspace(0.3, 0.7, 41)
                bal_accs = []
                for t in thrs:
                    pred_t = (y_prob >= t).astype(int)
                    bal_accs.append(balanced_accuracy_score(y[te], pred_t))
                best_idx = int(np.argmax(bal_accs))
                best_thr = float(thrs[best_idx])
                y_pred = (y_prob >= best_thr).astype(int)
            else:
                best_thr = 0.5

            m = fold_metrics(y[te], y_pred, y_prob)
            m["best_thr"] = best_thr
            m["fold"] = fold
            m.update(best_params)
            fold_rows.append(m)

        fold_df = pd.DataFrame(fold_rows)
        if "fold" in fold_df.columns:
            fold_df = fold_df.sort_values("fold")
        fold_df.to_csv(Path(args.outdir, f"{name}_fold_metrics.csv"), index=False)

        try:
            cur_final = Pipeline([("pre", pre_used), ("clf", clf)])
            n_q_all = int(min(256, max(16, len(X))))
            try:
                cur_final.set_params(pre__basis__quantile__n_quantiles=n_q_all)
            except ValueError:
                pass
            if name == "logreg":
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                gs_all = GridSearchCV(cur_final, param_grid={"clf__C": [0.1, 0.3, 1.0, 3.0]},
                                       cv=inner, scoring="balanced_accuracy", n_jobs=-1)
                gs_all.fit(X, y)
                cur_final = gs_all.best_estimator_
            elif name == "hgb" and hgb_grid:
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                gs_all = GridSearchCV(cur_final, param_grid=hgb_grid, cv=inner,
                                       scoring="balanced_accuracy", n_jobs=-1)
                gs_all.fit(X, y)
                cur_final = gs_all.best_estimator_
            elif name == "rf" and rf_grid:
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                gs_all = GridSearchCV(cur_final, param_grid=rf_grid, cv=inner,
                                       scoring="balanced_accuracy", n_jobs=-1)
                gs_all.fit(X, y)
                cur_final = gs_all.best_estimator_
            else:
                cur_final.fit(X, y)
            # Get feature names after preprocessing
            cur_final.named_steps["pre"].fit(X)
            feat_names = []
            if num_cols_no_basis:
                feat_names += num_cols_no_basis
            uses_bins = ("basis" in cur_final.named_steps["pre"].transformers_
                         and hasattr(cur_final.named_steps["pre"].named_transformers_["basis"], "named_steps")
                         and "kbins" in cur_final.named_steps["pre"].named_transformers_["basis"].named_steps)
            if basis_cols:
                if uses_bins:
                    for c in basis_cols:
                        feat_names += [f"{c}_bin_{i}" for i in range(3)]
                else:
                    feat_names += basis_cols
            if cat_cols:
                ohe = cur_final.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
                feat_names += list(ohe.get_feature_names_out(cat_cols))
            # Permutation importance on a 20% slice to keep it quick
            idx = np.random.RandomState(42).choice(np.arange(len(X)), size=max(100, len(X)//5), replace=False)
            perm = permutation_importance(cur_final, X.iloc[idx], y[idx], n_repeats=5, scoring="balanced_accuracy", random_state=42, n_jobs=-1)
            pim = pd.DataFrame({"feature": feat_names, "perm_importance": perm.importances_mean}).sort_values("perm_importance", ascending=False)
            pim.to_csv(Path(args.outdir, f"{name}_perm_importance.csv"), index=False)
        except Exception as e:
            with open(Path(args.outdir, f"{name}_perm_importance_ERROR.txt"), "w") as f:
                f.write(str(e))

        agg = fold_df.mean(numeric_only=True).to_dict()
        summary[name] = {**agg, "n_features": X.shape[1], "features": selected_cols, "basis_cols": basis_cols, "cat_cols": cat_cols}

        # crude feature importances/coefs
        try:
            pre_used = pre_trees if (args.trees_raw_basis and name in ("rf","hgb")) else pre_full
            pipe = Pipeline([("pre", pre_used), ("clf", clf)])
            if name == "logreg":
                # map back to preprocessed feature names
                pipe.named_steps["pre"].fit(X)  # ensure fitted for feature names
                feat_names = []
                if num_cols_no_basis:
                    feat_names += num_cols_no_basis
                uses_bins = ("basis" in pipe.named_steps["pre"].transformers_
                             and hasattr(pipe.named_steps["pre"].named_transformers_["basis"], "named_steps")
                             and "kbins" in pipe.named_steps["pre"].named_transformers_["basis"].named_steps)
                if basis_cols:
                    if uses_bins:
                        for c in basis_cols:
                            feat_names += [f"{c}_bin_{i}" for i in range(3)]
                    else:
                        feat_names += basis_cols
                if cat_cols:
                    ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
                    feat_names += list(ohe.get_feature_names_out(cat_cols))
                coefs = pipe.named_steps["clf"].coef_.ravel()
                imp = pd.DataFrame({"feature": feat_names, "weight": coefs}).sort_values("weight", key=np.abs, ascending=False)
                imp.to_csv(Path(args.outdir, f"{name}_feature_weights.csv"), index=False)
            else:
                # Tree-based
                pipe.fit(X, y)  # fit on all for a single global importance (ok for inspection)
                importances = pipe.named_steps["clf"].feature_importances_
                # names after preprocessing:
                pipe.named_steps["pre"].fit(X)
                feat_names = []
                if num_cols_no_basis:
                    feat_names += num_cols_no_basis
                uses_bins = ("basis" in pipe.named_steps["pre"].transformers_
                             and hasattr(pipe.named_steps["pre"].named_transformers_["basis"], "named_steps")
                             and "kbins" in pipe.named_steps["pre"].named_transformers_["basis"].named_steps)
                if basis_cols:
                    if uses_bins:
                        for c in basis_cols:
                            feat_names += [f"{c}_bin_{i}" for i in range(3)]
                    else:
                        feat_names += basis_cols
                if cat_cols:
                    ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
                    feat_names += list(ohe.get_feature_names_out(cat_cols))
                imp = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
                imp.to_csv(Path(args.outdir, f"{name}_feature_importances.csv"), index=False)
        except Exception as e:
            with open(Path(args.outdir, f"{name}_importances_ERROR.txt"), "w") as f:
                f.write(str(e))

    with open(Path(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary (mean across folds) ===")
    for k, v in summary.items():
        print(f"{k:7s} -> acc={v.get('acc',np.nan):.3f}  bal_acc={v.get('bal_acc',np.nan):.3f}  f1_macro={v.get('f1_macro',np.nan):.3f}  roc_auc={v.get('roc_auc',np.nan):.3f}")

if __name__ == "__main__":
    main()