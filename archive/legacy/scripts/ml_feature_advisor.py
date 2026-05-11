#!/usr/bin/env python3
import argparse
import math
import os
import sys
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import warnings

import time
import shutil

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from collections import Counter


# ------------------------------
# Utilities
# ------------------------------

class EmoLogger:
    """Lightweight console logger with emoji tags and a simple progress bar."""
    def __init__(self, use_emoji: bool = True):
        self.use_emoji = use_emoji

    def _tag(self, symbol: str, fallback: str) -> str:
        return symbol if self.use_emoji else fallback

    def info(self, msg: str) -> None:
        print(f"{self._tag('ℹ️', '[i]')} {msg}")

    def warn(self, msg: str) -> None:
        print(f"{self._tag('⚠️', '[!]')} {msg}")

    def success(self, msg: str) -> None:
        print(f"{self._tag('✅', '[ok]')} {msg}")

    def step(self, msg: str) -> None:
        print(f"{self._tag('▶️', '[>]')} {msg}")

    def write(self, msg: str) -> None:
        print(msg)

    def progress(self, curr: int, total: int, prefix: str = "") -> None:
        total = max(total, 1)
        width = shutil.get_terminal_size((80, 20)).columns
        bar_len = max(10, min(30, width - len(prefix) - 20))
        frac = max(0.0, min(1.0, curr / total))
        filled = int(bar_len * frac)
        bar = "█" * filled + "-" * (bar_len - filled)
        end = "\n" if curr >= total else "\r"
        print(f"{prefix} [{bar}] {curr}/{total}", end=end, flush=True)

LEAKY_NAME_HINTS = [
    "target","label","y","outcome","result","correct","error","score","feedback",
    "choice","gain","loss","reward","penalty","grade","approved","rejected"
]

NUMERIC_KINDS = ('i','u','f','b')  # int, unsigned, float, bool

GROUP_NAME_HINTS = [
    "participant","subject","user","customer","account","session",
    "household","patient","group","batch","cohort","pid","uid"
]


def is_group_like(s: pd.Series, name: str, min_members_per_group: int = 2, max_unique_ratio: float = 0.9) -> bool:
    ur = unique_ratio(s)
    if ur == 0.0 or ur >= max_unique_ratio:
        return False
    counts = pd.Series(s).value_counts(dropna=True)
    avg = float(counts.mean()) if len(counts) else 0.0
    lower = name.lower()
    name_hint = any(h in lower for h in GROUP_NAME_HINTS)
    return (avg >= min_members_per_group) and (name_hint or ur <= 0.5)

def is_numeric_series(s: pd.Series) -> bool:
    return s.dtype.kind in NUMERIC_KINDS

def safe_factorize(series: pd.Series) -> Tuple[np.ndarray, Dict[int, str]]:
    """Factorize any series to integers, preserving NaN as -1."""
    vals, uniques = pd.factorize(series, sort=True, na_sentinel=-1)
    mapping = {i: str(u) for i,u in enumerate(uniques)}
    return vals, mapping

def detect_target_type(y: pd.Series) -> str:
    """Return 'binary', 'multiclass', or 'regression'."""
    if y.dropna().nunique() == 2:
        return "binary"
    # Heuristic: integer with many unique or float -> regression
    if is_numeric_series(y) and y.nunique(dropna=True) > 10:
        return "regression"
    # Otherwise classification
    return "multiclass"

def try_parse_datetime(s: pd.Series, sample_size: int = 500, success_ratio: float = 0.85) -> bool:
    """Heuristically detect datetime-like columns without emitting pandas warnings.
    - If dtype is already datetime64[ns]/datetime-like → True
    - For object dtype: sample up to `sample_size` non-null values and attempt a
      vectorized parse with errors='coerce'. Consider it datetime-like if at
      least `success_ratio` of the sample parses successfully.
    """
    # Fast path: native datetime dtype
    if "datetime" in str(s.dtype) or getattr(s.dtype, "kind", None) == "M":
        return True

    # Only attempt parsing for object-like columns
    if s.dtype != object:
        return False

    # Sample a subset to avoid expensive full-column parsing
    sample = s.dropna().astype(str)
    if sample.empty:
        return False
    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    ok_ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
    return ok_ratio >= success_ratio

def percent_missing(s: pd.Series) -> float:
    return float(s.isna().mean()) * 100.0

def unique_ratio(s: pd.Series) -> float:
    n = len(s)
    if n == 0:
        return 0.0
    return float(s.nunique(dropna=True)) / float(n)

def avg_text_length(s: pd.Series) -> float:
    """Average string length for object dtype; 0 if not object."""
    if s.dtype == object:
        return float(s.fillna("").astype(str).str.len().mean())
    return 0.0


def detect_text_like(s: pd.Series) -> bool:
    """Heuristic: long-ish object strings with many unique values."""
    if s.dtype != object:
        return False
    return (avg_text_length(s) >= 30.0) and (unique_ratio(s) >= 0.3)


def summarize_target(y: pd.Series) -> Dict[str, object]:
    """Basic target summary used for insights and baselines."""
    ttype = detect_target_type(y)
    out: Dict[str, object] = {"type": ttype}
    if ttype in ("binary", "multiclass"):
        counts = y.value_counts(dropna=True)
        total = int(counts.sum())
        dist = {str(k): int(v) for k, v in counts.items()}
        majority = counts.max() / total if total > 0 else float("nan")
        out.update({
            "n_classes": int(y.nunique(dropna=True)),
            "class_distribution": dist,
            "majority_baseline_acc": float(majority),
        })
    else:
        y_num = pd.to_numeric(y, errors='coerce')
        mean_val = float(y_num.mean()) if y_num.notna().any() else float("nan")
        mae_mean = float((y_num - mean_val).abs().mean()) if y_num.notna().any() else float("nan")
        out.update({
            "mean_baseline": mean_val,
            "mae_mean_baseline": mae_mean,
        })
    return out


def split_column_roles(df: pd.DataFrame, features: List[str], high_cardinality_cat: int) -> Dict[str, List[str]]:
    """Return dict with lists: numeric, low_card_cat, high_card_cat, datetime_like, text_like."""
    numeric, low_cat, high_cat, dt_cols, text_cols = [], [], [], [], []
    for c in features:
        s = df[c]
        if is_numeric_series(s):
            numeric.append(c)
        else:
            if try_parse_datetime(s) or ("datetime" in str(s.dtype)):
                dt_cols.append(c)
            elif detect_text_like(s):
                text_cols.append(c)
            else:
                # treat as categorical
                (low_cat if s.nunique(dropna=True) <= high_cardinality_cat else high_cat).append(c)
    return {
        "numeric": numeric,
        "low_card_cat": low_cat,
        "high_card_cat": high_cat,
        "datetime_like": dt_cols,
        "text_like": text_cols,
    }


def suggest_cv_strategy(problem: str, y: pd.Series, group_col: pd.Series | None) -> Dict[str, str]:
    """Suggest CV strategy and provide a short justification."""
    if group_col is not None:
        ng = int(pd.Series(group_col).nunique(dropna=True))
        n_splits = 5 if ng >= 5 else max(2, ng)
        return {
            "cv": f"GroupKFold(n_splits={n_splits})",
            "why": f"Avoid leakage across groups (unique groups: {ng}).",
        }
    if problem in ("binary", "multiclass"):
        classes = int(y.nunique(dropna=True))
        return {
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "why": f"Preserves class distribution across folds (classes: {classes}).",
        }
    return {
        "cv": "KFold(n_splits=5, shuffle=True, random_state=42)",
        "why": "Standard CV for regression/continuous targets.",
    }


def build_sklearn_template(problem: str, roles: Dict[str, List[str]], target: str, group: str | None) -> str:
    """Return a runnable sklearn training template as a string."""
    import textwrap
    numeric = roles.get("numeric", [])
    low_cat = roles.get("low_card_cat", [])
    dt_cols = roles.get("datetime_like", [])
    text_cols = roles.get("text_like", [])

    # We'll suggest dropping datetime/text-like for a first baseline and note alternatives in comments.
    model_line = (
        "RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)"
        if problem in ("binary", "multiclass") else
        "RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)"
    )

    tmpl = f'''#!/usr/bin/env python3
# Auto-generated by ml_feature_advisor.py
# A simple, safe baseline pipeline you can adapt.

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, {('StratifiedKFold' if problem in ('binary','multiclass') else 'KFold')}, GroupKFold, cross_val_score
from sklearn.metrics import {('accuracy_score, balanced_accuracy_score, f1_macro' if problem in ('binary','multiclass') else 'mean_absolute_error, r2_score')}

# Load your data here
# df = pd.read_csv('PATH_TO_YOUR_DATA.csv')

TARGET = "{target}"
GROUP = {repr(group) if group else 'None'}

feature_cols = [c for c in df.columns if c != TARGET and (GROUP is None or c != GROUP)]
X = df[feature_cols]
y = df[TARGET]

numeric_features = {numeric}
low_card_cat_features = {low_cat}
# NOTE: datetime-like columns {dt_cols} and text-like columns {text_cols} are ignored in this baseline.
# For datetime, consider extracting year/month/day/weekday/hour.
# For text, consider TF-IDF with linear models.

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler(with_mean=False)),  # safe with sparse output downstream
])

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", cat_pipe, low_card_cat_features),
], remainder='drop')

model = {model_line}

pipe = Pipeline([
    ("prep", preprocess),
    ("model", model),
])

# Suggested CV
if GROUP is not None and GROUP in df.columns:
    cv = GroupKFold(n_splits=5)
    scores = cross_val_score(pipe, X, y, cv=cv, groups=df[GROUP], scoring={('"balanced_accuracy"' if problem in ('binary','multiclass') else '"neg_mean_absolute_error"')})
else:
    cv = {('StratifiedKFold(n_splits=5, shuffle=True, random_state=42)' if problem in ('binary','multiclass') else 'KFold(n_splits=5, shuffle=True, random_state=42)')}
    scores = cross_val_score(pipe, X, y, cv=cv, scoring={('"balanced_accuracy"' if problem in ('binary','multiclass') else '"neg_mean_absolute_error"')})

print("CV scores:", scores)
print("Mean:", scores.mean())
'''
    return textwrap.dedent(tmpl)

def score_target_candidates(df: pd.DataFrame, max_classes: int = 30) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        info = {
            "column": col,
            "dtype": str(s.dtype),
            "missing_pct": percent_missing(s),
            "n_unique": int(s.nunique(dropna=True)),
            "unique_ratio": unique_ratio(s),
            "is_numeric": is_numeric_series(s),
            "is_datetime_like": try_parse_datetime(s) if s.dtype == object or "datetime" in str(s.dtype) else ("datetime" in str(s.dtype)),
            "id_like": is_id_like(s, col),
            "text_like": detect_text_like(s),
        }
        # Classification candidacy
        cls_score = -np.inf
        if (not info["is_numeric"]) or (info["n_unique"] <= max_classes):
            k = info["n_unique"]
            if k >= 2:
                dist = s.value_counts(normalize=True, dropna=True)
                if len(dist):
                    majority = float(dist.max())
                else:
                    majority = 1.0
                cls_score = 1.0
                cls_score -= 0.5 * majority                  # prefer balanced
                cls_score -= 0.5 * (info["missing_pct"] / 100.0)
                if info["id_like"]: cls_score -= 0.6
                if info["text_like"]: cls_score -= 0.4
                if k > 10: cls_score -= min((k - 10) / max_classes, 0.5)
        # Regression candidacy
        reg_score = -np.inf
        if info["is_numeric"]:
            k = info["n_unique"]
            ur = info["unique_ratio"]
            if k > 10 and ur < 0.99:
                s_num = pd.to_numeric(s, errors='coerce')
                var = float(s_num.var()) if s_num.notna().any() else 0.0
                reg_score = 0.5 * np.tanh(var if np.isfinite(var) else 0.0)
                reg_score += 0.3 * (1.0 - abs(ur - 0.5))      # mid uniqueness preferred
                reg_score -= 0.5 * (info["missing_pct"] / 100.0)
                if info["id_like"]: reg_score -= 0.8
        best_type = "classification" if cls_score >= reg_score else "regression"
        best_score = float(max(cls_score, reg_score))
        info.update({
            "candidate_type": best_type,
            "cls_score": float(cls_score),
            "reg_score": float(reg_score),
            "target_score": best_score,
        })
        rows.append(info)
    cand = pd.DataFrame(rows)
    cand = cand.sort_values("target_score", ascending=False)
    return cand


def suggest_group_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        if detect_text_like(s) or is_id_like(s, col):
            continue
        if not (is_numeric_series(s) or s.dtype == object):
            continue
        ur = unique_ratio(s)
        if ur == 0.0 or ur >= 0.99:
            continue
        counts = pd.Series(s).value_counts(dropna=True)
        avg = float(counts.mean()) if len(counts) else 0.0
        hint = is_group_like(s, col)
        score = 0.0
        score += 0.4 * (1.0 - ur)              # more repeats -> better
        score += 0.3 * (avg / (avg + 10.0))    # larger avg group size -> better
        score += 0.3 if hint else 0.0          # name hint bonus
        rows.append({
            "column": col,
            "unique_ratio": float(ur),
            "avg_group_size": float(avg),
            "name_hint": bool(hint),
            "group_score": float(score),
        })
    if not rows:
        return pd.DataFrame(columns=["column","unique_ratio","avg_group_size","name_hint","group_score"]).astype({})
    return pd.DataFrame(rows).sort_values("group_score", ascending=False)

def is_id_like(s: pd.Series, name: str, unique_ratio_threshold: float=0.95) -> bool:
    if unique_ratio(s) >= unique_ratio_threshold:
        return True
    lower = name.lower()
    return any(k in lower for k in ["id","uuid","guid","hash"])

def is_high_cardinality_categorical(s: pd.Series, cardinality_threshold: int=50) -> bool:
    if is_numeric_series(s):
        return False
    return s.nunique(dropna=True) > cardinality_threshold

def pairwise_correlations(df_num: pd.DataFrame, threshold: float=0.95) -> List[Tuple[str,str,float]]:
    corrs = df_num.corr(numeric_only=True).abs()
    flagged = []
    cols = corrs.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            val = corrs.iloc[i,j]
            if pd.notna(val) and val >= threshold:
                flagged.append((cols[i], cols[j], float(val)))
    return flagged

def compute_mutual_info(X: pd.DataFrame, y: pd.Series, problem: str, debug: bool=False) -> np.ndarray:
    # Encode X (numeric → filled; categorical → factorize; ensure finite)
    X_enc = pd.DataFrame(index=X.index)
    for col in X.columns:
        s = X[col]
        try:
            if is_numeric_series(s):
                if s.notna().any():
                    med = s.median()
                    s2 = s.fillna(med)
                else:
                    s2 = s.fillna(0)
                X_enc[col] = pd.to_numeric(s2, errors="coerce").fillna(0.0).astype(float)
            else:
                vals, _ = safe_factorize(s)
                X_enc[col] = pd.Series(vals, index=X.index).astype(float)
        except Exception as e:
            if debug:
                print(f"[compute_mutual_info] Failed to encode column '{col}': {e}")
            X_enc[col] = 0.0

    # y encoding
    try:
        if problem in ("binary","multiclass"):
            if not is_numeric_series(y):
                y_enc, _ = safe_factorize(y)
            else:
                y_enc = pd.to_numeric(y, errors="coerce")
                # If all-NaN or constant after coercion, MI is undefined
                if not y_enc.notna().any() or y_enc.nunique(dropna=True) <= 1:
                    if debug:
                        print("[compute_mutual_info] y is empty or constant after coercion; returning zeros.")
                    return np.zeros(X_enc.shape[1])
                y_enc = y_enc.fillna(method="ffill").fillna(method="bfill").astype(int).values
            mi = mutual_info_classif(
                X_enc.values, y_enc,
                discrete_features="auto",
                random_state=42
            )
        else:
            y_enc = pd.to_numeric(y, errors="coerce").fillna(0.0).astype(float).values
            mi = mutual_info_regression(X_enc.values, y_enc, random_state=42)
        # Replace any nan/inf that might slip through
        mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
        return mi
    except Exception as e:
        if debug:
            import traceback
            print(f"[compute_mutual_info] MI computation error: {e}")
            traceback.print_exc()
        return np.zeros(X_enc.shape[1])

def score_feature_quality(df: pd.DataFrame, target: str, group: str=None,
                          high_missing_pct: float=40.0,
                          near_constant_unique_ratio: float=0.01,
                          high_cardinality_cat: int=50,
                          leaky_name_hints: List[str]=None,
                          corr_redundancy_threshold: float=0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if leaky_name_hints is None:
        leaky_name_hints = LEAKY_NAME_HINTS

    assert target in df.columns, f"Target column '{target}' not found"

    # Basic profile
    y = df[target]
    problem = detect_target_type(y)

    # Candidate features = all except target and group
    drop_cols = [target] + ([group] if group and group in df.columns else [])
    features = [c for c in df.columns if c not in drop_cols]

    roles = split_column_roles(df, features, high_cardinality_cat)

    profile_rows = []
    for col in features:
        s = df[col]
        info = {
            "feature": col,
            "dtype": str(s.dtype),
            "missing_pct": percent_missing(s),
            "n_unique": int(s.nunique(dropna=True)),
            "unique_ratio": unique_ratio(s),
        }
        info["is_numeric"] = is_numeric_series(s)
        info["is_datetime_like"] = try_parse_datetime(s) if s.dtype == object or "datetime" in str(s.dtype) else ("datetime" in str(s.dtype))
        info["is_constant"] = (s.nunique(dropna=True) <= 1)
        info["near_constant"] = (info["unique_ratio"] <= near_constant_unique_ratio)
        info["id_like"] = is_id_like(s, col)
        info["high_cardinality_cat"] = is_high_cardinality_categorical(s, high_cardinality_cat)
        info["text_like"] = detect_text_like(s)
        info["avg_text_len"] = avg_text_length(s)
        # name-based leaky hints
        lower = col.lower()
        info["leaky_name_match"] = any(h in lower for h in leaky_name_hints)
        profile_rows.append(info)

    profile = pd.DataFrame(profile_rows)

    # Mutual Information score with target (robust univariate relevance)
    try:
        mi = compute_mutual_info(df[features], y, problem)
    except Exception as e:
        mi = np.array([np.nan]*len(features))
    profile["mutual_info"] = mi

    # Redundancy via high correlation among numeric features
    num_cols = [c for c in features if is_numeric_series(df[c])]
    high_corr_pairs = []
    if len(num_cols) >= 2:
        high_corr_pairs = pairwise_correlations(df[num_cols], corr_redundancy_threshold)

    # Aggregate redundancy flags
    redundant_set = set()
    for a,b,r in high_corr_pairs:
        # keep the one with higher MI (if available), drop the other
        mia = profile.loc[profile["feature"]==a, "mutual_info"].values[0]
        mib = profile.loc[profile["feature"]==b, "mutual_info"].values[0]
        if np.isnan(mia) and np.isnan(mib):
            # arbitrary: mark 'b' as redundant
            redundant_set.add(b)
        else:
            if np.nan_to_num(mia, nan=-1) >= np.nan_to_num(mib, nan=-1):
                redundant_set.add(b)
            else:
                redundant_set.add(a)
    profile["redundant_by_corr"] = profile["feature"].isin(redundant_set)

    # Target & CV suggestion
    t_summary = summarize_target(y)
    cv_suggestion = suggest_cv_strategy(problem, y, df[group] if group and group in df.columns else None)

    # Recommendation logic
    recs = []
    for _, row in profile.iterrows():
        flags = []
        if row.get("text_like", False):
            flags.append("text-like")
        if row["is_constant"] or row["near_constant"]:
            flags.append("constant/near-constant")
        if row["missing_pct"] >= high_missing_pct:
            flags.append(f"high-missing({row['missing_pct']:.1f}%)")
        if row["id_like"]:
            flags.append("id-like(unique)")
        if row["high_cardinality_cat"]:
            flags.append("high-cardinality-categorical")
        if row["is_datetime_like"]:
            flags.append("datetime-like")
        if row["leaky_name_match"]:
            flags.append("name-suggests-leakage")
        if row["redundant_by_corr"]:
            flags.append("redundant-high-corr")

        # Score heuristic
        score = 0.0
        if not math.isnan(row["mutual_info"]):
            score += row["mutual_info"]
        # penalties
        penalty = 0.0
        if "constant/near-constant" in flags: penalty += 1.0
        if "id-like(unique)" in flags: penalty += 0.5
        if "high-cardinality-categorical" in flags: penalty += 0.3
        if "datetime-like" in flags: penalty += 0.3
        if "name-suggests-leakage" in flags: penalty += 1.0
        if "redundant-high-corr" in flags: penalty += 0.4
        if row["missing_pct"] >= high_missing_pct: penalty += 0.5
        if "text-like" in flags: penalty += 0.2

        net = score - penalty

        # Recommendation buckets
        if ("constant/near-constant" in flags) or ("id-like(unique)" in flags) or ("name-suggests-leakage" in flags):
            rec = "drop"
        elif net <= 0:
            rec = "maybe"
        else:
            rec = "keep"

        recs.append((row["feature"], rec, net, ";".join(flags)))

    rec_df = pd.DataFrame(recs, columns=["feature","recommendation","net_score","flags"])
    result = profile.merge(rec_df, on="feature", how="left").sort_values(by=["recommendation","net_score"], ascending=[True, False])

    # Build a human-readable summary
    summary_lines = []
    summary_lines.append(f"# ML Feature Advisor Report")
    summary_lines.append(f"- Rows: {len(df):,} | Columns (incl. target): {df.shape[1]}")
    summary_lines.append(f"- Target: {target} | Problem type: {problem}")
    if group and group in df.columns:
        summary_lines.append(f"- Group column detected: {group} (use for GroupKFold splits)")
    summary_lines.append("")

    # Dataset overview
    total_missing = float(df.isna().mean().mean()) * 100.0
    summary_lines.append("## Dataset Overview")
    summary_lines.append(f"- Overall missingness (avg across columns): {total_missing:.1f}%")
    summary_lines.append(f"- Numeric features: {len(roles['numeric'])} | Low-card categorical: {len(roles['low_card_cat'])} | High-card categorical: {len(roles['high_card_cat'])}")
    summary_lines.append(f"- Datetime-like: {len(roles['datetime_like'])} | Text-like: {len(roles['text_like'])}")
    summary_lines.append("")

    # Target summary & baselines
    summary_lines.append("## Target Summary & Baselines")
    if t_summary["type"] in ("binary","multiclass"):
        dist = ", ".join([f"{k}: {v}" for k,v in t_summary.get("class_distribution", {}).items()])
        summary_lines.append(f"- Classes: {t_summary.get('n_classes')} | Distribution: {dist}")
        summary_lines.append(f"- Majority-class baseline accuracy: {t_summary.get('majority_baseline_acc'):.3f}")
        summary_lines.append("- Recommended metrics: accuracy, balanced accuracy, macro-F1")
    else:
        summary_lines.append(f"- Mean baseline: {t_summary.get('mean_baseline'):.4f} | MAE of mean baseline: {t_summary.get('mae_mean_baseline'):.4f}")
        summary_lines.append("- Recommended metrics: MAE (primary), R^2 (secondary)")
    summary_lines.append("")

    # CV suggestion
    summary_lines.append("## Train/Validation Plan Suggestion")
    summary_lines.append(f"- CV: {cv_suggestion['cv']}")
    summary_lines.append(f"- Why: {cv_suggestion['why']}")
    summary_lines.append("")

    # Preprocessing playbook
    summary_lines.append("## Preprocessing Playbook (Suggested)")
    summary_lines.append("- Numeric → median impute; StandardScaler for linear/SVM; trees can skip scaling.")
    summary_lines.append(f"- Low-card categorical (≤ {high_cardinality_cat}) → OneHotEncoder(handle_unknown='ignore').")
    summary_lines.append("- High-card categorical → consider count/frequency encoding or hashing encoder; beware of leakage with target encoding.")
    summary_lines.append("- Datetime-like → extract year/month/day/weekday/hour; optionally time since start/event windows.")
    summary_lines.append("- Text-like → TF-IDF + linear model as separate pipeline; or sentence embeddings with caution.")
    summary_lines.append("- Handle class imbalance (if any) via class_weight='balanced' or simple up/down-sampling.")
    summary_lines.append("")

    # Recommendations counts
    summary_lines.append(f"## Recommendations")
    summary_lines.append(result["recommendation"].value_counts().to_string())
    summary_lines.append("")

    summary_lines.append("## Top 15 features by mutual information (higher = more predictive)")
    top_mi = result[["feature","mutual_info"]].dropna().sort_values("mutual_info", ascending=False).head(15)
    summary_lines.append(top_mi.to_string(index=False))
    summary_lines.append("")

    # Common flags
    summary_lines.append("## Common flags (counts)")
    flags_series = result["flags"].fillna("").str.split(";").explode()
    flags_counts = flags_series[flags_series!=""].value_counts()
    if len(flags_counts):
        summary_lines.append(flags_counts.to_string())
    else:
        summary_lines.append("(no flags)")

    # Actionable drops/keeps
    summary_lines.append("")
    summary_lines.append("## Suggested Drops (top 10 by penalty/leakiness)")
    drop_cands = result[result["recommendation"]=="drop"].copy()
    if not drop_cands.empty:
        drop_cands = drop_cands.sort_values(["net_score","mutual_info"], ascending=[True, False]).head(10)
        summary_lines.append(drop_cands[["feature","flags","mutual_info","missing_pct"]].to_string(index=False))
    else:
        summary_lines.append("(none)")

    summary_lines.append("")
    summary_lines.append("## Top 'Keep' Candidates (by net score)")
    keep_top = result[result["recommendation"]=="keep"].sort_values("net_score", ascending=False).head(10)
    if not keep_top.empty:
        summary_lines.append(keep_top[["feature","net_score","mutual_info","flags"]].to_string(index=False))
    else:
        summary_lines.append("(none)")

    summary_md = "\n".join(summary_lines)

    return result, summary_md


def main():
    ap = argparse.ArgumentParser(description="Non-destructive ML feature advisor: score columns, flag issues, and suggest keep/maybe/drop.")
    ap.add_argument("--data", required=True, help="Path to input dataset (.csv or .parquet)")
    ap.add_argument("--target", required=False, help="Target column name (optional; if omitted, auto-detect candidates)")
    ap.add_argument("--group", default=None, help="Optional group column (e.g., participant_id)")
    ap.add_argument("--out", default="reports/advisor/feature_advisor_report", help="Output file prefix (without extension; default saves under reports/advisor)")
    ap.add_argument("--missing-threshold", type=float, default=40.0, help="Percent missing to flag as high (default 40%)")
    ap.add_argument("--unique-ratio-threshold", type=float, default=0.01, help="Unique ratio to flag near-constant (default 0.01)")
    ap.add_argument("--high-cardinality", type=int, default=50, help="Cardinality threshold for categorical (default 50)")
    ap.add_argument("--corr-threshold", type=float, default=0.95, help="Correlation threshold to flag redundancy (default 0.95)")
    ap.add_argument("--auto-topk", type=int, default=3, help="When auto-detecting, generate reports for top-K target candidates (default 3)")
    ap.add_argument("--auto-group", action="store_true", help="Also auto-suggest group columns (participant/session/etc.)")
    args = ap.parse_args()
    logger = EmoLogger(use_emoji=True)

    # Normalize output prefix to reports/advisor/<dataset_name>/<out_base>
    data_base = os.path.splitext(os.path.basename(args.data))[0]
    dataset_name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in data_base)
    out_base = os.path.basename(args.out)
    dataset_dir = os.path.join("reports", "advisor", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    args.out = os.path.join(dataset_dir, out_base)
    logger.info(f"Saving reports under: {dataset_dir}")

    # Load
    if args.data.lower().endswith(".csv"):
        df = pd.read_csv(args.data)
    elif args.data.lower().endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        print("ERROR: Please provide .csv or .parquet file", file=sys.stderr)
        sys.exit(2)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.target is None:
        # AUTO-TARGET MODE
        target_cands = score_target_candidates(df)
        tc_csv = f"{args.out}_target_candidates.csv"
        target_cands.to_csv(tc_csv, index=False)

        # Optional: suggest group columns
        gc_csv = None
        if args.auto_group:
            group_cands = suggest_group_candidates(df)
            gc_csv = f"{args.out}_group_candidates.csv"
            group_cands.to_csv(gc_csv, index=False)

        logger.success(f"Wrote: {tc_csv}")
        if gc_csv:
            logger.success(f"Wrote: {gc_csv}")

        # Generate reports for top-K target candidates
        topk = max(0, int(args.auto_topk))
        logger.step(f"Evaluating top-{topk} target candidates…")
        total_cands = min(topk, len(target_cands))
        logger.write(f"Processing 0/{total_cands}")
        for idx, (_, row) in enumerate(target_cands.head(topk).iterrows(), start=1):
            cand_target = row["column"]
            t_start = time.time()
            logger.write(f"⏳ [{idx}/{total_cands}] {cand_target} …")
            logger.info(
                f"Evaluating candidate target: {cand_target} ({row['candidate_type']}, score={row['target_score']:.3f})"
            )
            r, smd = score_feature_quality(
                df, target=cand_target, group=None,
                high_missing_pct=args.missing_threshold,
                near_constant_unique_ratio=args.unique_ratio_threshold,
                high_cardinality_cat=args.high_cardinality,
                corr_redundancy_threshold=args.corr_threshold
            )
            # Save per-candidate outputs
            csv_path = f"{args.out}_{cand_target}.csv"
            md_path = f"{args.out}_{cand_target}.md"
            r.to_csv(csv_path, index=False)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(smd)
            # Baseline sklearn template for this candidate
            roles = split_column_roles(df, [c for c in df.columns if c != cand_target], args.high_cardinality)
            problem = detect_target_type(df[cand_target])
            template_code = build_sklearn_template(problem, roles, cand_target, None)
            py_tmpl_path = f"{args.out}_{cand_target}_sklearn_template.py"
            with open(py_tmpl_path, "w", encoding="utf-8") as f:
                f.write(template_code)
            logger.success(f"Wrote: {csv_path}")
            logger.success(f"Wrote: {md_path}")
            logger.success(f"Wrote: {py_tmpl_path}")
            # Mini report for logs
            _keep = int((r["recommendation"] == "keep").sum())
            _maybe = int((r["recommendation"] == "maybe").sum())
            _drop = int((r["recommendation"] == "drop").sum())
            logger.write(
                f"📊 Mini report for '{cand_target}': keep={_keep}, maybe={_maybe}, drop={_drop}"
            )
            elapsed = time.time() - t_start
            logger.write(f"✅ [{idx}/{total_cands}] {cand_target} finished in {elapsed:.1f}s")
        logger.success("Auto-target analysis complete.")
        logger.info("Tip: Re-run with --target <column> (and optionally --group <column>) once you select your target.")
        return

    # ORIGINAL path when target is provided
    result, summary_md = score_feature_quality(
        df, target=args.target, group=args.group,
        high_missing_pct=args.missing_threshold,
        near_constant_unique_ratio=args.unique_ratio_threshold,
        high_cardinality_cat=args.high_cardinality,
        corr_redundancy_threshold=args.corr_threshold
    )

    # Build and save a sklearn template for a quick baseline
    roles = split_column_roles(df, [c for c in df.columns if c not in ([args.target] + ([args.group] if args.group and args.group in df.columns else []))], args.high_cardinality)
    problem = detect_target_type(df[args.target])
    template_code = build_sklearn_template(problem, roles, args.target, args.group if args.group and args.group in df.columns else None)
    py_tmpl_path = f"{args.out}_sklearn_template.py"
    with open(py_tmpl_path, "w", encoding="utf-8") as f:
        f.write(template_code)

    # Save outputs
    csv_path = f"{args.out}.csv"
    md_path = f"{args.out}.md"
    result.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    logger.success(f"Wrote: {csv_path}")
    logger.success(f"Wrote: {md_path}")
    logger.success(f"Wrote: {py_tmpl_path}")
    # Print small tail to console
    keep_top = result[result["recommendation"]=="keep"].sort_values("net_score", ascending=False).head(10)
    logger.step("Top 'keep' candidates (preview)")
    logger.write(keep_top[["feature","net_score","mutual_info","flags"]].to_string(index=False))
    # Compact mini report in logs
    _keep = int((result["recommendation"] == "keep").sum())
    _maybe = int((result["recommendation"] == "maybe").sum())
    _drop = int((result["recommendation"] == "drop").sum())
    logger.write(f"📊 Mini report: keep={_keep}, maybe={_maybe}, drop={_drop}")
    logger.success("Done.")


if __name__ == "__main__":
    main()
