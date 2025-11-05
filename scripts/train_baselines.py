#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, glob
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib

ID_COLS = ["participant_id", "Block", "Trial", "global_trial_index"]

OUT_DIR = Path("reports/ml_baselines")
MODEL_DIR = Path("models/ml_baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def is_classification(target_name: str, y: pd.Series) -> bool:
    # Treat these as classification; others as regression
    if target_name in {"risky-choice", "behav_valence", "Chronotype"}:
        return True
    # Fallback: few unique values → likely classification
    nunique = y.dropna().nunique()
    return nunique <= 10 and y.dtype.name not in ("float32","float64")

def load_combo(meta_path: str):
    meta = json.load(open(meta_path))
    target = meta["target"]
    pack = meta["pack"]
    split = meta["split_mode"]
    base = meta_path.replace("__meta.json", "")
    train_csv = base + "__train.csv"
    test_csv  = base + "__test.csv"
    return target, pack, split, train_csv, test_csv

def split_xy(df: pd.DataFrame, target: str):
    assert target in df.columns, f"Missing target {target}"
    X = df.drop(columns=[target])
    # Drop IDs from features (kept only for traceability in saved CSVs)
    feat_cols = [c for c in X.columns if c not in ID_COLS]
    return X[feat_cols].copy(), df[target].copy(), feat_cols

def impute_numeric_cats(X_train, X_test):
    # Both sets may already be numeric after our builder; still impute safely.
    num_imp = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(num_imp.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(num_imp.transform(X_test), columns=X_test.columns)
    return X_train, X_test, {"num_imputer": {"strategy": "median"}}

def evaluate_classification(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    # Binary AUC if probabilities provided and exactly 2 classes
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
    return metrics

def evaluate_regression(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def save_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def run_one(meta_path: str):
    target, pack, split, train_csv, test_csv = load_combo(meta_path)
    tag = f"{target}__{pack}__{split}"
    print(f"\n=== {tag} ===")

    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)

    # Keep copies with IDs for saved predictions
    ids_train = train[ID_COLS]
    ids_test  = test[ID_COLS]

    X_train, y_train, feat_cols = split_xy(train, target)
    X_test,  y_test,  _         = split_xy(test, target)

    # Impute (safe guard)
    X_train, X_test, imp_meta = impute_numeric_cats(X_train, X_test)

    # Detect task
    task_cls = is_classification(target, y_train)

    reports_dir = OUT_DIR / tag
    models_dir  = MODEL_DIR / tag
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"target": target, "pack": pack, "split": split, "n_features": len(feat_cols)}
    all_results["feature_names"] = feat_cols
    all_results["imputation"] = imp_meta

    if task_cls:
        # Ensure labels are string-coded for robust metrics
        y_train_enc = y_train.astype(str)
        y_test_enc  = y_test.astype(str)
        classes = sorted(y_train_enc.unique().tolist())

        # 1) Logistic Regression
        logreg = LogisticRegression(max_iter=2000, n_jobs=None)
        logreg.fit(X_train, y_train_enc)
        pred_lr = logreg.predict(X_test)
        # Try probability for AUC (binary only)
        prob_lr = None
        if len(classes) == 2 and hasattr(logreg, "predict_proba"):
            prob_lr = logreg.predict_proba(X_test)[:, 1]

        res_lr = evaluate_classification(y_test_enc, pred_lr, prob_lr)
        all_results["logistic_regression"] = res_lr

        # 2) Random Forest
        rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train_enc)
        pred_rf = rf.predict(X_test)
        prob_rf = None
        if len(classes) == 2 and hasattr(rf, "predict_proba"):
            prob_rf = rf.predict_proba(X_test)[:, 1]
        res_rf = evaluate_classification(y_test_enc, pred_rf, prob_rf)
        all_results["random_forest"] = res_rf

        # Confusion matrix & class report for RF
        cm = confusion_matrix(y_test_enc, pred_rf, labels=classes)
        class_rep = classification_report(y_test_enc, pred_rf, output_dict=True, zero_division=0)
        all_results["confusion_matrix_labels"] = classes
        all_results["confusion_matrix_rf"] = cm.tolist()
        all_results["classification_report_rf"] = class_rep

        # Importances / coefficients
        try:
            all_results["rf_feature_importances"] = dict(zip(feat_cols, rf.feature_importances_.round(6)))
        except Exception:
            pass
        try:
            # Multi-class LR has coef_ shape [n_classes, n_features]
            coef = getattr(logreg, "coef_", None)
            if coef is not None:
                all_results["logreg_coef"] = {str(i): dict(zip(feat_cols, coef[i].round(6))) for i in range(coef.shape[0])}
        except Exception:
            pass

        # Save preds
        pred_df = ids_test.copy()
        pred_df["y_true"] = y_test_enc.values
        pred_df["lr_pred"] = pred_lr
        pred_df["rf_pred"] = pred_rf
        pred_df.to_csv(reports_dir / "predictions.csv", index=False)

        # Persist models
        joblib.dump(logreg, models_dir / "logreg.joblib")
        joblib.dump(rf, models_dir / "random_forest.joblib")

    else:
        # Regression
        # 1) Linear Regression
        lin = LinearRegression()
        lin.fit(X_train, y_train)
        pred_lin = lin.predict(X_test)
        res_lin = evaluate_regression(y_test, pred_lin)
        all_results["linear_regression"] = res_lin

        # 2) Random Forest Regressor
        rfr = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        rfr.fit(X_train, y_train)
        pred_rfr = rfr.predict(X_test)
        res_rfr = evaluate_regression(y_test, pred_rfr)
        all_results["random_forest"] = res_rfr

        # Coef / importances
        try:
            all_results["lin_coef"] = dict(zip(feat_cols, lin.coef_.round(6)))
        except Exception:
            pass
        try:
            all_results["rf_feature_importances"] = dict(zip(feat_cols, rfr.feature_importances_.round(6)))
        except Exception:
            pass

        # Save preds
        pred_df = ids_test.copy()
        pred_df["y_true"] = y_test.values
        pred_df["lin_pred"] = pred_lin
        pred_df["rf_pred"] = pred_rfr
        pred_df.to_csv(reports_dir / "predictions.csv", index=False)

        # Persist models
        joblib.dump(lin, models_dir / "linear_regression.joblib")
        joblib.dump(rfr, models_dir / "random_forest.joblib")

    # Save summary JSON
    save_json(all_results, reports_dir / "metrics.json")
    print(f"✅ Wrote {reports_dir}/metrics.json and saved models.")

def main():
    metas = sorted(glob.glob("data/processed/ml_subsets/*__meta.json"))
    if not metas:
        raise SystemExit("No meta files found under data/processed/ml_subsets/")
    for m in metas:
        try:
            run_one(m)
        except Exception as e:
            print(f"❌ Failed on {m}: {e}")

if __name__ == "__main__":
    main()