#!/usr/bin/env python3
import re, json, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def parse_feature(name: str):
    # Expect like: num__Cz_P300 or num__Cz_P300__std or num__Fz_FRN
    m = re.match(r'^(?:num|cat)__(?P<ch>[A-Za-z0-9]+)_(?P<comp>P300|FRN)(?:__(?P<stat>std))?$', name)
    if not m:
        return {"channel":"OTHER","component":"OTHER","stat":"mean"}
    d = m.groupdict()
    return {
        "channel": d.get("ch") or "OTHER",
        "component": d.get("comp") or "OTHER",
        "stat": (d.get("stat") or "mean")
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default="reports/participant_chronotype")
    ap.add_argument("--data", default="data/processed/ml_ready_participant.csv")
    ap.add_argument("--permutation", action="store_true",
                    help="Compute permutation importance using best_model.joblib on full data.")
    args = ap.parse_args()

    outdir = Path(args.report_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load RF importances
    imp_path = outdir / "rf_importances.csv"
    df = pd.read_csv(imp_path)
    # Parse groups
    meta = df["feature"].apply(parse_feature).apply(pd.Series)
    dfi = pd.concat([df, meta], axis=1)

    # --- Grouped summaries
    by_comp = dfi.groupby("component")["importance"].sum().sort_values(ascending=False)
    by_stat = dfi.groupby("stat")["importance"].sum().sort_values(ascending=False)
    by_chan = dfi.groupby("channel")["importance"].sum().sort_values(ascending=False)

    # Save summary table
    summary = pd.DataFrame({
        "component_sum": by_comp,
    }).rename_axis("component").reset_index()
    summary_stat = by_stat.rename_axis("stat").reset_index().rename(columns={"importance":"stat_sum"})
    summary_chan = by_chan.rename_axis("channel").reset_index().rename(columns={"importance":"channel_sum"})
    summary.to_csv(outdir / "importance_by_component.csv", index=False)
    summary_stat.to_csv(outdir / "importance_by_stat.csv", index=False)
    summary_chan.to_csv(outdir / "importance_by_channel.csv", index=False)

    # Quick bar plots
    plt.figure()
    by_comp.plot(kind="bar")
    plt.title("RF Importances by Component")
    plt.ylabel("Sum of importances")
    plt.tight_layout()
    plt.savefig(outdir / "imp_by_component.png", dpi=160)
    plt.close()

    plt.figure()
    by_stat.plot(kind="bar")
    plt.title("RF Importances: mean vs std")
    plt.ylabel("Sum of importances")
    plt.tight_layout()
    plt.savefig(outdir / "imp_by_stat.png", dpi=160)
    plt.close()

    # Optional: permutation importance on the trained pipeline
    if args.permutation:
        import joblib
        import numpy as np
        from sklearn.inspection import permutation_importance

        model_path = outdir / "best_model.joblib"
        if not model_path.exists():
            print("best_model.joblib not found; skipping permutation importance.")
        else:
            model = joblib.load(model_path)
            data = pd.read_csv(args.data)
            target = "Chronotype"
            y = data[target].astype(str)
            X = data.drop(columns=[target, "participant_id"], errors="ignore")

            # Use pipeline directly; it will preprocess internally
            r = permutation_importance(model, X, y, scoring="balanced_accuracy",
                                       n_repeats=50, random_state=42, n_jobs=-1)
            # Map back to transformed feature names if available; otherwise fall back safely
            n_feats = int(len(r.importances_mean))
            feat_names = None
            try:
                pre = model.named_steps.get("pre") or model.named_steps.get("preprocessor")
                if pre is not None and hasattr(pre, "get_feature_names_out"):
                    fn = pre.get_feature_names_out()
                    if len(fn) == n_feats:
                        feat_names = np.array(fn, dtype=str)
            except Exception:
                pass

            # Fallbacks
            if feat_names is None:
                if X.shape[1] == n_feats:
                    feat_names = np.array(X.columns, dtype=str)
                else:
                    # last resort: generate generic names
                    feat_names = np.array([f"f{i}" for i in range(n_feats)], dtype=str)

            p_df = pd.DataFrame({
                "feature": feat_names,
                "perm_importance_mean": r.importances_mean,
                "perm_importance_std": r.importances_std
            }).sort_values("perm_importance_mean", ascending=False)
            p_df.to_csv(outdir / "rf_permutation_importances.csv", index=False)

            # Group permutation by component/stat too
            p_meta = p_df["feature"].apply(parse_feature).apply(pd.Series)
            p_full = pd.concat([p_df, p_meta], axis=1)
            p_by_comp = p_full.groupby("component")["perm_importance_mean"].sum().sort_values(ascending=False)
            p_by_stat = p_full.groupby("stat")["perm_importance_mean"].sum().sort_values(ascending=False)
            p_by_comp.to_csv(outdir / "perm_importance_by_component.csv")
            p_by_stat.to_csv(outdir / "perm_importance_by_stat.csv")

            plt.figure()
            p_by_comp.plot(kind="bar")
            plt.title("Permutation Importances by Component")
            plt.ylabel("Sum of perm importance (mean)")
            plt.tight_layout()
            plt.savefig(outdir / "perm_imp_by_component.png", dpi=160)
            plt.close()

            plt.figure()
            p_by_stat.plot(kind="bar")
            plt.title("Permutation Importances: mean vs std")
            plt.ylabel("Sum of perm importance (mean)")
            plt.tight_layout()
            plt.savefig(outdir / "perm_imp_by_stat.png", dpi=160)
            plt.close()

    # One combined CSV for convenience
    dfi.to_csv(outdir / "importance_detailed.csv", index=False)
    print("Saved grouped importance summaries and plots to", outdir)

if __name__ == "__main__":
    main()