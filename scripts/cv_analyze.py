#!/usr/bin/env python3
import argparse, math
import pandas as pd
from scipy import stats

def ci95(s, n):  # normal approx
    return 1.96 * (s / math.sqrt(n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="reports/participant_chronotype/metrics_per_fold.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    # Expect columns: fold, model, acc, bal_acc, f1_macro, precision_macro, recall_macro
    # If you also store repeat index or split id, keep them – pairing works as long as rows align.

    print("=== Overall by model (mean ± 95% CI) ===")
    rows = []
    for m, g in df.groupby("model"):
        n = len(g)
        for col in ["acc","bal_acc","f1_macro","precision_macro","recall_macro"]:
            mean = g[col].mean()
            std  = g[col].std(ddof=1)
            ci   = ci95(std, n)
            rows.append((m, col, mean, std, n, ci))
    out = pd.DataFrame(rows, columns=["model","metric","mean","std","n","ci95"])
    print(out.pivot(index="model", columns="metric", values="mean").round(4))
    print("\n95% CI:")
    print(out.pivot(index="model", columns="metric", values="ci95").round(4))

    # Paired comparison HGB vs RF on bal_acc (rows aligned by order)
    try:
        df_h = df[df.model=="hgb"]["bal_acc"].reset_index(drop=True)
        df_r = df[df.model=="rf"]["bal_acc"].reset_index(drop=True)
        n = min(len(df_h), len(df_r))
        stat, p = stats.wilcoxon(df_h[:n], df_r[:n], zero_method="wilcox", alternative="greater")
        print(f"\nWilcoxon paired (HGB > RF) on bal_acc: W={stat}, p={p:.4f}, n={n}")
        print(f"HGB mean={df_h[:n].mean():.3f}, RF mean={df_r[:n].mean():.3f}")
    except Exception as e:
        print(f"\n[warn] Could not run paired test: {e}")

if __name__ == "__main__":
    main()