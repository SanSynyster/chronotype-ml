#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_ml_subsets.py

Builds ML-ready train/test CSV (and Parquet if available) subsets from
`data/processed/ml_ready_features.csv` for multiple targets and feature packs.

Key features:
- Robust looping over ALL targets × packs (no early exit)
- Participant-wise split (default) or random split
- Safe categorical encoding + optional scaling for numeric cols
- Writes a meta JSON per combo with feature names and row counts
- Loud, helpful logging and an end-of-run summary
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Small util helpers
# ----------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(obj, out_path: str) -> None:
    ensure_dir(Path(out_path).parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_frame(df: pd.DataFrame, base_path_no_ext: str) -> None:
    csv_path = f"{base_path_no_ext}.csv"
    ensure_dir(Path(csv_path).parent)
    df.to_csv(csv_path, index=False)
    # Parquet is optional
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(f"{base_path_no_ext}.parquet", index=False)
    except Exception as e:
        log(f"ℹ️ Parquet not written for {base_path_no_ext}: {e}")


# ----------------------------
# Encoding / scaling
# ----------------------------

def encode_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    if not cat_cols:
        return df.copy()
    return pd.get_dummies(df, columns=cat_cols, dummy_na=False)


def scale_numeric(train_df: pd.DataFrame, test_df: pd.DataFrame, num_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if not num_cols:
        return train_df, test_df, {"scale": {}}
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])
    params = {"scale": {c: {"mean": float(m), "std": float(s)} for c, m, s in zip(num_cols, scaler.mean_, scaler.scale_)}}
    return train_df, test_df, params


# ----------------------------
# Splitting strategies
# ----------------------------

def do_split(df: pd.DataFrame, *, test_size: float, seed: int, mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows either randomly or by participant (grouped)."""
    if mode == "participant":
        # Group-aware split: hold out participants, not rows
        pids = df["participant_id"].dropna().astype(int).unique()
        p_train, p_test = train_test_split(sorted(pids), test_size=test_size, random_state=seed)
        is_train = df["participant_id"].astype(int).isin(p_train)
        return df.loc[is_train].reset_index(drop=True), df.loc[~is_train].reset_index(drop=True)
    elif mode == "random":
        return train_test_split(df, test_size=test_size, random_state=seed)
    else:
        raise ValueError(f"Unknown split mode: {mode}")


# ----------------------------
# Core builder per combo
# ----------------------------

def build_for_target_pack(
    base_df: pd.DataFrame,
    target: str,
    pack_name: str,
    pack_cols: List[str],
    split_mode: str,
    test_size: float,
    seed: int,
    scale: bool,
    outdir: Path,
) -> dict:
    id_cols = ["participant_id", "Block", "Trial", "global_trial_index"]

    # Ensure target exists
    if target not in base_df.columns:
        raise ValueError(f"Target '{target}' not found in features file.")

    # Avoid duplicates if user mistakenly included target in pack
    pack_cols = [c for c in pack_cols if c != target]

    # Feature/label selection
    feat_keep = id_cols + pack_cols
    missing = [c for c in feat_keep if c not in base_df.columns]
    if missing:
        raise ValueError(f"Pack '{pack_name}' missing columns in data: {missing[:8]}{' …' if len(missing)>8 else ''}")

    X = base_df[feat_keep].copy()
    y = base_df[target].copy()

    # Drop rows with missing label
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # Temporary concat for split alignment
    tmp = X.copy()
    tmp["__target__"] = y

    train_df, test_df = do_split(tmp, test_size=test_size, seed=seed, mode=split_mode)

    y_train = train_df.pop("__target__").copy()
    y_test = test_df.pop("__target__").copy()

    # Determine feature columns (exclude IDs)
    feat_cols = [c for c in train_df.columns if c not in id_cols]
    num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in feat_cols if c not in num_cols]

    # Encode categoricals
    enc_train = encode_categoricals(train_df[feat_cols], cat_cols)
    enc_test = pd.get_dummies(test_df[feat_cols], columns=cat_cols, dummy_na=False)
    enc_test = enc_test.reindex(columns=enc_train.columns, fill_value=0)

    # Scale numeric columns
    orig_num_cols = [c for c in num_cols if c in enc_train.columns]
    if scale and orig_num_cols:
        enc_train, enc_test, scaler_params = scale_numeric(enc_train, enc_test, orig_num_cols)
    else:
        scaler_params = {"scale": {}}

    # Reattach IDs + labels
    train_out = pd.concat([
        train_df[["participant_id", "Block", "Trial", "global_trial_index"]].reset_index(drop=True),
        enc_train.reset_index(drop=True),
        y_train.reset_index(drop=True).rename(target),
    ], axis=1)

    test_out = pd.concat([
        test_df[["participant_id", "Block", "Trial", "global_trial_index"]].reset_index(drop=True),
        enc_test.reset_index(drop=True),
        y_test.reset_index(drop=True).rename(target),
    ], axis=1)

    base = outdir / f"{target}__{pack_name}__{split_mode}"
    ensure_dir(outdir)
    write_frame(train_out, str(base) + "__train")
    write_frame(test_out, str(base) + "__test")

    # Meta
    meta = {
        "target": target,
        "pack": pack_name,
        "split_mode": split_mode,
        "rows": {"train": int(train_out.shape[0]), "test": int(test_out.shape[0])},
        "features": {
            "n_features": int(len([c for c in train_out.columns if c not in (["participant_id", "Block", "Trial", "global_trial_index", target])])),
            "id_columns": ["participant_id", "Block", "Trial", "global_trial_index"],
            "encoded_feature_columns": list(enc_train.columns),
            "scaled_numeric_cols": list(scaler_params.get("scale", {}).keys()),
        },
        "label_distribution": {
            "train": train_out[target].value_counts(dropna=False).to_dict() if not pd.api.types.is_numeric_dtype(y_train) else "numeric",
            "test": test_out[target].value_counts(dropna=False).to_dict() if not pd.api.types.is_numeric_dtype(y_test) else "numeric",
        },
    }
    write_json(meta, str(base) + "__meta.json")
    log(f"✅ Built [{target}] × [{pack_name}] with {split_mode} split.")
    return meta


# ----------------------------
# Pack resolution
# ----------------------------

def default_packs_from_columns(cols: List[str]) -> Dict[str, List[str]]:
    """Construct default packs that match your schema counts.

    eeg_only = 11 columns with pattern like '{channel}_{window}'
    beh_only = 12 trial-level behaviour cols (incl. quality/valence/trigger)
    demo_only = 3 demographic cols
    combined = union of eeg_only+beh_only+demo_only
    """
    # EEG columns
    eeg_cols = [
        "CPz_P300", "Cz_FRN", "Cz_P300", "FC1_FRN", "FC2_FRN",
        "FCz_FRN", "FCz_P300", "Fz_FRN", "Fz_P300", "POz_P300", "Pz_P300",
    ]
    eeg_cols = [c for c in eeg_cols if c in cols]

    # Behaviour-only columns
    beh_cols = [
        "good_trial", "trigger_numeric", "trigger_val", "behav_valence",
        "Option1", "Option2", "ActualValue1", "ActualValue2",
        "ChoiceMade", "CorrectChoice", "ResponseTime", "CurrentScore",
    ]
    beh_cols = [c for c in beh_cols if c in cols]

    demo_cols = ["Age", "Gender", "Chronotype"]
    demo_cols = [c for c in demo_cols if c in cols]

    packs = {
        "eeg_only": eeg_cols,
        "beh_only": beh_cols,
        "demo_only": demo_cols,
        "combined": sorted(set(eeg_cols) | set(beh_cols) | set(demo_cols)),
    }
    return packs


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ML subsets from ml_ready_features.csv")
    p.add_argument("--infile", default="data/processed/ml_ready_features.csv", help="Input ML-ready features CSV")
    p.add_argument("--outdir", default="data/processed/ml_subsets", help="Output directory")
    p.add_argument("--targets", default="risky-choice,behav_valence,Chronotype,ResponseTime,CurrentScore", help="Comma-separated target columns")
    p.add_argument("--packs", default="eeg_only,beh_only,demo_only,combined", help="Comma-separated pack names to build")
    p.add_argument("--split", default="participant", choices=["participant", "random"], help="Split mode")
    p.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--scale", action="store_true", help="Scale numeric features")
    p.add_argument("--verbose", action="store_true", help="Verbose per-combo logging")
    p.add_argument("--list-combos", action="store_true", help="Only print potential combos and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    infile = Path(args.infile)
    outdir = Path(args.outdir)

    log(f"📥 Loading {infile}")
    df = pd.read_csv(infile)

    # Resolve packs
    packs = default_packs_from_columns(df.columns.tolist())

    # Persist a high-level schema snapshot
    schema = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "packs": {k: len(v) for k, v in packs.items()},
        "targets": args.targets.split(","),
        "split": args.split,
        "test_size": args.test_size,
        "seed": args.seed,
        "scale": bool(args.scale),
        "participants": int(df["participant_id"].nunique()) if "participant_id" in df.columns else None,
    }
    ensure_dir(outdir)
    write_json(schema, str(outdir / "schema.json"))

    # Optionally just list combos and exit
    if args.list_combos:
        log("Possible targets × packs:")
        for t in args.targets.split(","):
            for p in args.packs.split(","):
                log(f"  - {t} × {p}")
        return

    # Build all requested combos
    made: List[Tuple[str, str, dict]] = []
    failed: List[Tuple[str, str, str]] = []

    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        for pack_name in [p.strip() for p in args.packs.split(",") if p.strip()]:
            pack_cols = packs.get(pack_name)
            if pack_cols is None:
                failed.append((target, pack_name, "Unknown pack name"))
                log(f"❌ FAILED {target} × {pack_name}: unknown pack")
                continue
            try:
                meta = build_for_target_pack(
                    base_df=df,
                    target=target,
                    pack_name=pack_name,
                    pack_cols=pack_cols,
                    split_mode=args.split,
                    test_size=args.test_size,
                    seed=args.seed,
                    scale=args.scale,
                    outdir=outdir,
                )
                made.append((target, pack_name, meta["rows"]))
                if args.verbose:
                    log(f"✅ wrote {target} × {pack_name} rows={meta['rows']}")
            except Exception as e:
                failed.append((target, pack_name, str(e)))
                log(f"❌ FAILED {target} × {pack_name}: {e}")

    # Summary
    log(f"\nSummary: built {len(made)} combos; failed {len(failed)}.")
    for t, p, rows in made:
        log(f"  ✓ {t:<15} × {p:<10} → train={rows['train']}, test={rows['test']}")
    if failed:
        log("Failures:")
        for t, p, err in failed:
            log(f"  ✗ {t:<15} × {p:<10} → {err}")


if __name__ == "__main__":
    main()