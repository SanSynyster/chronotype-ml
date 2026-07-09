#!/usr/bin/env python3
"""Cross-subject EEGNet: single-trial feedback decoding on corrected data.

Uses the collaborator's cleaned epoched .set data (scripts/dl/load_clean_epochs.py),
n=39 subjects, 13.5k trials. Leakage-safe: GroupKFold over subjects, so every test
trial comes from a participant the model never trained on. Features are z-scored
per channel using TRAIN statistics only. Pooled out-of-fold metrics are reported.

This is the EEGNet entry-point task (validates the pipeline + that the FRN/P300 is
single-trial decodable). The chronotype/multimodal models build on this.

Run (env_dl):
    python scripts/dl/run_eegnet_feedback.py --epochs 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import load_clean_epochs as lce
from eegnet import EEGNet

OUTDIR = Path("reports/clean/discovery/eegnet_feedback")
SEED = 0


def chunked_stats(X, idx, chunk=512):
    """Per-channel mean/std over (trials, time) for X[idx], without copying X[idx]."""
    c = X.shape[1]
    n = 0
    s = np.zeros(c, np.float64)
    ss = np.zeros(c, np.float64)
    for i in range(0, len(idx), chunk):
        b = X[idx[i:i + chunk]].astype(np.float64)
        n += b.shape[0] * b.shape[2]
        s += b.sum(axis=(0, 2))
        ss += (b ** 2).sum(axis=(0, 2))
    mu = s / n
    sd = np.sqrt(np.maximum(ss / n - mu ** 2, 0)) + 1e-7
    return mu.astype("float32")[None, :, None], sd.astype("float32")[None, :, None]


def _batch(X, idx, mu, sd, device):
    xb = (X[idx].astype("float32") - mu) / sd     # small: (len(idx), C, T)
    return torch.from_numpy(xb).to(device)


def train_fold(X, tr, ytr, te, mu, sd, n_chan, n_time, n_classes, epochs, lr, batch, device):
    torch.manual_seed(SEED)
    model = EEGNet(n_chan, n_time, n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    ytr_t = torch.from_numpy(ytr).to(device)
    n = len(tr)
    model.train()
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            sel = perm[i:i + batch]
            xb = _batch(X, tr[sel], mu, sd, device)
            opt.zero_grad()
            loss = lossf(model(xb), ytr_t[sel])
            loss.backward()
            opt.step()
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(te), batch):
            xb = _batch(X, te[i:i + batch], mu, sd, device)
            probs.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(probs)


def score_task(y, prob):
    pred = prob.argmax(1)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }
    if prob.shape[1] == 2:
        out["roc_auc"] = float(roc_auc_score(y, prob[:, 1]))
    else:
        out["macro_ovr_auc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    return out


def bootstrap_ci(y, prob, groups, metric, n_boot=500):
    rng = np.random.default_rng(SEED)
    subjects = np.unique(groups)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == s) for s in sample])
        try:
            vals.append(score_task(y[idx], prob[idx])[metric])
        except ValueError:
            pass
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tasks", nargs="+", default=["valence", "correctness", "condition4"],
                    choices=["valence", "correctness", "condition4"])
    ap.add_argument("--permutations", type=int, default=0,
                    help="Full-pipeline label permutations. Use low values for fast checks.")
    ap.add_argument("--decim", type=int, default=1, help="Keep every nth time sample after loading.")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    np.random.seed(SEED)

    d = lce.load_all(labeled_only=False)
    X, groups = d["X"], d["subject"]
    if args.decim > 1:
        X = X[:, :, ::args.decim].copy()
    n_chan, n_time = X.shape[1], X.shape[2]

    results = {}
    for task in args.tasks:
        y = d[task].astype("int64")
        n_classes = int(y.max() + 1)
        oof = np.full((len(y), n_classes), np.nan, dtype="float32")
        gkf = GroupKFold(n_splits=args.folds)
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
            mu, sd = chunked_stats(X, tr)
            prob = train_fold(X, tr, y[tr], te, mu, sd, n_chan, n_time, n_classes,
                              args.epochs, args.lr, args.batch, device)
            oof[te] = prob
            fs = score_task(y[te], prob)
            print(f"{task} fold {fold}: test subj={len(np.unique(groups[te]))} "
                  f"BA={fs['balanced_accuracy']:.3f}")

        obs = score_task(y, oof)
        primary_metric = "roc_auc" if n_classes == 2 else "macro_ovr_auc"
        obs[f"{primary_metric}_ci_subject_bootstrap"] = bootstrap_ci(y, oof, groups, primary_metric)
        obs["permutation_p"] = None
        if args.permutations:
            rng = np.random.default_rng(SEED)
            null = []
            for p in range(args.permutations):
                y_perm = y.copy()
                # Permute labels within subject to preserve subject/trial counts while breaking EEG-label pairing.
                for s in np.unique(groups):
                    idx = np.flatnonzero(groups == s)
                    y_perm[idx] = rng.permutation(y_perm[idx])
                poof = np.full((len(y), n_classes), np.nan, dtype="float32")
                for tr, te in gkf.split(X, y_perm, groups):
                    mu, sd = chunked_stats(X, tr)
                    poof[te] = train_fold(X, tr, y_perm[tr], te, mu, sd, n_chan, n_time,
                                          n_classes, args.epochs, args.lr, args.batch, device)
                null.append(score_task(y_perm, poof)[primary_metric])
                print(f"{task} permutation {p + 1}/{args.permutations}: {null[-1]:.3f}")
            obs["permutation_p"] = float((1 + np.sum(np.array(null) >= obs[primary_metric])) / (args.permutations + 1))
            obs["permutation_null"] = [float(x) for x in null]
        results[task] = obs

    result = {
        "task": "single-trial feedback decoding, cross-subject, corrected participant_master IDs",
        "model": "EEGNet",
        "n_subjects": int(len(np.unique(groups))),
        "n_trials": int(len(groups)),
        "input_shape": [int(n_chan), int(n_time)],
        "cv": f"GroupKFold({args.folds}) over subjects",
        "epochs": args.epochs,
        "device": device,
        "decim": args.decim,
        "permutations": args.permutations,
        "out_of_fold": results,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["out_of_fold"], indent=2))
    print("wrote", OUTDIR / "summary.json")


if __name__ == "__main__":
    main()
