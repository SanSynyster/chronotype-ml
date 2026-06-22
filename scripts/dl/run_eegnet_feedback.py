#!/usr/bin/env python3
"""Cross-subject EEGNet baseline: single-trial feedback-valence (loss vs gain) decoding.

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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import load_clean_epochs as lce
from eegnet import EEGNet

OUTDIR = Path("reports/clean/eegnet_feedback")
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


def train_fold(X, tr, ytr, te, mu, sd, n_chan, n_time, epochs, lr, batch, device):
    torch.manual_seed(SEED)
    model = EEGNet(n_chan, n_time).to(device)
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
            probs.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(SEED)

    d = lce.load_all()
    X, y, groups = d["X"], d["valence"], d["subject"]
    n_chan, n_time = X.shape[1], X.shape[2]

    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=args.folds)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        mu, sd = chunked_stats(X, tr)
        prob = train_fold(X, tr, y[tr], te, mu, sd, n_chan, n_time,
                          args.epochs, args.lr, args.batch, device)
        oof[te] = prob
        ba = balanced_accuracy_score(y[te], (prob >= 0.5).astype(int))
        auc = roc_auc_score(y[te], prob)
        print(f"fold {fold}: test subj={len(np.unique(groups[te]))} "
              f"BA={ba:.3f} AUC={auc:.3f}")

    ba = balanced_accuracy_score(y, (oof >= 0.5).astype(int))
    auc = roc_auc_score(y, oof)
    result = {
        "task": "single-trial feedback valence (loss vs gain), cross-subject",
        "model": "EEGNet",
        "n_subjects": int(len(np.unique(groups))),
        "n_trials": int(len(y)),
        "input_shape": [int(n_chan), int(n_time)],
        "cv": f"GroupKFold({args.folds}) over subjects",
        "epochs": args.epochs,
        "out_of_fold": {"balanced_accuracy": round(float(ba), 4),
                         "roc_auc": round(float(auc), 4)},
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["out_of_fold"], indent=2))
    print("wrote", OUTDIR / "metrics.json")


if __name__ == "__main__":
    main()
