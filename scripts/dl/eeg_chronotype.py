#!/usr/bin/env python3
"""Chronotype from single-trial EEG -- the neural analogue of idea #5.

Mirrors the behavioral approach exactly: an EEGNet is trained ONLY on the
chronotype-agnostic auxiliary task (feedback valence: loss vs gain), cross-subject.
Its penultimate features, averaged out-of-fold per subject, become a leakage-safe
EEG embedding. Chronotype is then tested on those 39 embeddings with the shared
permutation-clean nested-LOO evaluator (scripts/dl/chrono_eval.py), so the EEG
result is directly comparable to the behavioral GRU one.

Run (env_dl):
    python scripts/dl/eeg_chronotype.py --epochs 30 --permutations 1000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))  # use most cores for CPU training
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import chrono_eval
import load_clean_epochs as lce
from eegnet import EEGNet
from run_eegnet_feedback import _batch, chunked_stats

OUTDIR = Path("reports/clean/eeg_chronotype")
SEED = 0


def train_and_extract(X, tr, ytr, te, mu, sd, n_chan, n_time, epochs, lr, batch, device):
    """Train EEGNet on valence; return test-trial (valence_prob, penultimate_features)."""
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
            opt.zero_grad()
            loss = lossf(model(_batch(X, tr[sel], mu, sd, device)), ytr_t[sel])
            loss.backward()
            opt.step()
    model.eval()
    probs, feats = [], []
    with torch.no_grad():
        for i in range(0, len(te), batch):
            logits, f = model(_batch(X, te[i:i + batch], mu, sd, device), return_feat=True)
            probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            feats.append(f.cpu().numpy())
    return np.concatenate(probs), np.concatenate(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(SEED)

    d = lce.load_all()
    X, yv, groups = d["X"], d["valence"], d["subject"]
    n_chan, n_time = X.shape[1], X.shape[2]

    oof_prob = np.full(len(yv), np.nan)
    oof_feat = None
    gkf = GroupKFold(n_splits=args.folds)
    for fold, (tr, te) in enumerate(gkf.split(X, yv, groups)):
        mu, sd = chunked_stats(X, tr)
        prob, feat = train_and_extract(X, tr, yv[tr], te, mu, sd, n_chan, n_time,
                                       args.epochs, args.lr, args.batch, device)
        if oof_feat is None:
            oof_feat = np.zeros((len(yv), feat.shape[1]), np.float32)
        oof_prob[te] = prob
        oof_feat[te] = feat
        print(f"fold {fold}: valence AUC={roc_auc_score(yv[te], prob):.3f}", flush=True)

    aux_auc = float(roc_auc_score(yv, oof_prob))  # auxiliary-task performance

    # Two per-subject EEG embeddings from the out-of-fold features:
    #   mean     = average features over all trials (loses the loss/gain contrast)
    #   contrast = mean(loss trials) - mean(gain trials), matching the prior ML
    #              finding that chronotype lives in the loss-MINUS-gain P300 contrast
    subs = np.unique(groups)
    loss = yv == 1
    emb_mean = np.vstack([oof_feat[groups == s].mean(0) for s in subs])
    emb_contrast = np.vstack([
        oof_feat[(groups == s) & loss].mean(0) - oof_feat[(groups == s) & ~loss].mean(0)
        for s in subs])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = {"auxiliary_valence_auc": round(aux_auc, 4), "embeddings": {}}
    for name, emb in [("mean", emb_mean), ("contrast", emb_contrast)]:
        df = pd.DataFrame(emb, columns=[f"eeg_emb_{i}" for i in range(emb.shape[1])])
        df.insert(0, "participant_id", subs)
        out = chrono_eval.evaluate(df, permutations=args.permutations, label=f"eeg_{name}")
        summary["embeddings"][name] = out["result"]
        df.to_csv(OUTDIR / f"subject_eeg_embeddings_{name}.csv", index=False)
        np.save(OUTDIR / f"null_auc_{name}.npy", out["null"])
        print(f"{name:9} AUC={out['result']['nested_loo_roc_auc']:.3f} "
              f"p={out['result']['perm_p_value']:.3f}", flush=True)
    # Keep the contrast embedding as the canonical EEG embedding for fusion.
    pd.read_csv(OUTDIR / "subject_eeg_embeddings_contrast.csv").to_csv(
        OUTDIR / "subject_eeg_embeddings.csv", index=False)

    (OUTDIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", OUTDIR / "metrics.json")


if __name__ == "__main__":
    main()
