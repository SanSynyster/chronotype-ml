#!/usr/bin/env python3
"""Causal GRU sequence model for trial-level risky choice (deep-learning idea #5).

Mirrors the leakage-safe setup of the existing baseline (scripts/risky_choice_baseline.py):
target `risky-choice`, grouped by participant, pre-choice + previous-trial features only
(no same-trial outcome/feedback). Reference baseline: balanced acc ~0.587, ROC AUC ~0.62.

Scientific point: the GRU is fed ONLY the current pre-choice context plus the *previous*
trial's outcome, and must learn temporal integration itself -- so we deliberately drop the
hand-engineered Rolling* history features. Predictions are unidirectional (causal): the
logit at trial t depends only on trials <= t, so no future leakage within a session.

Outputs (under reports/clean/risky_choice_seq/):
  - metrics.json : out-of-fold balanced accuracy / ROC AUC vs. baselines
  - participant_embeddings.csv : out-of-fold GRU embedding per participant (for the
    chronotype link -- each participant's embedding comes from a fold that never trained
    on them, so it is leakage-safe to regress chronotype/MEQ on it downstream).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DATA = "data/clean/risky_choice_prechoice.csv"
TARGET = "risky-choice"
GROUP = "participant_id"
ORDER = "global_trial_index"
OUTDIR = Path("reports/clean/risky_choice_seq")

# Current-trial pre-choice context (no choice/outcome of trial t).
CURRENT_FEATURES = [
    "Option1", "Option2", "OptionDiff", "AbsOptionDiff", "ValueSum", "ValueMax",
    "ValueMin", "OptionRatio", "AbsOptionRatio", "IsMixedSigns", "BothPositive",
    "BothNegative", "TrialProgress",
]
# Previous-trial outcome -- the recurrence signal the GRU integrates over time.
PREV_FEATURES = [
    "PrevRisky", "PrevFeedbackGain", "PrevFeedbackLoss", "PrevFeedbackError",
    "PrevGainCorrect", "PrevRTLog", "PrevScoreDelta",
]
FEATURES = CURRENT_FEATURES + PREV_FEATURES

SEED = 0
HIDDEN = 64
EPOCHS = 40
LR = 3e-3
N_SPLITS = 5


class GRUClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)  # unidirectional = causal
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)              # (B, T, H)
        return self.head(h).squeeze(-1), h  # logits (B, T), hidden states (B, T, H)


def build_sequences(df: pd.DataFrame):
    """Return padded tensors X (P, T, F), y (P, T), mask (P, T), and participant ids."""
    pids = sorted(df[GROUP].unique())
    seqs, ys, lengths = [], [], []
    for pid in pids:
        g = df[df[GROUP] == pid].sort_values(ORDER)
        seqs.append(g[FEATURES].to_numpy(np.float32))
        ys.append(g[TARGET].to_numpy(np.float32))
        lengths.append(len(g))
    T = max(lengths)
    P, F = len(pids), len(FEATURES)
    X = np.zeros((P, T, F), np.float32)
    y = np.zeros((P, T), np.float32)
    mask = np.zeros((P, T), np.float32)
    for i, (s, yy) in enumerate(zip(seqs, ys)):
        X[i, : len(s)] = s
        y[i, : len(yy)] = yy
        mask[i, : len(s)] = 1.0
    return X, y, mask, np.array(pids)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df = pd.read_csv(DATA)
    df = df[df[TARGET].notna()].copy()
    df[FEATURES] = df[FEATURES].fillna(0.0)

    return compute_oof(df, verbose=True)


def compute_oof(df: pd.DataFrame, verbose: bool = False):
    """Deterministic out-of-fold GRU pass.

    Returns a dict with the participant embeddings (P x HIDDEN), participant ids,
    pooled out-of-fold trial metrics, and the per-trial oof logits/labels/mask.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X, y, mask, pids = build_sequences(df)
    P, T, F = X.shape

    oof_logit = np.full((P, T), np.nan, np.float32)
    oof_emb = np.zeros((P, HIDDEN), np.float32)

    gkf = GroupKFold(n_splits=N_SPLITS)
    # GroupKFold needs per-sample groups; here each participant is one unit.
    for fold, (tr, te) in enumerate(gkf.split(np.arange(P), groups=pids)):
        # Standardize features on TRAIN participants' valid trials only.
        scaler = StandardScaler().fit(X[tr][mask[tr] > 0])
        Xs = ((X.reshape(-1, F) - scaler.mean_) / scaler.scale_).reshape(P, T, F)
        Xs = (Xs * mask[..., None]).astype(np.float32)  # keep pads at 0

        Xt = torch.tensor(Xs)
        yt = torch.tensor(y)
        mt = torch.tensor(mask)

        model = GRUClassifier(F)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        lossf = nn.BCEWithLogitsLoss(reduction="none")
        model.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            logits, _ = model(Xt[tr])
            loss = (lossf(logits, yt[tr]) * mt[tr]).sum() / mt[tr].sum()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits, hidden = model(Xt[te])
        oof_logit[te] = logits.numpy()
        # participant embedding = mean GRU hidden state over valid trials
        hsum = (hidden * mt[te][..., None]).sum(1)
        oof_emb[te] = (hsum / mt[te].sum(1, keepdim=True)).numpy()
        if verbose:
            print(f"fold {fold}: train P={len(tr)} test P={len(te)} final loss={loss.item():.4f}")

    # Pool out-of-fold trial predictions.
    valid = mask > 0
    y_true = y[valid].astype(int)
    p_hat = 1 / (1 + np.exp(-oof_logit[valid]))
    y_pred = (p_hat >= 0.5).astype(int)

    ba = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, p_hat)

    emb_df = pd.DataFrame(oof_emb, columns=[f"gru_emb_{i}" for i in range(HIDDEN)])
    emb_df.insert(0, GROUP, pids)

    metrics = {
        "model": "causal_GRU",
        "n_trials": int(valid.sum()),
        "n_participants": int(P),
        "features_used": FEATURES,
        "note": "Rolling* history features deliberately excluded; GRU learns history.",
        "out_of_fold": {"balanced_accuracy": round(float(ba), 4),
                         "roc_auc": round(float(auc), 4)},
        "reference_baseline": {"balanced_accuracy": 0.587, "roc_auc": 0.62},
    }

    if verbose:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        (OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
        emb_df.to_csv(OUTDIR / "participant_embeddings.csv", index=False)
        print(json.dumps(metrics["out_of_fold"], indent=2))
        print("baseline:", metrics["reference_baseline"])
        print("wrote", OUTDIR / "metrics.json", "and participant_embeddings.csv")

    return {"embeddings": emb_df, "metrics": metrics}


if __name__ == "__main__":
    main()
