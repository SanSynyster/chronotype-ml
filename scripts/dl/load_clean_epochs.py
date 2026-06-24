"""Load the cleaned, epoched EEGLAB .set files into deep-learning tensors.

These are the collaborator's preprocessed single-trial epochs
(data/raw/shifted_set/Epochs_Extracted_<id>_updated_trialinfo_shifted.set):
64 ch, 250 Hz, ~379 epochs/subject, window -1.1..+1.4 s, standard 10-20.
Already denoised + baseline-corrected, so no preprocessing here -- just crop,
label, and stack. Much better than the raw .cnt path (scripts/load_raw_eeg.py).

Condition is encoded in the ERPLAB bin labels `B#(code)`:
    50 = gain_correct, 60 = gain_error, 70 = loss_correct, 80 = loss_error
Multiple bins can map to the same code. Feedback *valence*: gain = {50,60},
loss = {70,80}.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")

SET_DIR = Path("data/raw/shifted_set")
CHRONO = "data/clean/chronotype_participant.csv"

CODE_TO_COND = {50: "gain_correct", 60: "gain_error", 70: "loss_correct", 80: "loss_error"}
GAIN_CODES, LOSS_CODES = {50, 60}, {70, 80}


def _subject_id(path: str) -> int:
    return int(re.search(r"_(\d{4})_", Path(path).name).group(1))


def _epoch_codes(ep: mne.Epochs) -> np.ndarray:
    """Per-epoch feedback code (50/60/70/80) parsed from the bin labels."""
    id_to_code = {}
    for label, eid in ep.event_id.items():
        m = re.search(r"\((\d+)\)", label)
        id_to_code[eid] = int(m.group(1)) if m else -1
    return np.array([id_to_code[e] for e in ep.events[:, 2]])


def load_subject(path: str, tmin: float = -0.2, tmax: float = 0.8, ref_chs=None):
    """Return (X, codes, ch_names, times) for one subject, cropped to [tmin, tmax].

    X is (trials, channels, times) float32 in microvolts. If ref_chs is given,
    channels are reordered to match it (so all subjects stack consistently).
    """
    ep = mne.read_epochs_eeglab(path)
    if ref_chs is not None and list(ep.ch_names) != list(ref_chs):
        ep = ep.copy().reorder_channels(list(ref_chs))
    ep = ep.crop(tmin, tmax)
    codes = _epoch_codes(ep)
    X = (ep.get_data() * 1e6).astype("float32")  # V -> uV
    return X, codes, ep.ch_names, ep.times


def load_all(tmin: float = -0.2, tmax: float = 0.8, labeled_only: bool = True):
    """Stack all subjects into one dataset.

    Returns dict with:
      X         (N, C, T) float32, all trials concatenated
      codes     (N,) feedback code per trial
      valence   (N,) 1=loss, 0=gain
      subject   (N,) subject id per trial (use as CV group)
      chronotype(N,) 1=evening, 0=morning (NaN if unlabeled)
      ch_names, times
    """
    paths = sorted(glob.glob(str(SET_DIR / "*.set")))
    lab = pd.read_csv(CHRONO).set_index("participant_id")["Chronotype"]
    lab = (lab.str.lower() == "evening").astype(int).to_dict()

    ref_chs = None
    Xs, codes, subj = [], [], []
    for p in paths:
        sid = _subject_id(p)
        if labeled_only and sid not in lab:
            continue
        Xi, ci, ch, times = load_subject(p, tmin, tmax, ref_chs)
        if ref_chs is None:
            ref_chs, ref_times = ch, times
        Xs.append(Xi)
        codes.append(ci)
        subj.append(np.full(len(ci), sid))

    X = np.concatenate(Xs).astype("float32")
    codes = np.concatenate(codes)
    subject = np.concatenate(subj)
    valence = np.where(np.isin(codes, list(LOSS_CODES)), 1, 0).astype("int64")
    chrono = np.array([lab.get(s, np.nan) for s in subject], dtype="float64")
    # keep only real feedback trials (codes in the 50/60/70/80 set)
    keep = np.isin(codes, list(CODE_TO_COND))
    return {
        "X": X[keep], "codes": codes[keep], "valence": valence[keep],
        "subject": subject[keep], "chronotype": chrono[keep],
        "ch_names": ref_chs, "times": ref_times,
    }


if __name__ == "__main__":
    d = load_all()
    n_subj = len(np.unique(d["subject"]))
    print(f"subjects: {n_subj} | trials: {d['X'].shape[0]} | tensor {d['X'].shape} (N,C,T)")
    print(f"valence balance: loss={int(d['valence'].sum())} gain={int((1-d['valence']).sum())}")
    print(f"time window: {d['times'][0]:.3f}..{d['times'][-1]:.3f}s ({len(d['times'])} samples)")
    print(f"chronotype-labeled trials: {int((~np.isnan(d['chronotype'])).sum())}")
