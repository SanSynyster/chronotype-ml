"""Load the study's raw ANT Neuro .cnt EEG into MNE / deep-learning tensors.

The raw files are NOT Neuroscan CNT. They are ANT Neuro (ASA/eego) RIFF-based
.cnt files (chunks: info / eeph / raw3chan / evt). MNE's read_raw_cnt CANNOT
read them (you get an OverflowError / UnicodeDecodeError). read_raw_ant exists
only in MNE >= 1.9; this repo's env is MNE 1.7, so we use the low-level `antio`
reader directly and build the MNE object ourselves.

    pip install antio

64 ch, 250 Hz, standard 10-20 labels. Feedback trigger codes:
    50 gain_correct, 60 gain_error, 70 loss_correct, 80 loss_error
"""
from __future__ import annotations
import numpy as np
import mne
from antio import read_cnt

FEEDBACK_IDS = {"gain_correct": 50, "gain_error": 60, "loss_correct": 70, "loss_error": 80}


def load_raw_ant(path: str) -> mne.io.RawArray:
    """Read an ANT Neuro .cnt into an MNE RawArray (volts), with montage."""
    c = read_cnt(str(path))
    n = c.get_sample_count()
    nch = c.get_channel_count()
    sf = int(round(c.get_sample_frequency()))
    labels = [c.get_channel(i, encoding="latin1")[0] for i in range(nch)]
    arr = c.get_samples_as_nparray(0, n)        # (chan, samples), microvolts
    if arr.shape[0] == n:
        arr = arr.T
    info = mne.create_info(labels, sf, ch_types="eeg")
    raw = mne.io.RawArray(arr * 1e-6, info, verbose="ERROR")  # uV -> V
    raw.set_montage("standard_1020", on_missing="ignore")
    return raw


def read_events(path: str) -> np.ndarray:
    """Return an MNE events array (sample, 0, code) from the .cnt triggers."""
    c = read_cnt(str(path))
    out = []
    for i in range(c.get_trigger_count()):
        t = c.get_trigger(i)
        try:
            code = int(t[0])
        except (ValueError, TypeError):
            continue
        if code > 0:
            out.append([int(t[1]), 0, code])
    return np.array(out, dtype=int)


def make_feedback_epochs(path: str, tmin=-0.2, tmax=0.8, l_freq=0.1, h_freq=30.0,
                         baseline=(-0.2, 0.0)) -> mne.Epochs:
    """Filtered, baseline-corrected feedback-locked epochs ready for DL."""
    raw = load_raw_ant(path)
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose="ERROR")
    events = read_events(path)
    present = {k: v for k, v in FEEDBACK_IDS.items() if v in set(events[:, 2])}
    return mne.Epochs(raw, events, event_id=present, tmin=tmin, tmax=tmax,
                      baseline=baseline, preload=True, on_missing="ignore",
                      verbose="ERROR")


def to_tensor(epochs: mne.Epochs):
    """Return (X, y): X=(trials, channels, time) float32, y=int condition codes."""
    X = epochs.get_data().astype("float32")
    y = epochs.events[:, 2]
    return X, y


if __name__ == "__main__":
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 else (
        "/Users/sahabtaali/Documents/Mahsima's Data/cnt for all subjects/Mahsima_1001.cnt")
    ep = make_feedback_epochs(f)
    X, y = to_tensor(ep)
    print("epochs:", X.shape, "(trials, channels, time)")
    print("conditions:", {k: int((y == v).sum()) for k, v in FEEDBACK_IDS.items()})
