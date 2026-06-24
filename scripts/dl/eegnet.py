"""EEGNet (Lawhern et al., 2018) in PyTorch -- compact CNN for single-trial EEG.

Input is (batch, channels, time); the channel axis is treated as the spatial
dimension of a depthwise convolution. Defaults are scaled for 250 Hz data.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_chan: int, n_time: int, n_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16, kern: int = 125,
                 dropout: float = 0.5):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(  # spatial filter across channels
            nn.Conv2d(F1, F1 * D, (n_chan, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        with torch.no_grad():
            x = torch.zeros(1, 1, n_chan, n_time)
            x = self.separable(self.depthwise(self.firstconv(x)))
            self.flat = x.numel()
        self.classify = nn.Linear(self.flat, n_classes)

    def features(self, x):
        """Penultimate flattened representation (before the classifier head)."""
        x = x.unsqueeze(1)                       # (B, 1, C, T)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return x.flatten(1)

    def forward(self, x, return_feat: bool = False):
        f = self.features(x)
        logits = self.classify(f)
        return (logits, f) if return_feat else logits
