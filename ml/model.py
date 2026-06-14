#!/usr/bin/env python3
"""MLP architecture for VLSI cell-delay prediction."""
from __future__ import annotations

import torch
import torch.nn as nn


class TimingPredictor(nn.Module):
    """
    Multi-layer perceptron for VLSI cell delay prediction.

    Architecture rationale
    -----------------------
    * Input layer sized to the feature count (11 features).
    * 4 hidden layers with decreasing width; a residual connection is applied
      across the two equal-width (256) layers for stable gradient flow.
    * Output: 3 values (tpd, tpHL, tpLH) -- multi-output regression.
    * BatchNorm after each wide linear for training stability.
    * GELU activation (smoother than ReLU for physics-based regression).
    * Dropout=0.1 for mild regularization (the data is dense, not sparse).
    """

    def __init__(self, input_dim: int = 11, output_dim: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # equal-width block wrapped by a residual connection
        self.block = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = h + self.block(h)        # residual across the 256-wide layers
        return self.head(h)


if __name__ == "__main__":
    m = TimingPredictor()
    n = sum(p.numel() for p in m.parameters())
    print(m)
    print(f"parameters: {n:,}")
    print("output shape:", m(torch.randn(8, 11)).shape)
