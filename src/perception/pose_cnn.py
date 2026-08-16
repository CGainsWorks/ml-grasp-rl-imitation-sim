"""A small convolutional pose estimator, and what it is for.

This is not here to be a good pose estimator. It is here so that the sensing
noise this repository randomises over can be *checked* against an error an
actual estimator makes, instead of being asserted from plausibility --
`docs/randomisation-sources.md` exists because the same problem applied to the
friction and latency ranges.

Deliberately small: four strided convolutions and a linear head, about 200k
parameters, trained on 64x64 frames in a few minutes on a CPU. A larger network
would estimate pose better and answer the question less well, because the
interesting quantity is the *structure* of the error -- is it correlated in
time, does it worsen under occlusion -- and that structure is a property of the
task and the viewpoint rather than of the architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PoseCNN(nn.Module):
    """64x64 RGB -> object (x, y, z) in world coordinates."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 4, stride=2, padding=1), nn.ReLU(),      # 32
            nn.Conv2d(width, width * 2, 4, stride=2, padding=1), nn.ReLU(),   # 16
            nn.Conv2d(width * 2, width * 2, 4, stride=2, padding=1), nn.ReLU(),  # 8
            nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1), nn.ReLU(),  # 4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )
        # The object lives in a small box; predicting an offset from its centre
        # rather than an absolute position keeps the head's outputs order-one
        # and stops the network spending capacity on a constant.
        self.register_buffer("centre", torch.tensor([0.0, 0.0, 0.45]))
        self.register_buffer("scale", torch.tensor([0.2, 0.2, 0.2]))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``image`` is (N, 3, 64, 64) float in [0, 1]."""
        return self.centre + self.scale * self.head(self.features(image))
