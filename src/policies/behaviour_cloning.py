"""Behaviour cloning: fitting an actor to demonstrated actions.

Kept apart from ``src/train_il.py`` because that module imports the MuJoCo
environment at import time, and this loop is pure torch -- no simulator at all.
The Isaac training script needs exactly this and nothing else, and it runs in an
interpreter where ``mujoco`` is not installed.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .networks import SquashedGaussianActor


def fit(
    actor: SquashedGaussianActor,
    obs: np.ndarray,
    act: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    rng: np.random.Generator,
    val_fraction: float = 0.1,
) -> Tuple[List[float], List[float]]:
    """Train the actor's deterministic mode to match the expert's actions."""
    n = len(obs)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Follow the actor rather than assuming the CPU: the Isaac trainer can put
    # it on the same card as the simulator.
    device = next(actor.parameters()).device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act, dtype=torch.float32, device=device)
    actor.norm.update(obs_t[train_idx])

    opt = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=weight_decay)
    train_curve: List[float] = []
    val_curve: List[float] = []

    for _ in range(epochs):
        actor.train()
        order = rng.permutation(len(train_idx))
        total = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            idx = train_idx[order[start : start + batch_size]]
            pred, _ = actor(obs_t[idx], deterministic=True, with_logprob=False)
            loss = F.mse_loss(pred, act_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
        train_curve.append(total / max(1, batches))

        actor.eval()
        with torch.no_grad():
            pred, _ = actor(obs_t[val_idx], deterministic=True, with_logprob=False)
            val_curve.append(float(F.mse_loss(pred, act_t[val_idx])))

    return train_curve, val_curve
