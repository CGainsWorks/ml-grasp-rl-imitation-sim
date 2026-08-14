"""Networks and the observation normaliser shared by SAC and behaviour cloning.

Deliberately small. The observation is 32 floats of already-structured state,
not pixels, so two hidden layers of 256 units is not the bottleneck; the
bottleneck is exploration. Keeping SAC and BC on the same trunk is what lets a
behaviour-cloned actor be loaded straight into SAC as a starting point.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class RunningNorm(nn.Module):
    """Welford running mean/variance over observations, frozen at evaluation.

    Kept inside the module (rather than in the training loop) so that a saved
    checkpoint carries its own normalisation. A policy that is loaded with the
    wrong statistics fails in a way that looks like a bad policy, and that is a
    horrible bug to chase.
    """

    def __init__(self, dim: int, clip: float = 10.0) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(normed, -self.clip, self.clip)


def mlp(sizes: Sequence[int], activation=nn.ReLU, output_activation=nn.Identity) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed diagonal Gaussian policy, as used by SAC.

    ``deterministic=True`` returns the mode, which is what evaluation and
    demonstration replay use. Sampling during evaluation would add noise that
    has nothing to do with the policy's quality.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.trunk = mlp([obs_dim, *hidden], activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu_head = nn.Linear(hidden[-1], act_dim)
        self.log_std_head = nn.Linear(hidden[-1], act_dim)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False, with_logprob: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.norm(obs))
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            pre_tanh = mu
        else:
            pre_tanh = mu + std * torch.randn_like(mu)

        action = torch.tanh(pre_tanh)

        if not with_logprob:
            return action, torch.zeros(action.shape[0], device=action.device)

        # Log-probability with the tanh change of variables, in the numerically
        # stable form: log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u)).
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(pre_tanh).sum(dim=-1)
        logp -= (2.0 * (np.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(dim=-1)
        return action, logp

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.forward(tensor, deterministic=deterministic, with_logprob=False)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def act_with_noise(self, obs: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """Act using a supplied noise vector instead of a fresh Gaussian draw.

        Same reparameterisation SAC uses -- ``tanh(mu + sigma * eps)`` -- with
        ``eps`` provided by the caller, so exploration noise can be correlated
        in time rather than independent per step. With ``eps`` drawn from a
        standard normal this is exactly ``act(deterministic=False)``.
        """
        tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        hidden = self.trunk(self.norm(tensor))
        mu = self.mu_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), LOG_STD_MIN, LOG_STD_MAX)
        noise = torch.as_tensor(
            eps, dtype=torch.float32, device=self.device).unsqueeze(0)
        return torch.tanh(
            mu + torch.exp(log_std) * noise).squeeze(0).cpu().numpy()


class TwinQ(nn.Module):
    """Two independent Q networks; SAC takes the minimum to fight overestimation."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([self.norm(obs), act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)
