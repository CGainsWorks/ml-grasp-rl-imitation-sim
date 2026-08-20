"""Soft actor-critic, written out rather than imported.

Why not Stable Baselines3
-------------------------
The brief allows either. A custom implementation was chosen for one reason:
the imitation-plus-RL variant needs to reach inside the algorithm, seed the
replay buffer with demonstrations, initialise the actor from a cloned policy
and anneal a behaviour-cloning term against the policy loss. Doing that through
a framework's callback surface is more code, and more obscure code, than the
250 lines below.

SAC rather than PPO because the sample budget is small. Every run in this
repository is a few hundred thousand environment steps on a CPU, and an
off-policy learner with a replay buffer gets far more out of that than an
on-policy one.

The implementation is the standard one: twin critics, a target critic pair
updated by Polyak averaging, a tanh-squashed Gaussian actor, and an entropy
coefficient tuned automatically against a target entropy of ``-act_dim``.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .networks import SquashedGaussianActor, TwinQ


@dataclasses.dataclass
class SACConfig:
    """Hyperparameters. The defaults produced every number in the README."""

    gamma: float = 0.98
    tau: float = 0.005                # Polyak coefficient for the target critics
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 400_000
    start_steps: int = 5_000          # uniform-random actions before learning starts
    update_every: int = 50            # env steps between update bursts
    # Gradient steps per environment step. 0.5 rather than the SAC-usual 1.0,
    # measured rather than guessed: on the nominal world 0.5 scores 1.000 across
    # three seeds against 1.0's 0.989 and runs 1.94x faster, and on `medium`,
    # where nothing saturates, the two are indistinguishable (0.522 against
    # 0.467, Welch t = 0.18). Halving the updates is faster *and* no worse,
    # which is what over-updating a critic on a small replay buffer looks like.
    # experiments/compute_ablation.py is the evidence; docs/limitations.md
    # records what was rejected, including a 2.39x setting that scored 1.000 on
    # the saturated benchmark and 0.000 on every randomised seed.
    updates_per_step: float = 0.5
    target_entropy_scale: float = 1.0
    init_alpha: float = 0.1
    alpha_floor: float = 0.0          # lower bound on the entropy coefficient
    critic_warmup_updates: int = 0    # critic-only updates before the actor moves
    hidden: Tuple[int, int] = (256, 256)
    # Imitation options, unused by the plain baseline
    bc_coef: float = 0.0              # weight on the demonstration BC term
    bc_q_scale: float = 2.5           # TD3+BC style normalisation of the Q term
    # Apply the BC term only where the critic prefers the expert action.
    # Off by default; the reasoning is in ``update`` below.
    bc_q_filter: bool = False
    bc_decay_steps: int = 0           # linear decay of bc_coef to zero over N steps
    anchor_coef: float = 0.0          # weight on a frozen-policy anchor
    demo_sample_fraction: float = 0.0  # share of each batch drawn from demos

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)


class ReplayBuffer:
    """Flat float32 ring buffer.

    Demonstration transitions can be written into a reserved prefix and pinned
    there, so that the ring never overwrites them. That is what makes the
    imitation-plus-RL run keep its demonstrations for the whole training rather
    than losing them after the first few thousand steps.
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int,
                 device: str = "cpu") -> None:
        self.capacity = int(capacity)
        # Storage stays in numpy on the host: the buffer is far larger than a
        # batch, and holding it on an 8 GB card would cost more than the copy
        # saves. Only the sampled batch is moved.
        self.device = torch.device(device)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.pinned = 0

    def add(self, obs, act, rew, next_obs, done) -> None:
        idx = self.pinned + (self.ptr % max(1, self.capacity - self.pinned))
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew
        self.next_obs[idx] = next_obs
        self.done[idx] = done
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs, act, rew, next_obs, done) -> None:
        """Add ``n`` transitions at once.

        Identical in effect to calling :meth:`add` ``n`` times -- there is a
        test that asserts exactly that -- but without the Python loop, which is
        what the vectorised Isaac environment was spending its per-step budget
        on with 32 environments.
        """
        n = len(obs)
        span = max(1, self.capacity - self.pinned)
        idx = self.pinned + (np.arange(self.ptr, self.ptr + n) % span)
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew
        self.next_obs[idx] = next_obs
        self.done[idx] = done
        self.ptr += n
        self.size = min(self.size + n, self.capacity)

    def add_demonstrations(self, obs, act, rew, next_obs, done) -> int:
        """Write ``n`` demonstration transitions into the pinned prefix."""
        n = len(obs)
        if n > self.capacity // 2:
            raise ValueError("demonstrations would fill more than half the buffer")
        self.obs[:n] = obs
        self.act[:n] = act
        self.rew[:n] = rew
        self.next_obs[:n] = next_obs
        self.done[:n] = done
        self.pinned = n
        self.size = max(self.size, n)
        return n

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=batch_size)
        return self._gather(idx)

    def sample_mixed(
        self, batch_size: int, demo_fraction: float, rng: np.random.Generator
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Sample a batch with a fixed share drawn from the pinned demonstrations."""
        if self.pinned == 0 or demo_fraction <= 0.0:
            return self._gather(rng.integers(0, self.size, size=batch_size)), 0
        n_demo = int(round(batch_size * demo_fraction))
        n_demo = min(n_demo, self.pinned)
        demo_idx = rng.integers(0, self.pinned, size=n_demo)
        if self.size > self.pinned:
            agent_idx = rng.integers(self.pinned, self.size, size=batch_size - n_demo)
        else:
            agent_idx = rng.integers(0, self.pinned, size=batch_size - n_demo)
        # Demonstrations first, so the BC term can slice them off the front.
        return self._gather(np.concatenate([demo_idx, agent_idx])), n_demo

    def _gather(self, idx: np.ndarray) -> Dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(array[idx]).to(self.device, non_blocking=True)
            for key, array in (
                ("obs", self.obs), ("act", self.act), ("rew", self.rew),
                ("next_obs", self.next_obs), ("done", self.done),
            )
        }


class SAC:
    """Soft actor-critic agent."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: Optional[SACConfig] = None,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg or SACConfig()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.actor = SquashedGaussianActor(obs_dim, act_dim, self.cfg.hidden).to(self.device)
        self.critic = TwinQ(obs_dim, act_dim, self.cfg.hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        # Set by `freeze_anchor()` when fine-tuning from a checkpoint.
        self._anchor = None

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)

        self.log_alpha = torch.tensor(
            float(np.log(self.cfg.init_alpha)), requires_grad=True, device=self.device
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)
        self.target_entropy = -float(act_dim) * self.cfg.target_entropy_scale

        self.buffer = ReplayBuffer(obs_dim, act_dim, self.cfg.buffer_size,
                                   device=device)
        self._updates = 0

    # ------------------------------------------------------------------
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.actor.act(obs, deterministic=deterministic)

    def observe_normalisation(self, obs_batch: np.ndarray) -> None:
        """Feed observations to both running normalisers."""
        tensor = torch.as_tensor(
            np.asarray(obs_batch, dtype=np.float32)).to(self.device)
        self.actor.norm.update(tensor)
        self.critic.norm.update(tensor)
        self.critic_target.norm.mean.copy_(self.critic.norm.mean)
        self.critic_target.norm.var.copy_(self.critic.norm.var)

    # ------------------------------------------------------------------
    def freeze_anchor(self) -> None:
        """Snapshot the current actor as the reference for the anchor term.

        Called after loading a checkpoint and before training starts, so the
        anchor is what the policy knew on arrival rather than whatever it
        drifts into.
        """
        self._anchor = copy.deepcopy(self.actor)
        for p in self._anchor.parameters():
            p.requires_grad_(False)
        self._anchor.eval()

    def update(self, step: int) -> Dict[str, float]:
        cfg = self.cfg
        batch, n_demo = self.buffer.sample_mixed(
            cfg.batch_size, cfg.demo_sample_fraction, self.rng
        )
        obs, act, rew = batch["obs"], batch["act"], batch["rew"]
        next_obs, done = batch["next_obs"], batch["done"]

        # ---- critics
        with torch.no_grad():
            next_act, next_logp = self.actor(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_act)
            target_q = torch.min(q1_t, q2_t) - self.alpha.detach() * next_logp
            backup = rew + cfg.gamma * (1.0 - done) * target_q

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ---- critic warm-up
        # Starting from a cloned actor and a *random* critic destroys the clone
        # in a few hundred updates: the actor loss is -Q, and at that point Q is
        # noise, so the policy is dragged towards whatever the untrained critic
        # happens to prefer. The Q-filter on the behaviour-cloning term does not
        # help either, because it is also asking the random critic. Holding the
        # actor still while the critic fits the demonstrations costs a few
        # seconds and fixes it.
        if self._updates < cfg.critic_warmup_updates:
            with torch.no_grad():
                for p, p_targ in zip(self.critic.parameters(),
                                     self.critic_target.parameters()):
                    p_targ.mul_(1.0 - cfg.tau).add_(cfg.tau * p)
            self._updates += 1
            return {
                "critic_loss": float(critic_loss.detach()),
                "actor_loss": float("nan"),
                "bc_loss": float("nan"),
                "bc_coef": float(self._bc_coef(step)),
                "alpha": float(self.alpha.detach()),
                "entropy": float("nan"),
                "q_mean": float(q1.mean().detach()),
                "warmup": 1.0,
            }

        # ---- actor
        for p in self.critic.parameters():
            p.requires_grad_(False)
        new_act, logp = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, new_act)
        q_pi = torch.min(q1_pi, q2_pi)

        bc_loss = torch.zeros((), device=self.device)
        bc_coef = self._bc_coef(step)

        # Scale of the Q term. Returns on this task run to several hundred, and
        # a behaviour-cloning MSE is of order 0.01, so an unnormalised sum is
        # not a trade-off between the two -- it is the Q term with a rounding
        # error attached, and the cloned policy is destroyed within a couple of
        # thousand updates. Dividing by the mean absolute Q, as TD3+BC does,
        # makes the coefficient mean what it says.
        if bc_coef > 0.0 or cfg.anchor_coef > 0.0:
            q_weight = cfg.bc_q_scale / q_pi.abs().mean().detach().clamp(min=1e-6)
        else:
            q_weight = torch.ones((), device=self.device)
        actor_loss = (self.alpha.detach() * logp - q_weight * q_pi).mean()

        if cfg.anchor_coef > 0.0 and self._anchor is not None:
            # A frozen-policy anchor, for fine-tuning from a checkpoint rather
            # than from demonstrations. The behaviour-cloning term above needs
            # expert actions in the batch; when the initialisation is a trained
            # policy there are none, and this repository's own measurements say
            # what happens then: starting SAC at `medium` from a nominal
            # checkpoint that transfers at 0.200 leaves it at 0.000 within
            # 50 000 steps. Unanchored RL walks away from a good initialisation
            # and the entropy term is what walks it.
            #
            # Same normalisation as the BC term, and for the same reason: an
            # unnormalised sum is the Q term with a rounding error attached.
            with torch.no_grad():
                ref_act, _ = self._anchor(obs, deterministic=True,
                                          with_logprob=False)
            pi_all, _ = self.actor(obs, deterministic=True, with_logprob=False)
            anchor_loss = (pi_all - ref_act).pow(2).mean()
            actor_loss = actor_loss + cfg.anchor_coef * anchor_loss

        if bc_coef > 0.0 and n_demo > 0:
            # Behaviour-cloning term, on the demonstration slice of the batch.
            #
            # The Q-filter -- applying the term only where the critic prefers the
            # expert's action to the policy's -- is off by default, and that is
            # a deliberate correction rather than an oversight. Early in a
            # fine-tuning run the critic systematically overrates the actions
            # the policy is already taking, so the mask is mostly zero and the
            # anchor disappears at exactly the moment it is holding the cloned
            # policy together. Measured: with the filter on, a clone scoring
            # 0.70 was destroyed within 1500 actor updates; with it off, the
            # same run keeps the clone and improves on it.
            #
            # It is worth having as an option because late in training the
            # argument for it is real: a policy that has genuinely overtaken the
            # demonstrations should not be dragged back to them. By then the
            # coefficient has decayed anyway, which is the cheaper fix.
            demo_obs, demo_act = obs[:n_demo], act[:n_demo]
            pi_act, _ = self.actor(demo_obs, deterministic=True, with_logprob=False)
            per_sample = (pi_act - demo_act).pow(2).mean(dim=-1)
            if cfg.bc_q_filter:
                with torch.no_grad():
                    q_demo = torch.min(*self.critic(demo_obs, demo_act))
                    q_pi_demo = torch.min(*self.critic(demo_obs, pi_act))
                    per_sample = per_sample * (q_demo > q_pi_demo).float()
            bc_loss = per_sample.mean()
            actor_loss = actor_loss + bc_coef * bc_loss

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        # ---- entropy coefficient
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        if cfg.alpha_floor > 0.0:
            # A floor under the entropy coefficient. Automatic tuning drives
            # alpha towards zero once the policy is confident, which is correct
            # when the policy is confident about the *right* thing and fatal
            # when it has settled into a local optimum: with alpha at 0.02 the
            # policy is effectively deterministic and never tries anything else.
            with torch.no_grad():
                self.log_alpha.clamp_(min=float(np.log(cfg.alpha_floor)))

        # ---- target networks
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.mul_(1.0 - cfg.tau).add_(cfg.tau * p)

        self._updates += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "bc_loss": float(bc_loss.detach()),
            "bc_coef": float(bc_coef),
            "alpha": float(self.alpha.detach()),
            "entropy": float(-logp.mean().detach()),
            "q_mean": float(q_pi.mean().detach()),
            "warmup": 0.0,
        }

    def _bc_coef(self, step: int) -> float:
        if self.cfg.bc_coef <= 0.0:
            return 0.0
        if self.cfg.bc_decay_steps <= 0:
            return self.cfg.bc_coef
        frac = max(0.0, 1.0 - step / float(self.cfg.bc_decay_steps))
        return self.cfg.bc_coef * frac

    # ------------------------------------------------------------------
    def state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "config": self.cfg.to_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.actor.load_state_dict(state["actor"])
        if "critic" in state:
            self.critic.load_state_dict(state["critic"])
            self.critic_target = copy.deepcopy(self.critic)
            for p in self.critic_target.parameters():
                p.requires_grad_(False)
        if "log_alpha" in state:
            with torch.no_grad():
                self.log_alpha.copy_(state["log_alpha"].to(self.device))
