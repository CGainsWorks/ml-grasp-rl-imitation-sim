"""Tests for the learning code that do not require a full training run.

None of this asserts that SAC reaches a given success rate: that takes twenty
minutes and belongs in ``make experiments``, not in CI. What is asserted here is
that the machinery is wired correctly, which is where the bugs that waste a
day's compute actually live -- a replay buffer that overwrites its pinned
demonstrations, an actor whose normalisation is not saved with it, a BC loop
that does not reduce its own training loss.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.policies.networks import RunningNorm, SquashedGaussianActor  # noqa: E402
from src.policies.sac import SAC, ReplayBuffer, SACConfig  # noqa: E402
from src.utils.stats import summarise_seeds, t_interval, wilson_interval  # noqa: E402

OBS, ACT = 32, 4


def test_actor_output_is_bounded():
    actor = SquashedGaussianActor(OBS, ACT)
    obs = torch.randn(64, OBS) * 10.0
    action, logp = actor(obs)
    assert action.shape == (64, ACT)
    assert torch.all(action.abs() <= 1.0)
    assert torch.all(torch.isfinite(logp))


def test_deterministic_action_is_repeatable():
    actor = SquashedGaussianActor(OBS, ACT)
    obs = np.random.default_rng(0).normal(size=OBS).astype(np.float32)
    a = actor.act(obs, deterministic=True)
    b = actor.act(obs, deterministic=True)
    assert np.allclose(a, b)


def test_stochastic_action_differs():
    actor = SquashedGaussianActor(OBS, ACT)
    obs = np.random.default_rng(0).normal(size=OBS).astype(np.float32)
    a = actor.act(obs, deterministic=False)
    b = actor.act(obs, deterministic=False)
    assert not np.allclose(a, b)


def test_running_norm_tracks_mean_and_variance():
    norm = RunningNorm(4)
    rng = np.random.default_rng(0)
    data = rng.normal(loc=3.0, scale=2.0, size=(5000, 4)).astype(np.float32)
    for start in range(0, 5000, 250):
        norm.update(torch.as_tensor(data[start : start + 250]))
    assert np.allclose(norm.mean.numpy(), data.mean(axis=0), atol=0.05)
    assert np.allclose(np.sqrt(norm.var.numpy()), data.std(axis=0), atol=0.05)


def test_normalisation_survives_a_round_trip():
    """A checkpoint that loses its normaliser looks exactly like a bad policy."""
    actor = SquashedGaussianActor(OBS, ACT)
    actor.norm.update(torch.randn(500, OBS) * 5.0 + 2.0)
    clone = SquashedGaussianActor(OBS, ACT)
    clone.load_state_dict(actor.state_dict())
    assert torch.allclose(clone.norm.mean, actor.norm.mean)
    assert torch.allclose(clone.norm.var, actor.norm.var)


def test_replay_buffer_never_overwrites_pinned_demonstrations():
    buffer = ReplayBuffer(OBS, ACT, capacity=1000)
    demo_obs = np.ones((200, OBS), dtype=np.float32)
    buffer.add_demonstrations(
        demo_obs, np.ones((200, ACT), dtype=np.float32), np.ones(200, dtype=np.float32),
        demo_obs, np.zeros(200, dtype=np.float32),
    )
    for _ in range(5000):
        buffer.add(np.zeros(OBS), np.zeros(ACT), 0.0, np.zeros(OBS), 0.0)
    assert np.all(buffer.obs[:200] == 1.0), "demonstrations were overwritten by the ring"


def test_mixed_sampling_puts_demonstrations_first():
    buffer = ReplayBuffer(OBS, ACT, capacity=1000)
    buffer.add_demonstrations(
        np.ones((100, OBS), dtype=np.float32), np.ones((100, ACT), dtype=np.float32),
        np.ones(100, dtype=np.float32), np.ones((100, OBS), dtype=np.float32),
        np.zeros(100, dtype=np.float32),
    )
    for _ in range(500):
        buffer.add(np.zeros(OBS), np.zeros(ACT), 0.0, np.zeros(OBS), 0.0)
    batch, n_demo = buffer.sample_mixed(64, 0.25, np.random.default_rng(0))
    assert n_demo == 16
    assert torch.all(batch["obs"][:n_demo] == 1.0)
    assert torch.all(batch["obs"][n_demo:] == 0.0)


def test_sac_update_runs_and_changes_parameters():
    agent = SAC(OBS, ACT, SACConfig(batch_size=32), seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        agent.buffer.add(
            rng.normal(size=OBS), rng.uniform(-1, 1, ACT), float(rng.normal()),
            rng.normal(size=OBS), 0.0,
        )
    before = agent.actor.mu_head.weight.detach().clone()
    metrics = agent.update(step=1)
    after = agent.actor.mu_head.weight.detach()
    assert not torch.allclose(before, after)
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])


def test_bc_coefficient_decays_to_zero():
    agent = SAC(OBS, ACT, SACConfig(bc_coef=1.0, bc_decay_steps=1000), seed=0)
    assert agent._bc_coef(0) == pytest.approx(1.0)
    assert agent._bc_coef(500) == pytest.approx(0.5)
    assert agent._bc_coef(1000) == pytest.approx(0.0)
    assert agent._bc_coef(5000) == pytest.approx(0.0)


def test_behaviour_cloning_reduces_its_loss():
    from src.train_il import fit

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(2000, OBS)).astype(np.float32)
    # A learnable target: a bounded linear map of the first few features.
    weights = rng.normal(size=(OBS, ACT)).astype(np.float32) * 0.1
    act = np.tanh(obs @ weights).astype(np.float32)
    actor = SquashedGaussianActor(OBS, ACT, (64, 64))
    train_curve, val_curve = fit(actor, obs, act, epochs=15, batch_size=128,
                                 lr=1e-3, weight_decay=0.0, rng=rng)
    assert train_curve[-1] < train_curve[0] * 0.5
    assert val_curve[-1] < val_curve[0]


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def test_wilson_interval_stays_inside_zero_one():
    for successes, trials in [(0, 20), (20, 20), (1, 3), (99, 100)]:
        interval = wilson_interval(successes, trials)
        assert 0.0 <= interval.low <= interval.point <= interval.high <= 1.0


def test_wilson_is_not_degenerate_at_the_extremes():
    """The normal approximation gives a zero-width interval at 0/n; Wilson does not."""
    interval = wilson_interval(0, 30)
    assert interval.high > 0.0


def test_t_interval_widens_with_seed_spread():
    tight = t_interval([0.80, 0.81, 0.79, 0.80, 0.80])
    loose = t_interval([0.20, 0.95, 0.55, 0.80, 0.30])
    assert (loose.high - loose.low) > (tight.high - tight.low)


def test_across_seed_interval_is_wider_than_pooled_when_seeds_disagree():
    """The point of reporting both: pooling hides seed variance."""
    summary = summarise_seeds([95, 20, 60, 88, 35], [100] * 5)
    across = summary["across_seeds"]
    pooled = summary["pooled_wilson"]
    assert (across["high"] - across["low"]) > (pooled["high"] - pooled["low"])


def test_exported_policy_matches_the_actor(tmp_path):
    """The export must carry its normalisation, and survive a TorchScript round trip."""
    from src.export_policy import DeployablePolicy
    from src.policies.networks import SquashedGaussianActor

    actor = SquashedGaussianActor(OBS, ACT, (64, 64))
    actor.norm.update(torch.randn(500, OBS) * 3.0 + 1.0)
    actor.eval()

    module = DeployablePolicy(actor).eval()
    path = str(tmp_path / "policy.ts.pt")
    torch.jit.script(module).save(path)
    loaded = torch.jit.load(path)

    obs = torch.randn(8, OBS)
    with torch.no_grad():
        direct = actor(obs, deterministic=True, with_logprob=False)[0]
        exported = loaded(obs)
    assert torch.allclose(direct, exported, atol=1e-6)


def test_colored_noise_is_correlated_in_time():
    """White noise is what SAC does by default; pink and red are progressively
    more correlated. If this ordering breaks, the exploration ablation is
    comparing three flavours of the same thing."""
    from src.utils.exploration import colored_noise

    rng = np.random.default_rng(0)
    autocorr = {}
    for name, beta in (("white", 0.0), ("pink", 1.0), ("red", 2.0)):
        seq = colored_noise(beta, 512, 4, rng)
        autocorr[name] = float(
            np.mean([np.corrcoef(seq[:-1, d], seq[1:, d])[0, 1] for d in range(4)])
        )
        assert abs(seq.std() - 1.0) < 0.2, name
    assert autocorr["white"] < 0.2
    assert autocorr["white"] < autocorr["pink"] < autocorr["red"]


def test_alpha_floor_holds_the_entropy_coefficient():
    """Automatic tuning drives alpha to zero on a confident policy; the floor is
    what stops that when the confidence is in a local optimum."""
    agent = SAC(OBS, ACT, SACConfig(batch_size=32, alpha_floor=0.15, init_alpha=0.2), seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        agent.buffer.add(rng.normal(size=OBS), rng.uniform(-1, 1, ACT), 0.0,
                         rng.normal(size=OBS), 0.0)
    for step in range(50):
        agent.update(step)
    assert float(agent.alpha) >= 0.15 - 1e-6
