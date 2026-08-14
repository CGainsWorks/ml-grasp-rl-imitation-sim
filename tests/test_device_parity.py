"""Putting the agent on the GPU must not change what the agent computes.

The Isaac Lab port's whole claim is "the same algorithm in a second simulator",
so an agent that runs on CUDA there and on the CPU in MuJoCo would quietly
undermine the comparison the port exists to support. This file is the same
treatment `test_reward_parity.py` gives the reward function, applied to the
training loop's two device-sensitive pieces.

The batched buffer write is checked in CI, because it is pure numpy and the
Python-loop version it replaces is still there to compare against. The
CPU-against-CUDA comparison skips where there is no card, which includes CI --
it is meant to be run on the machine that will use the flag:

    C:\\isaac\\venv311\\Scripts\\python.exe -m pytest tests/test_device_parity.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.policies.sac import SAC, ReplayBuffer, SACConfig

torch = pytest.importorskip("torch")

OBS_DIM, ACT_DIM = 6, 3


def _transitions(n: int, rng: np.random.Generator):
    return (
        rng.normal(size=(n, OBS_DIM)).astype(np.float32),
        rng.uniform(-1.0, 1.0, size=(n, ACT_DIM)).astype(np.float32),
        rng.normal(size=n).astype(np.float32),
        rng.normal(size=(n, OBS_DIM)).astype(np.float32),
        (rng.random(n) < 0.1).astype(np.float32),
    )


def test_batched_add_matches_the_loop_it_replaces():
    """add_batch(n) has to equal n calls to add(), including the ring wrap."""
    rng = np.random.default_rng(0)
    obs, act, rew, next_obs, done = _transitions(150, rng)

    loop = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=64)
    for i in range(len(obs)):
        loop.add(obs[i], act[i], rew[i], next_obs[i], done[i])

    batched = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=64)
    for start in range(0, len(obs), 32):
        stop = start + 32
        batched.add_batch(obs[start:stop], act[start:stop], rew[start:stop],
                          next_obs[start:stop], done[start:stop])

    assert batched.size == loop.size
    assert batched.ptr == loop.ptr
    for name in ("obs", "act", "rew", "next_obs", "done"):
        np.testing.assert_array_equal(getattr(batched, name), getattr(loop, name))


def test_batched_add_respects_the_pinned_demonstrations():
    """The pinned prefix is the point of this buffer; a batched write must not
    reach into it however many times the ring wraps."""
    rng = np.random.default_rng(1)
    demo = _transitions(10, rng)
    buffer = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=40)
    buffer.add_demonstrations(*demo)
    before = buffer.obs[:10].copy()

    obs, act, rew, next_obs, done = _transitions(200, rng)
    for start in range(0, len(obs), 32):
        stop = start + 32
        buffer.add_batch(obs[start:stop], act[start:stop], rew[start:stop],
                         next_obs[start:stop], done[start:stop])

    np.testing.assert_array_equal(buffer.obs[:10], before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_actor_agrees_between_cpu_and_cuda():
    """Same weights, same observations, both devices."""
    cfg = SACConfig(hidden=(32, 32))
    cpu = SAC(OBS_DIM, ACT_DIM, cfg, seed=0, device="cpu")
    cuda = SAC(OBS_DIM, ACT_DIM, cfg, seed=0, device="cuda")
    cuda.actor.load_state_dict(cpu.actor.state_dict())

    rng = np.random.default_rng(2)
    obs = rng.normal(size=(64, OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        a_cpu, _ = cpu.actor(torch.as_tensor(obs), deterministic=True,
                             with_logprob=False)
        a_cuda, _ = cuda.actor(torch.as_tensor(obs).cuda(), deterministic=True,
                               with_logprob=False)
    np.testing.assert_allclose(a_cpu.numpy(), a_cuda.cpu().numpy(),
                               rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_critic_agrees_between_cpu_and_cuda():
    """The other half of the forward pass, on identical inputs."""
    cfg = SACConfig(hidden=(32, 32))
    cpu = SAC(OBS_DIM, ACT_DIM, cfg, seed=0, device="cpu")
    cuda = SAC(OBS_DIM, ACT_DIM, cfg, seed=0, device="cuda")
    cuda.critic.load_state_dict(cpu.critic.state_dict())

    rng = np.random.default_rng(4)
    obs = rng.normal(size=(64, OBS_DIM)).astype(np.float32)
    act = rng.uniform(-1.0, 1.0, size=(64, ACT_DIM)).astype(np.float32)
    with torch.no_grad():
        q_cpu = cpu.critic(torch.as_tensor(obs), torch.as_tensor(act))
        q_cuda = cuda.critic(torch.as_tensor(obs).cuda(), torch.as_tensor(act).cuda())
    for a, b in zip(q_cpu, q_cuda):
        np.testing.assert_allclose(a.numpy(), b.cpu().numpy(), rtol=1e-4, atol=1e-5)


def _train(agent, batch, torch_seed: int, steps: int = 20):
    agent.buffer.add_batch(*batch)
    agent.observe_normalisation(batch[0])
    agent.rng = np.random.default_rng(7)      # identical minibatch indices
    torch.manual_seed(torch_seed)
    if agent.device.type == "cuda":
        torch.cuda.manual_seed_all(torch_seed)
    metrics = {}
    for step in range(1, steps + 1):
        metrics = agent.update(step)
    return metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_training_on_cuda_drifts_no_more_than_the_noise_already_does():
    """Twenty updates on identical data, with a control for the RNG.

    Step-for-step equality is not available and asking for it would be a
    misunderstanding: SAC's actor samples through ``torch.randn_like`` at every
    update, and the CPU and CUDA generators are different implementations, so
    the same seed does not give the same stream. The two agents genuinely see
    different exploration noise.

    So the comparison needs a control. Two *CPU* agents, identical weights and
    minibatches, differing only in their torch seed, measure how far apart the
    noise alone drives the losses. A CPU agent and a CUDA agent are then
    required to end up no further apart than that. If the device introduced an
    error of its own, this fails; if it only reshuffles the noise, it passes.
    """
    cfg = SACConfig(hidden=(32, 32), batch_size=64, start_steps=0)
    reference = SAC(OBS_DIM, ACT_DIM, cfg, seed=0, device="cpu")
    noise_control = SAC(OBS_DIM, ACT_DIM, cfg, seed=0, device="cpu")
    on_cuda = SAC(OBS_DIM, ACT_DIM, cfg, seed=0, device="cuda")
    for other in (noise_control, on_cuda):
        for attr in ("actor", "critic", "critic_target"):
            getattr(other, attr).load_state_dict(getattr(reference, attr).state_dict())

    rng = np.random.default_rng(3)
    batch = _transitions(512, rng)
    m_ref = _train(reference, batch, torch_seed=11)
    m_noise = _train(noise_control, batch, torch_seed=12)
    m_cuda = _train(on_cuda, batch, torch_seed=11)

    for key in ("critic_loss", "actor_loss", "alpha"):
        from_noise = abs(m_ref[key] - m_noise[key])
        from_device = abs(m_ref[key] - m_cuda[key])
        assert from_device <= max(3.0 * from_noise, 1e-3), (
            "{}: moving to CUDA shifted the loss by {:.5f}, where changing only "
            "the noise seed shifts it by {:.5f}".format(key, from_device, from_noise)
        )
