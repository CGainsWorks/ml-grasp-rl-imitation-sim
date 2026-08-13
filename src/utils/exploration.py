"""Colored exploration noise.

SAC samples its actions from an uncorrelated Gaussian, which is *white* noise:
every step is independent of the last. That is a poor way to explore a task
whose solution is a temporally extended action. Lifting a box means twenty
consecutive upward commands; independent samples average out to nearly no net
displacement, so the behaviour is essentially never tried, and a policy that
has already learned to grasp sits in the "hold it on the table" optimum
indefinitely. That is the failure this repository measured on three of five
seeds, and reproduced in a second simulator.

Eberhard, Hollenstein, Pinneri and Martius, *Pink Noise Is All You Need:
Colored Noise Exploration in Deep Reinforcement Learning* (ICLR 2023), evaluate
the whole colored-noise family on SAC and MPO and find pink noise -- halfway
between white and Brownian -- beats white noise, OU noise and the rest across a
wide range of continuous-control tasks, and recommend it as the default.

    https://openreview.net/forum?id=hQ9V5QN27eS

A colored-noise sequence has a power spectrum proportional to ``1 / f**beta``:

===========  ======  ==============================================
Name         beta    Character
===========  ======  ==============================================
white        0       independent samples, what SAC does by default
pink         1       correlated, but still mean-reverting
red/Brown    2       a random walk; OU noise is close to this
===========  ======  ==============================================

The sequence is generated per episode by shaping a white spectrum and taking an
inverse FFT, which is the method the paper uses. It is cheap: one FFT per
episode per action dimension.
"""

from __future__ import annotations

import numpy as np

WHITE, PINK, RED = 0.0, 1.0, 2.0


def colored_noise(beta: float, steps: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """A ``(steps, dim)`` array of unit-variance noise with a 1/f**beta spectrum.

    ``beta = 0`` reproduces ordinary white Gaussian noise, so the caller can
    switch colours without switching code paths.
    """
    if steps < 2:
        return rng.normal(size=(steps, dim))
    if beta == 0.0:
        return rng.normal(size=(steps, dim))

    freqs = np.fft.rfftfreq(steps)
    scale = np.ones_like(freqs)
    # The zero-frequency (constant) component is left unscaled; giving it the
    # 1/f weight would add an arbitrary offset to the whole episode.
    scale[1:] = freqs[1:] ** (-beta / 2.0)

    out = np.empty((steps, dim), dtype=np.float64)
    for d in range(dim):
        spectrum = (
            rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
        ) * scale
        signal = np.fft.irfft(spectrum, n=steps)
        std = signal.std()
        out[:, d] = signal / std if std > 1e-12 else signal
    return out


class ColoredNoiseProcess:
    """Per-episode colored noise, refreshed on reset.

    Holds one sequence of length ``horizon`` and walks through it, so the noise
    a policy sees within an episode is correlated in time but independent
    between episodes.
    """

    def __init__(
        self,
        beta: float,
        horizon: int,
        dim: int,
        rng: np.random.Generator,
    ) -> None:
        self.beta = float(beta)
        self.horizon = int(horizon)
        self.dim = int(dim)
        self.rng = rng
        self._sequence = colored_noise(self.beta, self.horizon, self.dim, self.rng)
        self._step = 0

    def reset(self) -> None:
        self._sequence = colored_noise(self.beta, self.horizon, self.dim, self.rng)
        self._step = 0

    def sample(self) -> np.ndarray:
        """The next noise vector, wrapping if an episode outruns the horizon."""
        if self._step >= self.horizon:
            self.reset()
        value = self._sequence[self._step]
        self._step += 1
        return value
