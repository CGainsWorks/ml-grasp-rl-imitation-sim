"""The pose estimator in the loop, rather than substituted at evaluation time.

`experiments/perception_eval.py` takes a policy trained on ground truth and
swaps the CNN's estimate in at evaluation. That measures what perception *costs*
a policy that never saw it -- 18 points -- which is a useful number and a
different question from the one a robot asks. A robot has never had ground
truth. It has to be trained through whatever its camera actually gives it.

This wrapper is that: every observation the policy receives has the object's
position replaced by the estimate from a 64x64 render, at training time and at
evaluation time alike. The substitution is the same one `perception_eval.py`
makes, and deliberately shares its indices, so the two are comparable.

Two things make training through it feasible.

The first is that ``clean_observation`` still returns the true state. That is
what privileged distillation needs: a demonstrator that can see teaching a
policy that cannot. Trying to demonstrate *through* the estimator does not work
for the same reason it did not work at `measured_camera` -- an expert servoing
on a position that is wrong by more than the box is wide does not produce
demonstrations worth imitating.

The second is that rendering dominates the cost. Measured on this machine, a
step through the wrapper takes **75.9 ms** against roughly 0.3 ms for the
state-based environment -- the render, not the 200k-parameter network, is where
it goes. That is a factor of about 250, and it decides what can be run: 20 000
demonstration transitions is about 25 minutes, while 200 000 environment steps
of RL would be over four hours. So the runs here are cloning, and from-scratch
RL through the camera is left undone rather than claimed.

The estimator is frozen, and unfreezing it was tried. Retraining on 12 000
frames from a trained policy's own state distribution improves its error from
0.0499 m to 0.0477 m and the policy from 0.728 to 0.734 -- nothing, on
overlapping intervals. The collection report says why: on-policy frames flag
0.0% as hard, because a trained policy keeps the box in view. DAgger's premise
is that the learner wanders where the demonstrator never went; here it wanders
somewhere easier. `scripts/collect_pose_data.py --policy` is the machinery if a
harder camera ever makes it worth revisiting.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch

from src.perception.pose_cnn import PoseCNN

# Shared with experiments/perception_eval.py on purpose: if these drift the two
# measurements stop being comparable, and the drift would be silent.
OBJ = slice(8, 11)
OBJ_MINUS_GRIP = slice(11, 14)
GOAL = slice(26, 29)
GOAL_MINUS_OBJ = slice(29, 32)
GRIP = slice(0, 3)

DEFAULT_CHECKPOINT = os.path.join("experiments", "perception", "pose_cnn.pt")
WRIST_CHECKPOINT = os.path.join("experiments", "perception",
                                "pose_cnn_wrist.pt")
# Which camera each checkpoint was trained on. Pairing a checkpoint with
# the wrong view does not fail -- both are 64x64 RGB -- it just produces a
# confidently wrong pose, which is the hardest kind of error to notice.
WRIST_DAGGER_CHECKPOINT = os.path.join("experiments", "perception",
                                       "pose_cnn_wrist_dagger.pt")
CHECKPOINT_CAMERA = {DEFAULT_CHECKPOINT: "front_cam",
                     WRIST_CHECKPOINT: "wrist_cam",
                     WRIST_DAGGER_CHECKPOINT: "wrist_cam"}


class PerceptionEnv:
    """Wrap a `GraspEnv` so the object's position comes from a camera.

    Delegates everything it does not override, so it can be passed anywhere the
    unwrapped environment is accepted.
    """

    def __init__(self, env, checkpoint: str = DEFAULT_CHECKPOINT,
                 device: str = "cpu") -> None:
        if env.history != 1:
            # The window would have to be rebuilt from substituted frames
            # rather than true ones, and nothing here needs it yet. Better to
            # refuse than to stack a mixture and report the number anyway.
            raise ValueError(
                "perception wrapping and observation history are not combined "
                "yet; the window would mix estimated and true frames")
        self.env = env
        self.device = torch.device(device)
        self.model = PoseCNN()
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self._last_error: Optional[float] = None

    def __getattr__(self, name):
        # Only reached for attributes this class does not define.
        return getattr(self.env, name)

    def _estimate(self, obs: np.ndarray) -> np.ndarray:
        frame = self.env.render()
        if frame.shape[:2] != (64, 64):
            raise ValueError(
                "estimator expects 64x64 frames, got {}x{}: the checkpoint was "
                "trained at that size and a different one is wrong rather than "
                "merely worse".format(frame.shape[0], frame.shape[1]))
        with torch.no_grad():
            x = torch.as_tensor(frame, dtype=torch.float32, device=self.device)
            x = x.permute(2, 0, 1)[None] / 255.0
            pred = self.model(x)[0].cpu().numpy()
        # Recorded so a run can report how good its own eyes were rather than
        # leaving that to be inferred from the success rate.
        self._last_error = float(np.linalg.norm(pred - obs[OBJ]))
        out = obs.copy()
        out[OBJ] = pred
        out[OBJ_MINUS_GRIP] = pred - out[GRIP]
        out[GOAL_MINUS_OBJ] = out[GOAL] - pred
        return out

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        return self._estimate(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._last_error is not None:
            info = dict(info, pose_error=self._last_error)
        return self._estimate(obs), reward, terminated, truncated, info

    def clean_observation(self) -> np.ndarray:
        """True state, for a privileged demonstrator only.

        Deliberately not routed through the estimator: this is the signal the
        demonstrator uses precisely because the policy will not have it.
        """
        return self.env.clean_observation()

    @property
    def last_pose_error(self) -> Optional[float]:
        return self._last_error


def make_perception_env(*args, checkpoint: str = DEFAULT_CHECKPOINT,
                        device: str = "cpu", **kwargs) -> PerceptionEnv:
    """`make_env`, with the camera in the loop.

    Forces ``render_mode='rgb_array'``: the wrapper cannot work without it, and
    failing here is better than failing inside the first render.
    """
    from envs.mujoco.grasp_env import make_env

    # The estimator was trained on 64x64 frames from the front camera. Passing
    # anything else silently changes what it is looking at, so these are set
    # here rather than left to the caller: a different size does not fail, it
    # just makes the network wrong.
    kwargs["render_mode"] = "rgb_array"
    kwargs.setdefault("width", 64)
    kwargs.setdefault("height", 64)
    expected = CHECKPOINT_CAMERA.get(os.path.normpath(checkpoint))
    kwargs.setdefault("camera", expected or "front_cam")
    if expected is not None and kwargs["camera"] != expected:
        raise ValueError(
            "{} was trained on {} and this environment renders {}. Both are "
            "64x64 RGB so nothing would fail; the estimator would simply be "
            "wrong.".format(os.path.basename(checkpoint), expected,
                            kwargs["camera"]))
    return PerceptionEnv(make_env(*args, **kwargs), checkpoint=checkpoint,
                         device=device)
