"""Render rollout videos.

    python src/render_rollout.py --policy experiments/runs/sac_medium_s0/policy.pt \
        --randomisation medium --episodes 3 --output videos/sac_medium.mp4

    python src/render_rollout.py --expert --randomisation none --gif videos/expert.gif

Writes an MP4 when ``--output`` ends in ``.mp4`` and an animated GIF when
``--gif`` is given; the GIFs are what the README embeds, because a GIF renders
inline on GitHub and a video file does not.

Rendering needs a GL context. On a desktop that is automatic; headless (CI,
a container without a display) it needs EGL or OSMesa, which is why no video
target runs in CI. ``MUJOCO_GL=egl`` is the usual incantation on a Linux box
with an NVIDIA driver, ``MUJOCO_GL=osmesa`` for software rendering.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import make_env  # noqa: E402
from src.policies.scripted_expert import ScriptedExpert  # noqa: E402
from src.policies.scripted_place_expert import ScriptedPlaceExpert  # noqa: E402


def _overlay(frame: np.ndarray, ok: bool, height_frac: float) -> np.ndarray:
    """Draw a two-pixel status bar: green while succeeding, plus a lift gauge.

    Deliberately crude. It avoids a font dependency and it makes a contact
    sheet of rollouts readable at a glance: a run that never turns green never
    held the box.
    """
    out = frame.copy()
    bar = np.array([40, 200, 90], dtype=np.uint8) if ok else np.array([200, 70, 60], dtype=np.uint8)
    out[:4, :, :] = bar
    width = out.shape[1]
    filled = int(np.clip(height_frac, 0.0, 1.0) * width)
    out[-6:, :filled, :] = np.array([90, 160, 230], dtype=np.uint8)
    return out


def render(
    policy_fn,
    randomisation: str,
    episodes: int,
    seed: int,
    camera: str,
    width: int,
    height: int,
    max_steps: int,
    on_episode_start=None,
    task: str = "lift",
    arm: bool = False,
    handled: bool = False,
    wrist: bool = False,
    max_half_size=None,
    perception=None,
) -> List[np.ndarray]:
    if perception:
        # A policy trained through the camera has to be *shown* through the
        # camera, or the clip is of a different task than the caption claims.
        from envs.mujoco.perception_env import make_perception_env

        env = make_perception_env(
            randomisation, seed=seed, max_steps=max_steps, task=task, arm=arm,
            handled=handled, wrist=handled or wrist,
            max_half_size=max_half_size, checkpoint=perception,
        )
        # The wrapper pins the estimator to its own 64x64 front camera, which is
        # the wrong size and angle for a clip. A second environment is stepped
        # with the identical action sequence from the identical seed and used
        # only for the picture. Both are deterministic given (seed, actions), so
        # the film shows the trajectory the policy actually flew -- but it is a
        # replay rather than the same object, and if that ever stopped holding
        # the clip would quietly disagree with the run.
        film = make_env(
            randomisation, seed=seed, render_mode="rgb_array", camera=camera,
            width=width, height=height, max_steps=max_steps, task=task, arm=arm,
            handled=handled, wrist=handled or wrist,
            max_half_size=max_half_size,
        )
    else:
        film = None
        env = make_env(
            randomisation, seed=seed, render_mode="rgb_array", camera=camera,
            width=width, height=height, max_steps=max_steps, task=task, arm=arm,
            handled=handled, wrist=handled or wrist,
            max_half_size=max_half_size,
        )
    frames: List[np.ndarray] = []
    successes = 0
    for ep in range(episodes):
        if on_episode_start is not None:
            on_episode_start()
        obs, _ = env.reset(seed=seed + ep)
        if film is not None:
            film.reset(seed=seed + ep)
        info = {}
        while True:
            action = policy_fn(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if film is not None:
                film.step(action)
            frames.append(
                _overlay(
                    (film if film is not None else env).render(),
                    bool(info.get("is_success", False)),
                    float(info.get("object_height", 0.0)) / (0.06 if task == "place" else 0.15),
                )
            )
            if terminated or truncated:
                break
        successes += int(bool(info.get("is_success", False)))
        # Two frames of black between episodes, so a viewer can see the cut.
        frames.extend([np.zeros_like(frames[-1])] * 2)
    env.close()
    if film is not None:
        film.close()
    print("rendered {} episodes, {} successful, {} frames".format(episodes, successes, len(frames)))
    return frames


def load_policy(path: str):
    import torch

    from src.evaluate import load_actor

    torch.set_num_threads(1)
    actor = load_actor(path)
    return lambda obs: actor.act(obs, deterministic=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=None, help="checkpoint to roll out")
    parser.add_argument("--expert", action="store_true", help="roll out the scripted expert")
    parser.add_argument("--randomisation", default="none")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=900_000,
                        help="matches the evaluation seed block, so the clips are "
                             "episodes from the evaluated set rather than cherry-picked")
    parser.add_argument("--camera", default="scene_cam")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--task", default="lift", choices=("lift", "place"))
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--handled", action="store_true",
                        help="the grasp-point-selection shape; implies --wrist")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--output", default=None, help="path to an .mp4")
    parser.add_argument("--gif", default=None, help="path to a .gif")
    parser.add_argument("--gif-stride", type=int, default=2, help="keep every Nth frame in the GIF")
    parser.add_argument("--gif-width", type=int, default=320)
    args = parser.parse_args()

    # Take the robot from the run rather than the command line. A clip of an
    # arm policy rendered on the welded hand does not fail -- the observation
    # widths coincide -- it just films a different robot than the caption says.
    arm, handled, wrist = args.arm, args.handled, False
    max_half_size, perception = None, None
    if args.policy:
        cfg_path = os.path.join(os.path.dirname(args.policy), "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            arm = arm or bool(cfg.get("arm", False))
            handled = handled or bool(cfg.get("handled", False))
            wrist = bool(cfg.get("wrist", False))
            max_half_size = cfg.get("max_half_size")
            perception = cfg.get("perception")

    if not args.policy and not args.expert:
        parser.error("give --policy or --expert")

    on_start: Optional[callable] = None
    if args.expert:
        if args.handled:
            from envs.mujoco.grasp_env import HANDLE_CENTRE, HANDLE_HEIGHT
            expert = ScriptedExpert(wrist=True, grasp_offset=HANDLE_CENTRE,
                                    grasp_yaw_offset=np.pi / 2.0,
                                    grasp_height=HANDLE_HEIGHT)
        elif args.task == "place":
            expert = ScriptedPlaceExpert()
        else:
            expert = ScriptedExpert()
        on_start = expert.reset
        policy_fn = expert.act
    else:
        policy_fn = load_policy(args.policy)

    frames = render(
        policy_fn, args.randomisation, args.episodes, args.seed,
        args.camera, args.width, args.height, args.max_steps, on_start,
        args.task, arm, handled, wrist, max_half_size, perception,
    )

    import imageio.v2 as imageio

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        imageio.mimsave(args.output, frames, fps=args.fps, macro_block_size=1)
        print("wrote {} ({:.1f} MB)".format(args.output, os.path.getsize(args.output) / 1e6))

    if args.gif:
        os.makedirs(os.path.dirname(os.path.abspath(args.gif)), exist_ok=True)
        small = frames[:: max(1, args.gif_stride)]
        if args.gif_width and args.gif_width != args.width:
            scale = args.width // args.gif_width
            if scale > 1:
                small = [f[::scale, ::scale] for f in small]
        imageio.mimsave(args.gif, small, duration=1000.0 / (args.fps / max(1, args.gif_stride)),
                        loop=0)
        print("wrote {} ({:.1f} MB)".format(args.gif, os.path.getsize(args.gif) / 1e6))


if __name__ == "__main__":
    main()
