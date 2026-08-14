"""Render every rollout clip the README embeds.

    python scripts/make_videos.py

Needs a GL context, so this is not part of CI. Each clip is rendered from the
final-evaluation seed block, which means the episodes shown are episodes from
the evaluated set rather than a hand-picked highlight reel. Failures that appear
in the clips are failures that appear in the numbers.

GIFs go to ``videos/`` and are committed, because GitHub renders them inline.
MP4s are written next to them and are gitignored; ``make videos`` regenerates
both.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIPS = [
    {
        "name": "expert_nominal",
        "args": ["--expert", "--randomisation", "none", "--episodes", "2"],
        "why": "the scripted expert on the clean world: the demonstrations' source",
    },
    {
        "name": "expert_shifted",
        "args": ["--expert", "--randomisation", "shifted", "--episodes", "3"],
        "why": "the same expert on the held-out worlds, where it starts to fail",
    },
    {
        "name": "sac_none",
        "policy": "experiments/runs/sac_none_s0/policy.pt",
        "args": ["--randomisation", "none", "--episodes", "3"],
        "why": "SAC from scratch on the nominal world -- one of the two seeds "
               "out of five that solved it",
    },
    {
        "name": "sac_none_stalled",
        "policy": "experiments/runs/sac_none_s3/policy.pt",
        "args": ["--randomisation", "none", "--episodes", "2"],
        "why": "a stalled seed of the same run: it grasps the box and holds it "
               "on the table, which is the local optimum the entropy collapse "
               "leaves it in",
    },
    {
        "name": "sac_none_floor",
        "policy": "experiments/runs/explore_alphafloor_s3/policy.pt",
        "args": ["--randomisation", "none", "--episodes", "2"],
        "why": "the same seed 3 with a floor under the entropy coefficient: the "
               "before-and-after of the one-line fix, with the seed held fixed "
               "so nothing else can explain the difference",
    },
    {
        "name": "bcrl_medium",
        "policy": "experiments/runs/bcrl_medium_s0/policy.pt",
        "args": ["--randomisation", "medium", "--episodes", "3"],
        "why": "imitation plus RL, on the distribution it trained on",
    },
    {
        "name": "bcrl_high_shifted",
        "policy": "experiments/runs/bcrl_high_s0/policy.pt",
        "args": ["--randomisation", "shifted", "--episodes", "3"],
        "why": "the widest-randomisation policy on the held-out shifted worlds",
    },
    {
        "name": "bcrl_none_shifted",
        "policy": "experiments/runs/bcrl_none_s0/policy.pt",
        "args": ["--randomisation", "shifted", "--episodes", "3"],
        "why": "the same method trained without randomisation, on the same "
               "worlds, for contrast",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="videos")
    parser.add_argument("--no-mp4", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    os.chdir(REPO)
    os.makedirs(args.outdir, exist_ok=True)

    for clip in CLIPS:
        if args.only and clip["name"] not in args.only:
            continue
        if "policy" in clip and not os.path.exists(clip["policy"]):
            print("skip {}: {} not trained yet".format(clip["name"], clip["policy"]))
            continue

        cmd = [sys.executable, "src/render_rollout.py", *clip["args"],
               "--gif", os.path.join(args.outdir, clip["name"] + ".gif")]
        if "policy" in clip:
            cmd += ["--policy", clip["policy"]]
        if not args.no_mp4:
            cmd += ["--output", os.path.join(args.outdir, clip["name"] + ".mp4")]

        print("\n=== {} : {}".format(clip["name"], clip["why"]))
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
