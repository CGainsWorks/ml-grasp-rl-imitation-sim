# Where the randomisation ranges come from

The four training levels (`none`, `low`, `medium`, `high`) were chosen for
plausibility. `docs/limitations.md` has said so from the start: "measured
randomisation ranges instead of plausible ones" has been the second item on the
what-would-change-these-numbers list.

No hardware was available to measure them, so the next best thing is to check
them against published measurements. That is what this document does, and the
answer is not flattering: **the hand-picked ranges are optimistic on the two
axes that matter most for this task, and they omit one axis entirely.**

`src/randomisation/configs/measured.json` is the result — an *evaluation*
distribution built from the numbers below rather than a training curriculum.

## The comparison

`medium` is the reference column because it is the level most results are
quoted at. The control rate is 25 Hz, so one step of latency is 40 ms.

| parameter | `medium` | published | verdict |
| --- | --- | --- | --- |
| object friction | µ 0.90–1.60 | 0.46 (rubber on rubber) to 1.9 (rubber on cardboard at low load) | **too narrow**, and centred too high — misses the slippery half |
| command latency | 0–80 ms (0–144 ms at `high`) | 55 ms perception-to-command; 170–250 ms end to end; 350–440 ms on a commodity teleoperation stack | **optimistic by 2–5×** |
| object position error | 0–4 mm | 5.6 mm tracking, 6.8 mm single-shot on YCB-Video | **optimistic**, and the good end is 0 mm, which no estimator achieves |
| object orientation error | *not randomised* | 8.1° tracking, 17.6° single-shot | **missing entirely** |
| gravity | ±5% | ±0.02% anywhere on Earth | absurdly generous, and harmless |

The gravity row is worth keeping visible. Randomising gravity by ±5% is common
in sim-to-real papers and it is not defensible as a model of gravity — real
gravity varies by about 0.5% pole to equator. It is a proxy for unmodelled
mass/inertia error wearing gravity's name, and if that is what it is for, it
should be in `object_mass`, which is already randomised 0.5–2×.

## Sources

* **Friction.** Leddy and Dollar, [*Examining the Frictional Behavior of
  Primitive Contact Geometries for Use as Robotic Finger
  Pads*](https://www.eng.yale.edu/grablab/pubs/Leddy_RAL2020.pdf), IEEE RA-L
  5(2):3137–3144, 2020, measures effective static and kinetic coefficients for
  fabricated finger pads at 1 N, 12.5 N and 25 N normal load. Reported static
  coefficients for rubber pads span roughly 0.46 against rubber to 1.9 against
  cardboard at low load, falling about 20–26% as normal force rises over that
  range. The load dependence is itself unmodelled here: this environment's
  friction is a constant per episode.
* **Pose estimation error.** Translation error on the YCB-Video benchmark is
  around 5.6 mm for tracking and 6.8 mm for single-shot estimation, with
  rotation error 8.1° and 17.6° respectively — see the survey of results
  collected for [SE(3)-PoseFlow](https://arxiv.org/pdf/2511.01501) and the
  original [DenseFusion](https://arxiv.org/pdf/1901.04780) benchmark protocol.
  The 0.14–0.31 rad range in `measured.json` is 8°–18°.
* **Latency.** [PEERNet](https://arxiv.org/pdf/2409.06078) profiles end-to-end
  networked robot pipelines; reported figures elsewhere put perception-to-command
  at about 55 ms, human-to-motion at 170–250 ms, and a commodity teleoperation
  stack at 350–440 ms measured on 240 fps video. Both the UR5 and the Franka
  take commands at about 30 Hz and interpolate internally (125 Hz and 1000 Hz),
  so a policy running at 25 Hz is already near the command rate before any
  network hop.

## What this level does *not* claim

It is not a real robot. Every caveat in [sim-to-real.md](sim-to-real.md) still
applies: no camera, no detector, no arm dynamics, one object shape, and an error
model that is Gaussian and independent per step where a real estimator's error is
correlated in time and worst exactly when the gripper occludes the object. A
policy that scored well here would still have to be proven on hardware.

What it does is remove one specific excuse. "The randomisation ranges were
guessed" can no longer be answered with "but they were sensible guesses",
because the guesses can now be checked, and two of them were wrong in the
direction that makes the results look better.
