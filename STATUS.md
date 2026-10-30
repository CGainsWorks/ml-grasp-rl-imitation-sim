# Status

Last updated 2026-10-16.

Everything in this repository is simulated. MuJoCo 3.11 runs on CPU and is
exercised in CI; Isaac Sim 5.1 needs an NVIDIA RTX GPU and was run by hand on
one local RTX 4060; no robot has been used for anything.

## Complete

* **MuJoCo tasks**: lift-and-hold and pick-and-place, the optional six-jointed
  UR5-proportioned arm, the wrist joint and the grasp-point (handled) shape,
  with the reward design documented term by term.
* **Learning**: SAC written out, behaviour cloning, DAgger, BC plus SAC with
  pinned demonstrations, sparse reward with hindsight relabelling and the
  entropy-floor fix for the stalled seeds.
* **Evaluation**: one definition of success, seed blocks held apart from
  training, mean over seeds with 95% t intervals, ten seeds on the headline
  grid and five elsewhere.
* **Domain randomisation ablation**: four training levels against a held-out
  `shifted` distribution, from scratch, with the floor and seeded with
  demonstrations.
* **Perception**: a CNN pose estimator on 64x64 renders, fixed and wrist
  cameras, trained and evaluated in the loop.
* **Traceability**: every number in the README tables is written by
  `analysis/readme_tables.py` from `experiments/results/`; and
  `scripts/check_docs.py` recomputes the headline claims, the other quoted
  values (Isaac grids, perception error statistics, the arm rows) and the test
  counts on this page and in the README on every CI run. Committed run records
  use repository-relative POSIX paths, enforced by a test.
* **CI, MuJoCo job**: lint, 98 unit tests in `tests/` (three CUDA parity
  tests skip on the CPU runners), the documents check, the scripted expert
  and a smoke chain of SAC training, evaluation on three levels, TorchScript
  export, a closed-loop episode driven by the exported policy, demonstration
  recording and behaviour cloning. Python 3.10 and 3.12.
* **ROS 2 policy server** (`ros2/grasp_policy_server`, Humble, `ament_python`):
  an `rclpy` node serving 4-value actions for 32-value observations from an
  exported TorchScript policy, a replay node, a launch file, 10 unit tests of
  the policy wrapper and a launch test with 3 checks (the replayed stream
  reproduces the simulated episode's actions to 1e-5, a malformed observation
  is rejected without an action, both processes exit cleanly). Built and tested
  in `ros:humble-ros-base-jammy` by the CI `ros2` job and by `make ros-test`.
* **Isaac Lab port**: both tasks, bring-up checks, five-seed grids at four
  randomisation levels and the cross-simulator studies, run by hand. Import
  and shared-code tests run in CI. What was run and where the evidence is:
  [envs/isaac/README.md](envs/isaac/README.md#what-has-been-run-and-how).

## In progress

Nothing is being worked on.

## Not started

* **Connecting the policy server to `CGainsWorks/ros2-industrial-arm-cell-sim`.**
  Needs an adapter that builds the observation from the cell's TF, joint states
  and perception (which has no object tracker) and turns the action into
  checked motion. Listed in [docs/ros2-policy-server.md](docs/ros2-policy-server.md).
* **Moving the run records to a release artefact** to shrink the clone; they
  are about 2 300 small files today.
* Rendering in CI (needs a headless GL context).

## Limitations

* **No hardware.** `shifted` is a proxy for a real robot and the sensing the
  headline numbers assume is five to thirteen times better than the
  repository's own camera estimator delivers (README, Limitations).
* **Isaac results are manual and single-machine.** They need an RTX GPU, are
  not run in CI and have not been reproduced on another machine. The
  bring-up check results are transcribed in `envs/isaac/README.md` with no
  committed log.
* **CI training is a smoke test.** 2 000 SAC steps and 3 cloning epochs prove
  the pipeline runs; the learned numbers come from committed run records made
  locally on CPU, not from CI.
* **The ROS 2 node has only met recorded data.** It has not run against a live
  simulator, the arm cell or a robot; the policy it serves was trained on a
  free-floating hand, not the UR5e.
* **Superseded results are kept, not deleted.** Three summaries had been
  overwritten by later reruns; the earlier versions are restored from history
  as `ablation_bcrl_5seed.json`, `isaac_place_grid_64env.json` and
  `perception_error.json` (the fixed camera), next to the files that replaced
  them.
