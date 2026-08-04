# ml-grasp-rl-imitation-sim

> **Status: planned. This repository is a brief, not an implementation.**
> Nothing here is built yet. The sections below define what goes in it.

> ### Clean-room reimplementation
> This repository is an **independent reimplementation of a problem class**, built from
> scratch on personal time using public or synthetic data and open-source simulation.
> It is **not** derived from, and contains no material belonging to, any employer or
> client: no employer source code, schematics, calibration data, client data or internal
> documentation. It exists to demonstrate the architecture and engineering approach only.

Learned grasping and trajectory policies trained purely in simulation. Covers reward design, imitation learning from recorded demonstrations, domain randomisation and an ablation showing how randomisation affects the sim-to-real gap. Uses open simulation assets throughout.

---

## What this covers

CV entries this repository is the evidence for:

- RL and imitation learning for grasp and trajectory optimisation (Convex Technology)

---

## Requirements

### Hardware

- Training GPU
- Optional physical target: any 6-DoF arm with a gripper for sim-to-real transfer

### Software

- MuJoCo
- NVIDIA Isaac Sim or Isaac Lab
- PyTorch
- Stable Baselines3 or a custom RL implementation
- Demonstration recording tools
- Domain randomisation config

### Report and documentation deliverables

- Reward function design
- Training curves
- Success rate in simulation against reality
- Sim-to-real gap analysis
- Ablation across randomisation settings

---

## Inputs

- Simulation scene and asset files
- Reward and termination definitions
- Expert demonstration trajectories
- Randomisation ranges
- Training hyperparameters

## Outputs

- Trained policy weights
- Training curves and metrics
- Evaluation success rate
- Rollout videos
- Exported policy for deployment

---

## Intended structure

```
envs/
  mujoco/                 grasp environments
  isaac/                  Isaac Lab task definitions
src/
  rewards/
  policies/
  train_rl.py
  train_il.py             behaviour cloning, DAgger
  evaluate.py
  randomisation/
demonstrations/           recorded expert trajectories
experiments/              training curves, ablation results
videos/                   rollout recordings
docs/
```

Plus the standard baseline: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`
(Git LFS rules for weights, bags and media), `.github/workflows/ci.yml`, `docs/`,
`scripts/setup.sh`, `scripts/run_demo.sh` and `tests/`.

---

## Build checklist

- [ ] MuJoCo grasp environment with a defined observation and action space
- [ ] Reward function design and documentation
- [ ] PPO or SAC baseline training
- [ ] Training curves and evaluation protocol
- [ ] Demonstration recording tooling
- [ ] Behaviour cloning from demonstrations
- [ ] Combined imitation plus RL fine-tuning
- [ ] Isaac Sim port of the task
- [ ] Domain randomisation implementation
- [ ] Randomisation ablation study
- [ ] Rollout videos and success-rate reporting

---

## Definition of done

This repository is finished when:

- [ ] The whole thing runs from one command on a clean machine.
- [ ] The README opens with a diagram or a GIF, not a wall of text.
- [ ] Every number quoted in the README is reproducible from a script in the repo.
- [ ] Limitations are stated honestly in their own section.
- [ ] A licence is declared.
- [ ] Large binaries live in Releases or LFS, not in the Git history.
- [ ] The repository URL is added to the matching CV bullet point.

---

## Starter prompt

Paste this into Claude Code or your agent of choice to begin.

<details>
<summary>Expand</summary>

```
You are helping me build a portfolio project from scratch. Read this brief and then
propose a concrete implementation plan before writing any code.

PROJECT: ml-grasp-rl-imitation-sim
Learned grasping and trajectory policies trained purely in simulation. Covers reward design, imitation learning from recorded demonstrations, domain randomisation and an ablation showing how randomisation affects the sim-to-real gap. Uses open simulation assets throughout.

TARGET STACK
- MuJoCo
- NVIDIA Isaac Sim or Isaac Lab
- PyTorch
- Stable Baselines3 or a custom RL implementation
- Demonstration recording tools
- Domain randomisation config

HARDWARE CONTEXT
- Training GPU
- Optional physical target: any 6-DoF arm with a gripper for sim-to-real transfer

THE SYSTEM MUST TAKE THESE INPUTS
- Simulation scene and asset files
- Reward and termination definitions
- Expert demonstration trajectories
- Randomisation ranges
- Training hyperparameters

AND PRODUCE THESE OUTPUTS
- Trained policy weights
- Training curves and metrics
- Evaluation success rate
- Rollout videos
- Exported policy for deployment

INTENDED REPOSITORY LAYOUT
envs/
  mujoco/                 grasp environments
  isaac/                  Isaac Lab task definitions
src/
  rewards/
  policies/
  train_rl.py
  train_il.py             behaviour cloning, DAgger
  evaluate.py
  randomisation/
demonstrations/           recorded expert trajectories
experiments/              training curves, ablation results
videos/                   rollout recordings
docs/

CONSTRAINTS
- Everything must run from a single command: `docker compose up` or `make demo`.
- No placeholder or stub code presented as working. If something is not implemented, say so.
- Every quantitative claim in the README must be reproducible from a script in this repo.
- Use only public or synthetic data. Record the licence of every dataset used.
- This is a clean-room reimplementation. Do not reproduce any proprietary design, protocol detail or dataset. Build the equivalent capability independently.

START BY
1. Confirming or challenging the proposed repository layout.
2. Listing the specific libraries and versions you intend to pin.
3. Identifying the single riskiest technical unknown and how to de-risk it first.
Then work through the milestones in the README build checklist, one at a time.

```

</details>

---

## Notes

- Do not backdate commits to match CV dates. Publishing older work today is normal and expected.
- Record dataset licences before training on anything. Several common research datasets are
  non-commercial only.
- Flip this repository to public once there is something in it worth showing.
