# Isaac Lab port

**Status: written and reviewed, never executed.**

There is no Isaac Sim installation on the machine this repository was built on.
`grasp_task.py` has never been run, no number in the top-level README comes from
it, and CI does not import it beyond the parity test described below. It is a
port waiting for its first bring-up, and it is listed here as such rather than
quietly implied to work.

## What is genuinely shared with the MuJoCo task

The reward, the success condition and the randomisation ranges are imported,
not copied:

| Shared thing | File |
| --- | --- |
| Reward terms and weights | `src/rewards/grasp_reward.py` |
| Success and drop conditions | `src/rewards/grasp_reward.py` |
| Randomisation ranges per level | `src/randomisation/configs/*.json` |

`grasp_reward` dispatches on the array library it is given, so the same code
scores one MuJoCo world with numpy and `num_envs` Isaac worlds with torch on the
GPU. `tests/test_reward_parity.py` runs both backends on identical inputs and
asserts they agree to single precision. That test runs in CI and does not need
Isaac; it is the one part of the port that can be verified without a simulator.

## What is deliberately different

| | MuJoCo | Isaac Lab |
| --- | --- | --- |
| Embodiment | free-floating parallel-jaw hand | Franka Panda arm and gripper |
| Cartesian control | mocap body plus a weld constraint | differential IK (damped least squares) |
| Vectorisation | one world | `num_envs` worlds, partial resets |
| Randomisation mechanism | model fields edited in place at reset | Isaac Lab event manager |
| Grasp test | contact list, both pads on the object | geometric proximity plus finger closure |

The last row is the weakest part of the port. A geometric test reports a grasp
for a pinch that is merely close, which is exactly the failure the MuJoCo
version avoids by reading contacts. Adding a `ContactSensor` to each finger in
`GraspSceneCfg` and switching `_grasped` to read it is the first thing to fix.

Because the control path differs, a policy trained in MuJoCo is **not** expected
to run in Isaac unchanged. Cross-simulator transfer is a separate experiment and
it has not been done.

## First bring-up, in order

1. `python -c "import isaaclab"` inside the Isaac Python environment. If the
   import path is `omni.isaac.lab`, this is Isaac Lab 1.x and the imports at the
   top of `grasp_task.py` need the older names.
2. Instantiate with `num_envs=2` and step it with zero actions for 100 steps.
   Nothing should explode, and the box should stay on the table.
3. Check the observation layout against the table in `envs/mujoco/grasp_env.py`.
   Index by index. A silently transposed rotation block will train to a
   plausible-looking mediocre policy and cost a day.
4. Replace `_grasped` with a contact sensor and confirm it agrees with the
   geometric test on obvious cases, and disagrees on near misses.
5. Drive it with the scripted expert from `src/policies/scripted_expert.py`. It
   reads the observation vector and nothing else, so it should transfer. If the
   expert cannot grasp, the environment is wrong, not the policy.
6. Only then train.

## Running it

```bash
# Inside the Isaac Sim python environment
${ISAACSIM_PATH}/python.sh scripts/isaac_train.py --task GraspLift-Direct-v0 --num_envs 4096
```

`scripts/isaac_train.py` does not exist in this repository: training on Isaac
would use Isaac Lab's own `train.py` with `rsl_rl` or `skrl`, and shipping a
launcher that has never been executed would be worse than shipping none.
