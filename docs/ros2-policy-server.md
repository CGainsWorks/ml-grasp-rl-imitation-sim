# ROS 2 policy server

`ros2/grasp_policy_server` is a small colcon package (`ament_python`, ROS 2
Humble) that loads a policy exported by `src/export_policy.py` and serves its
actions over ROS 2 topics. It exists so that a learned grasp policy from this
repository can be offered to a ROS robot cell as an alternative to a planner,
for example the UR5e cell in `CGainsWorks/ros2-industrial-arm-cell-sim`.

**It has not been connected to that cell, or to any robot.** What is tested is
the node itself: it builds with colcon, loads the committed TorchScript policy
and, fed a recorded stream of observations from one simulated MuJoCo episode,
returns the same actions the policy produced in simulation. The adapter that
would build this observation from a real cell's joint states, TF and
perception and turn the action into motion, does not exist yet; what it would
have to do is set out at the end of this page.

## Nodes

| executable | role |
| --- | --- |
| `policy_server` | subscribes to observations, publishes one action per accepted observation |
| `observation_replay` | publishes a recorded stream (JSON) at a fixed rate, then exits |

```bash
ros2 launch grasp_policy_server policy_server.launch.py \
    policy_path:=experiments/policies/bcrl_medium_s0.ts.pt
# optionally, replay the recorded episode into it:
ros2 launch grasp_policy_server policy_server.launch.py \
    policy_path:=experiments/policies/bcrl_medium_s0.ts.pt \
    stream_path:=ros2/grasp_policy_server/test/data/replay_bcrl_medium_s0.json
```

## Interface

| topic | type | direction |
| --- | --- | --- |
| `grasp_policy/observation` | `std_msgs/Float32MultiArray` | in, 32 values |
| `grasp_policy/action` | `std_msgs/Float32MultiArray` | out, 4 values |

QoS on both: reliable, keep last 10. The names are relative, so a launch file
can put them under a namespace or remap them.

Parameters of `policy_server`:

| parameter | default | meaning |
| --- | --- | --- |
| `policy_path` | (required) | `<name>.ts.pt` written by `src/export_policy.py` |
| `meta_path` | next to the policy | `<name>.meta.json`, which declares the layouts below |
| `num_threads` | 1 | torch intra-op threads |

### Observation (32 values, world frame, SI units)

The layout is read from the policy's `.meta.json` and checked against the
network at load time; this table is that file's content for the committed
policies.

| indices | name | meaning |
| --- | --- | --- |
| 0-2 | `grip_pos` | grip site position, metres |
| 3-5 | `grip_vel` | grip site linear velocity, m/s |
| 6 | `gripper_width` | distance between the finger pads, metres |
| 7 | `gripper_width_rate` | rate of change of that width, m/s |
| 8-10 | `object_pos` | object position, metres |
| 11-13 | `object_rel_pos` | object position minus grip position |
| 14-19 | `object_rot6` | first two columns of the object rotation matrix |
| 20-22 | `object_linvel` | object linear velocity, m/s |
| 23-25 | `object_angvel` | object angular velocity, rad/s |
| 26-28 | `goal_pos` | hold point, metres |
| 29-31 | `goal_rel_pos` | hold point minus object position |

### Action (4 values, each in [-1, 1])

| indices | name | meaning |
| --- | --- | --- |
| 0-2 | `delta_xyz` | Cartesian displacement of the grip site, multiplied by 0.02 m per step |
| 3 | `gripper` | -1 fully open, +1 fully closed |

The policy was trained at a 0.04 s control period (25 Hz), so one action is a
displacement of at most 0.02 m per axis per 0.04 s.

### Failure behaviour

* An observation of the wrong length, or with a NaN or infinity in it, is
  rejected with a warning and produces **no** action. A consumer that times out
  on a missing action therefore stops rather than acting on a guess.
* A missing policy file, missing metadata, or metadata whose widths disagree
  with the network stops the node at start-up with the reason.
* Actions are clipped to the declared range.

## Tests

| where | what |
| --- | --- |
| `ros2/grasp_policy_server/test/test_core.py` | the policy wrapper without ROS: loading, width and finiteness checks, clipping, the recorded stream |
| `ros2/grasp_policy_server/test/test_policy_server_launch.py` | `launch_testing`: starts both nodes, replays 100 recorded observations and requires 100 actions matching the simulated episode to 1e-5, then sends a 31-value observation and requires a rejection and no action, then requires both processes to exit cleanly |
| `tests/test_ros_fixture.py` | in the main suite: the recording names the committed policy by hash, matches the environment's observation width and starts where the environment starts that seed today |

The recorded stream is `ros2/grasp_policy_server/test/data/replay_bcrl_medium_s0.json`,
one simulated MuJoCo episode (level `none`, seed 900000, the first of the
final-evaluation block) driven in closed loop by `bcrl_medium_s0`, which
succeeds. `make stream` regenerates it. CI runs the package tests in
`ros:humble-ros-base-jammy`; `make ros-test` runs the same steps locally in
Docker.

## What connecting it to the arm cell would take

Not done and not small. The policy was trained on a parallel-jaw hand on a
mocap weld in MuJoCo, not on a UR5e, so before any of the plumbing below the
honest expectation is the cross-simulator result in this repository: policies
that do not transfer between engines without retraining (README, "Policies do
not transfer between the simulators").

An adapter node in the cell would have to:

1. Build the observation at 25 Hz: grip position and velocity from TF of the
   tool frame, pad width and its rate from the gripper joint states, object
   pose and velocity from the cell's perception (its grasp planner publishes
   candidate poses, not a tracked object state, so a tracker is missing) and
   the hold point from the cell's task definition, all expressed in the frame
   the policy was trained in.
2. Turn each action into motion through something that enforces reachability,
   joint limits, collisions and the cell's protective stop, for example the
   cell's straight-line `linear_pose` target on its trajectory executor, with
   the displacement accumulated onto the current tool pose.
3. Treat a missing action as a stop.

None of that exists here or in the cell. No number in this repository
comes from a robot.
