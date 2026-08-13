# Sim-to-real, and what this repository can honestly say about it

## There is no robot here

Nothing in this repository has touched hardware. Every number comes from
MuJoCo. The brief asks for a "success rate in simulation against reality" and a
"sim-to-real gap analysis", and the honest version of that, without a robot, is
this:

> The `shifted` distribution is a **proxy** for a real arm. It is a set of
> worlds whose dynamics, actuation and sensing lie outside every training
> distribution, chosen to look like the ways a real gripper differs from its
> model. The train-to-shifted gap is a lower bound on a real sim-to-real gap,
> and it is not a substitute for measuring one.

Everywhere a number is quoted against `shifted`, it means that proxy. It is
never called a real-robot number.

## What the proxy contains, and why each piece

| Shift | Range | The real-world thing it stands for |
| --- | --- | --- |
| Box mass | 3.2–4.0× nominal | the part in the bin is not the part in the CAD |
| Box friction | 0.50–0.62× | surface finish, oil, dust, a different plastic |
| Table friction | 0.40–0.60× | ditto for the fixture |
| Gripper gain | 0.36–0.46× | a real gripper's force is lower and less repeatable than its datasheet |
| Wrist compliance | 2.5–3.1× | tool-plate flex and the compliance of any real mount |
| Command latency | 3–4 steps (120–160 ms) | fieldbus round trip, controller queue, driver buffering |
| Position noise | 7.5–9.5 mm | a camera pose estimate, not a simulator's ground truth |
| Velocity noise | 45–60 mm/s | differentiated pose estimates are noisy |
| Action noise | 0.085–0.105 | servo tracking error |

The three that matter most on real hardware, in the order they usually bite,
are latency, pose noise and gripper force. Simulation gives all three for free
and exactly, which is precisely why a policy trained without randomisation
learns to depend on them.

## What the proxy is missing

This is the part that keeps the claim honest.

* **No arm.** The MuJoCo hand floats. There are no joint limits, no
  self-collision, no arm inertia, no singularities and no reachability
  constraint. A real cell fails at all of these, and the Isaac port exists
  partly to have somewhere to put a real arm.
* **No perception.** The policy is handed the object pose. On a real cell that
  pose comes from a camera, and its error is neither Gaussian nor independent
  between steps — it is correlated, biased by viewpoint, and worst exactly when
  the gripper occludes the object. Additive Gaussian noise is a weak model of
  that failure.
* **No contact-model error.** Randomising friction coefficients inside a soft
  contact model does not simulate having the *wrong contact model*. Real
  fingertip contact involves deformation, rolling and stick-slip that MuJoCo's
  formulation does not attempt.
* **No calibration error.** The hand-eye transform is exact here. On hardware it
  is the single most common source of a systematic offset, and a systematic
  offset is a different failure from noise.
* **No wear, drift or temperature.** Nothing changes between episodes except
  what the randomiser draws.

## What to do with a real arm, if there were one

The order matters, and it is the order that makes each step falsifiable:

1. **Measure the plant first.** Step-response the gripper, measure real command
   latency, characterise the pose estimator's error against a fiducial. This
   turns randomisation ranges from guesses into measurements — the single
   biggest improvement available to this repository.
2. **Re-centre, do not just widen.** If the measured latency is 90 ms, centre
   the range there rather than widening around zero. Wide randomisation around
   the wrong centre buys robustness in directions that do not exist and costs
   performance in the one that does.
3. **Run the scripted expert on hardware first.** It reads the same observation
   vector and needs no training. If it cannot grasp, the transfer problem is in
   the plant model or the calibration, not in the policy.
4. **Then the policy, with a reachability check in front of it.** The action is
   a Cartesian delta, so a real deployment needs joint limits and a workspace
   check between the policy and the servo loop.
5. **Report the real number with the same protocol.** Same episode count, same
   success definition, same intervals across seeds.

Until step 5 has been done, the only defensible claim is the one at the top of
this document: a proxy gap, measured carefully, on a simulator.
