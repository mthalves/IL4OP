# Sim-to-sim validation

A policy that works in Isaac Lab has only been shown to work in Isaac Lab. This harness runs
the same policy in **MuJoCo**, on a model built from the robot's own URDF, and puts it through
the sequence a real deployment goes through: the robot starts folded on the floor, the
fixed-gain controller stands it up, and the legs are handed over only when asked.

What is deliberately different here is the physics engine, the actuator model and the contact
solver. That is the point: a policy that survives the change is one step closer to the robot,
and one that does not has been caught before it reached the hardware.

```bash
# scripted: one command, one report
python tools/sim2sim/play.py --robot go1 \
    --policy logs/rsl_rl/unitree_go1_rough/<run>/exported/policy.pt \
    --command 1.0 0 0 --duration 10

# drive it yourself, as you would the real robot
python tools/sim2sim/play.py --robot go1 --teleop --control ramp \
    --policy logs/rsl_rl/unitree_go1_rough/<run>/exported/policy.pt
```

## Before the first run

Three things are needed on top of the IL4OP environment: MuJoCo, a robot description, and an
exported policy.

**1. MuJoCo.** It is not among the requirements of the project, as nothing else here uses it:

```bash
pip install "mujoco>=3.1"          # developed against 3.14
```

**2. A robot description.** The URDFs are not part of this repository, and nor are the MuJoCo
scenes built from them -- a scene carries the absolute mesh paths of the machine that built it,
so every clone builds its own.

| Robot | Where its description comes from |
|---|---|
| Go1 | [`unitree_ros`](https://github.com/unitreerobotics/unitree_ros): `git clone https://github.com/unitreerobotics/unitree_ros.git ~/unitree_ros` |
| Go2W | [`robot_lab`](https://github.com/fan-ziqi/robot_lab) `v2.3.2`, the same clone the Go2W Isaac Lab tasks need |

`unitree_ros` is looked for at `~/unitree_ros`, at `~/unitree_rl_lab/unitree_ros`, beside this
repository and inside it; `$UNITREE_ROS` points at a clone kept anywhere else. Then:

```bash
python tools/sim2sim/build_models.py        # writes tools/sim2sim/models/*_scene.xml
```

Whichever description is present is built and the other is reported and skipped, so one robot
is enough to get started.

The builder adds what a URDF cannot say: the floating base, the floor, one torque actuator per
joint, the 1 kHz timestep, and the rotor inertia of the motors -- without that last one the
stiff gains of the stand-up controller are unstable at 1 kHz, which hardware is not. Each joint
keeps the torque limit its URDF gives it, so a Go1 knee gets the 35.55 Nm its belt stage buys it
rather than the 23.7 Nm of the motor alone.

**3. An exported policy**, if a policy is to be run at all -- driving the robot and walking it
need none. The harness loads the TorchScript file, not a training checkpoint:

```bash
python isaaclab_experiments/play_rsl_rl.py --task IL4OP-Velocity-Rough-Unitree-Go1-v0 \
    --checkpoint logs/rsl_rl/unitree_go1_rough/<run>/model_<n>.pt \
    --export --headless --max_steps 2
# -> logs/rsl_rl/unitree_go1_rough/<run>/exported/policy.pt
```

Pass `--obs_dim` when the policy is not the plain 48-input one: `52` for the Go1
contact-framework policies, `57` / `60` / `61` for Go2W flat / rough / flat-z.

Nothing else is needed: the harness runs on the `torch` and `numpy` the project already has, and
needs a display only for the viewer. The Go1 contact-framework policies also read `cnet.py` and
its checkpoint from `isaaclab_experiments/go1_locomotion_contact_framework/`, which are in the
repository.

## Driving it

`--teleop` opens the viewer and reads the keyboard **from the terminal**, so keep the terminal
focused: the viewer binds nearly every letter to its own render toggles, and a key pressed in
its window flips wireframe or shadows instead of moving the robot. `--keys viewer` reverts to
reading the viewer window if you prefer.

| key | |
|---|---|
| `1` | damping -- the motors are released and the robot flops down, as a real one does |
| `5` | hold the folded pose it starts in |
| `2` | stand up on fixed gains, like the built-in controller |
| `3` | hand the legs to the trained policy, and press again to take them back |
| `6` | move around **without a policy**, as pressing start on the robot does |
| `4` | fold the legs back down |
| `0` | put the robot back on the floor |
| `X`, `esc` | quit |

While the controller holds the stand (state `2`), the keys move the **trunk** over its planted
feet -- `W`/`S` fore and aft, `A`/`D` sideways, `R`/`F` up and down, `Y`/`H` roll, `T`/`G`
pitch, `Q`/`E` yaw, `space` back to the standing pose. Once the legs are walking (state `3` or
`6`) the same keys ask for a velocity instead. `--control` says how they do it:

| | |
|---|---|
| `sticky` (default) | each press adds a step, and the command stays until it is changed |
| `hold` | a key commands a fixed speed while it is held, zero when it is let go |
| `ramp` | the speed grows while the key is held and falls back when it is let go, like a stick |

`hold` and `ramp` tell a held key from a released one by its auto-repeat, so a short tap is a
short pulse.

Asking for the policy squares the robot up first: if the trunk is leaning, the controller
returns it to the standing pose and hands the legs over only once it is there, because that is
the pose the policy was trained around.

## Walking without a policy

State `6` is locomotion that needs no policy at all, so a robot can be driven around before
any policy exists -- and so a policy has something to be compared against.

It is **not** Unitree's controller. Unitree does not ship one: `unitree_sdk2` contains the
client that asks for it (`sport_client.Move(vx, vy, vyaw)`, `SwitchGait(1)`) and the service
that answers runs closed on the robot, while `unitree_ros/unitree_controller` holds only the
posture demos. So the gait here is written for this harness, and it is the classical one.

- **Go1** trots: the diagonal pairs of legs take turns, a swinging foot is thrown to where the
  commanded velocity says it should land, and the standing feet are drawn back underneath the
  robot, which is what carries the body forwards. The whole foot pattern is turned with the
  measured tilt of the trunk, so the standing legs push it back towards level.
- **Go2W** drives: a wheeled-legged robot does not trot to get around, so the legs hold the
  stand and the wheels are given differential-drive speeds. It does not strafe, as the wheels
  do not steer, and turning scrubs four of them sideways, which its wheel motors feel.

Both watch what the robot actually did and trim the command by what it fell short of. Measured
on flat ground: Go1 tracks 0.2-0.6 m/s forwards, 0.3 m/s sideways and 0.8 rad/s to within
0.01, leaning no more than 0.8°, with no motor at its limit; Go2W drives straight to within
0.01 m/s and turns at about 0.3 rad/s before the wheels run out of torque. Neither has any
force control and neither looks at the ground, so this is a flat-floor gait at moderate speed
and nothing more.

## What it checks, and what it does not

The scripted run reports whether the robot stayed up, how well it tracked the command, how far
it travelled, and the mean and peak torque -- the last of these is what says whether a policy
is asking the hardware for something it cannot give.

Known limits, so the results are read for what they are:

- the floor is flat, so a rough-terrain policy is only partly exercised;
- `base_lin_vel` is taken from the simulator, which the real robot has to estimate;
- the Go1 contact-framework policies run ContactNet on the simulated state, with the
  normalization constants the paper does not publish left at zero mean and unit variance.

## The code

| | |
|---|---|
| `play.py` | the command line and the scripted run |
| `build_models.py` | URDF to MuJoCo scene |
| `robot/specs.py` | the two robots: joints, poses, gains, observations, limits |
| `robot/simulation.py` | the model, the motors, and the observation a policy reads |
| `robot/kinematics.py` | where a leg can reach, and which joint angles reach there |
| `robot/controller.py` | the states the robot goes through and the poses it holds |
| `robot/gait.py` | getting around without a policy |
| `robot/contact.py` | ContactNet, for the policies that observe it |
| `robot/teleop.py` | the key map and what each key does |
| `robot/keyboard.py` | reading the terminal, writing to it |
| `robot/session.py` | a driving session: robot, viewer, keyboard, clock |
