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

# the same battery of commands for every policy, under what a real Go1 differs by (see below)
python tools/sim2sim/evaluate.py \
    --policy logs/rsl_rl/unitree_go1_rough/<run>/exported/policy.pt

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

The harness reads the observation size from the policy and picks the layout to match: 45 numbers
are the Go1's own, 49 add the four ContactNet probabilities (the CEP and contact-framework tasks;
CCR and baseline policies read 45), 48 / 52 are the same for policies trained while the base
linear velocity was still observed, and Go2W flat / rough / flat-z read 57 / 60 / 61. A size it has
no layout for is reported, not guessed; `--obs_dim` is only there to assert what you expect.

A policy also has to be run with the nominal pose it was trained around, because that pose
is the zero of both its action and the joint positions it reads. `--nominal legacy` is the
Isaac Lab default pose that the Go1 policies trained before `STAND_POSE` used; without it
they stand 42 mm too low and the comparison means nothing.

Nothing else is needed: the harness runs on the `torch` and `numpy` the project already has, and
needs a display only for the viewer. The Go1 policies that observe the contact estimate also read
`cnet.py` and its checkpoint from `isaaclab_experiments/go1_locomotion/`, which are in the repository.

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

## Towards the real robot

Sim-to-sim is the step before the robot, not the last one. What this harness has settled:

- the policy runs at the rate, the gains and the torque limits the robot has, from the pose
  the controller stands it up in, and the hand-over in both directions is tested;
- the observation it reads is one the robot can produce -- 45 numbers, none of which is the
  base linear velocity a Go1 cannot measure;
- the peak torque a policy asks for is measured, and no policy here comes near the limits.

What is still missing before a policy drives the hardware:

| | |
|---|---|
| the low-level program | a Go1 speaks the legacy `unitree_legged_sdk` over UDP, not the DDS of `unitree_sdk2`, so the deployment side has to be written: read the 1 kHz state, build the observation, run the policy at 50 Hz, send position targets with kp 20 / kd 0.5 |
| the joint order | Isaac Lab orders the joints by type (all hips, all thighs, all calves), the SDK by leg; getting this wrong is the classic way to break a robot |
| calibration | the encoder zero of a real Go1 is not the zero of the URDF, and an offset of a few degrees is a different posture |
| safety | a torque and velocity clamp, a tip-over cut-out and a damping fallback on the deployment side, not in the policy |
| ContactNet on the robot | for the CEP and contact-framework policies: the 48 channels have to be rebuilt from what the robot measures, at 50 Hz and over the same 150-step window. The foot velocity is the velocity *relative to the trunk*, `ω × r + J(q) q̇` (gyro and leg Jacobian), not `J(q) q̇` alone, and the standardization is the one in `cnet.py` |
| the gap that is left | flat floor only here, no estimator in the loop, and the actuator network stands in for the motors |

## What it checks, and what it does not

The scripted run reports whether the robot stayed up, how well it tracked the command, how far
it travelled, and the mean and peak torque -- the last of these is what says whether a policy
is asking the hardware for something it cannot give.

Known limits, so the results are read for what they are:

- the floor is flat, so a rough-terrain policy is only partly exercised;
- `base_lin_vel` is taken from the simulator, which the real robot has to estimate;
- the motors are an ideal PD (`kp` 20, `kd` 0.5), while Isaac Lab trains the Go1 through a learned
  actuator network whose response to a velocity is that of `kd` = 0.76 -- measured on the network
  itself -- and whose torque falls off with joint speed. Left alone, the same policy moves its joints
  about 30% faster here at 1 m/s than it did in training; `--joint_damping 0.26` (N m s/rad, on top of
  `kd`) closes that gap, and is the setting to use when comparing against numbers from Isaac Lab;
- the Go1 policies that observe the contact estimate run ContactNet on the simulated state,
  standardized with statistics measured for it in `cnet.py` -- the paper does not publish its own.
  The estimator advances every control cycle, whoever has the legs, as it would on the robot.

## Isaac Lab and MuJoCo, side by side

What a Go1 policy is trained on and what this harness runs it on. The same policy was run in both,
and in Isaac Lab on an ideal PD as well, to tell the actuator apart from the rest; the actuator and
the mass were measured on their own, the other rows were not isolated, and the agreement below is
the only evidence about them.

| | Isaac Lab (training) | MuJoCo (this harness) | what it does to a policy |
|---|---|---|---|
| policy / control rate | 50 Hz, PD at 200 Hz (5 ms) | 50 Hz, PD at 1 kHz (1 ms) | not isolated |
| actuator | learned network: `kp` 19.6, `kd` 0.76, torque limit falling to 0 at 30 rad/s | ideal PD: `kp` 20, `kd` 0.5, flat limit 23.7 N m (35.55 at the knee) | **the one systematic gap**: +20 - 40% peak joint speed and joint jitter at 1 m/s; `--joint_damping 0.26` closes it |
| rotor inertia | 0.005 - 0.007 (hips, thighs), 0.011 - 0.017 (knees), random | 0.0059, 0.0141 | same |
| joint friction | 0 - 0.04, random | none | not isolated |
| robot mass | asset 13.10 kg, trunk -2 .. +2 kg | URDF 12.03 kg | about +6% torque and -4 mm of ride height per kg; no change in gait |
| action latency | 0 - 10 ms, random per robot | none | not isolated |
| ground friction | 0.4 - 1.2, random | 1.0 | not isolated |
| joint limits | 90% of the URDF's, soft | the URDF's, hard | not isolated |

For policies trained with the current rewards the two agree on ride height (about 1 cm), tilt,
velocity tracking, torque and action rate (about 15%); none fell in either. Two behaviours that looked like
sim-to-sim failures on the first Go1 policies were there in Isaac Lab as well, so they are the
policies' and not the harness': a foot held off the floor at a standstill, and a sagging stance.
The Go1 contact-framework policy trained before ContactNet's inputs were standardized also stands
low in both, because the estimate it reads is no longer the one it learned on.

## Evaluating a policy

`evaluate.py` runs the same seven commands (standing, forwards, backwards, sideways, turning, a mix)
under six conditions and reports the worst case of each: the harness as it is; `damped`, which
matches the actuator the policy was trained with and is the one to compare with Isaac Lab; and then
on top of `damped` a heavier trunk (+2 kg), a slippery floor (friction 0.4), encoders 0.03 rad off
the model's zero, and a command reaching the motors 20 ms late. Those are the four ways a real
Go1 most plausibly differs from the training, with the margin the randomization of the training
leaves (latency was trained up to 10 ms, calibration not at all). A condition passes if the robot
stays up through all the commands; the other columns -- tilt, tracking, torque against its limit,
peak joint speed, action rate, joint jitter -- are for comparing policies, and `--detail` splits
them by command. A policy that fails one of these is not ready for the robot, whatever it does
in the nominal case.

## The code

| | |
|---|---|
| `play.py` | the command line and the scripted run |
| `evaluate.py` | the battery of commands and conditions, for comparing policies |
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
