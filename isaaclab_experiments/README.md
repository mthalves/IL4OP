# Environments and scripts

Everything IL4OP adds on top of Isaac Lab lives here: the environments, and the scripts that
train, play and evaluate in them. Importing `isaaclab_experiments` registers every task in
gymnasium, so any task below can be passed to the scripts by name. Tasks from Isaac Lab itself
work in exactly the same way, which is how the navigation policy is trained.

> Start the scripts from the repository root -- `python isaaclab_experiments/<script>.py` --
> because they import `utils.launcher` relative to that directory.

## Scripts

| Script | What it does |
|---|---|
| `train_rsl_rl.py` | train a task with RSL-RL |
| `train_skrl.py` | train a task with skrl (torch or jax) |
| `play_rsl_rl.py` | replay a trained policy, report episode statistics, `--export` it to TorchScript/ONNX |
| `play_skrl.py` | the same for skrl agents |
| `planning.py` | run one planning experiment in the Anymal-C planning environment |
| `run_planning_experiments.py` | a batch of planning experiments with screen recording |

Common options: `--headless`, `--num_envs`, `--max_iterations`, `--seed`; for the play scripts
`--checkpoint` (otherwise the latest run of the task), `--max_steps`, `--video`, `--real_time`.

## Planning under uncertainty

`anymal_c_planning/` is the environment the planning algorithms are measured in: an Anymal-C on
a terrain with obstacles, driven by a high-level navigation policy, with the planner choosing
where to go. **It evaluates planners; it does not train anything.**

| Task | Decision and world model |
|---|---|
| `Anymal-C-Planning-v0` | discrete -- `astar`, `despot`, `ibpomcp`, `pomcp`, `tbrhopomcp` |
| `Continuous-Anymal-C-Planning-v0` | continuous -- `pomcpdpw`, `pomcpow`, `pftdpw` |

The planner and its parameters are chosen in `anymal_c_planning/agents/planning_cfg.py`; the
environment itself is assembled from the configuration modules in `anymal_c_planning/configs/`
(scene, terrain, observations, actions, rewards, events, terminations).

Two policies drive the robot, and neither is trained by these tasks:

- the **high-level navigation** policy, `policies/anymal_c_navigation.jit.pt`, which turns a
  goal into a velocity command;
- the **low-level locomotion** policy, Isaac Lab's blind Anymal-C policy, pulled from the asset
  server at run time.

To retrain the navigation policy, train Isaac Lab's navigation task and convert the result to
the TorchScript file this environment loads:

```bash
python isaaclab_experiments/train_skrl.py --task Isaac-Navigation-Flat-Anymal-C-v0 --headless
python isaaclab_experiments/policies/convert_pt2jit.py --headless \
    --task Isaac-Navigation-Flat-Anymal-C-v0 \
    --checkpoint logs/skrl/<run>/checkpoints/best_agent.pt
```

The converter traces the skrl PPO policy, writes `policy.jit.pt` in the working directory and
checks its output against the original agent. Copy it over
`policies/anymal_c_navigation.jit.pt` to put it to use; its observation layout has to match
what the planning environment feeds it.

## Legged locomotion

Four task packages, all velocity-tracking on flat and rough terrain, trained with RSL-RL. Each
one also registers a `-Play-v0` variant with fewer environments and no randomization.

### Unitree Go1

`go1_locomotion/` is a standalone copy of Isaac Lab's Go1 velocity task -- base configuration,
MDP terms and agent configs -- so it can be changed without touching the vendored Isaac Lab.
Beyond the original it adds gait-quality and hardware-safety terms: a minimum stance width, a
swing-time reward that pays for lifting a foot, penalties on joint-velocity and torque limits,
and a tip-over termination.

| Task | |
|---|---|
| `IL4OP-Velocity-Flat-Unitree-Go1-v0` | flat terrain |
| `IL4OP-Velocity-Rough-Unitree-Go1-v0` | rough terrain with curriculum |

```bash
python isaaclab_experiments/train_rsl_rl.py --task IL4OP-Velocity-Rough-Unitree-Go1-v0 --headless
```

### Unitree Go1 with contact estimation

Two packages build on the Go1 task with **ContactNet**, a learned contact estimator (CNN and
GRU over a 150-step window of 48 proprioceptive channels) that returns a contact probability
per foot, in the order `LF, RF, LH, RH`.

| Package | Task prefix | What it adds |
|---|---|---|
| `go1_locomotion_contact_consistency/` | `...-Go1-ContactConsistency-v0` | a reward for agreeing with the estimator: `1 - ¼ Σ|ĉᵢ - cᵢ|`, weight `0.05` |
| `go1_locomotion_contact_framework/` | `...-Go1-ContactFramework-v0` | the same reward, **and** the four probabilities as policy observations (48 -> 52 inputs) |

Both come in `Flat` and `Rough` variants, and both carry their own copy of `cnet.py` and the
checkpoint, so a task is self-contained.

### Unitree Go2W

`go2w_locomotion/` needs [robot_lab](https://github.com/fan-ziqi/robot_lab) (`v2.3.2`) for the
Go2W description and the base wheeled-legged task. Without it these tasks are simply not
registered, and the scripts say so and carry on.

```bash
git clone --branch v2.3.2 https://github.com/fan-ziqi/robot_lab.git
pip install -e robot_lab/source/robot_lab --config-settings editable_mode=compat
```

| Task | |
|---|---|
| `IL4OP-Velocity-Flat-Unitree-Go2W-v0` | flat terrain |
| `IL4OP-Velocity-Rough-Unitree-Go2W-v0` | rough terrain with curriculum |
| `IL4OP-Velocity-Flat-Z-Unitree-Go2W-v0` | flat terrain with a commanded base height, 0.25 - 0.40 m |

The `Flat-Z` variant adds the `base_height` command to the observations and the
`base_height_penalty`, `go2w_joint_mirror` and `wheel_position_penalty` rewards (see
`go2w_locomotion/mdp/`).

## Adding a task

Put the package in this folder and register its tasks in its own `__init__.py`, then import it
from [`__init__.py`](__init__.py) so that importing `isaaclab_experiments` registers it. Keep a
dependency that is not part of this repository behind a `try/except ImportError`, as the Go2W
tasks are, so a clone without it still works.

## Afterwards

A trained locomotion policy should be checked in a second simulator before it goes near the
hardware: export it with `play_rsl_rl.py --export` and run it through
[`tools/sim2sim`](../tools/sim2sim/README.md).
