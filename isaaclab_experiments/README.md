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

Three task packages, all velocity-tracking on flat and rough terrain, trained with RSL-RL. Each
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

These are the baseline of the ablation below; the other variants of this package differ from them in
exactly one or two switches.

It is also set up to be deployed rather than only trained, which is what the settings
at the top of `rough_env_cfg.py` and `velocity_env_cfg.py` are about:

| | |
|---|---|
| `STAND_POSE`, `STAND_HEIGHT` | the policy is trained around the pose the robot is stood up in -- the thigh and calf angles of the built-in controller, 0.329 m of ride height -- and a reward holds that height while it walks, so the posture does not change hands at the hand-over |
| `OBSERVE_BASE_LIN_VEL` | off: a Go1 cannot measure its linear velocity, so the policy reads the 45 numbers it does have and the critic is given the true velocity instead (`obs_groups` in the agent config) |
| `ACTION_DELAY_STEPS` | the commanded targets reach the motors up to 10 ms late, drawn per robot, as they do over a real bus |
| `USE_SENSORS` | off: no height scanner, so the policy is blind to the terrain ahead, as the robot is |
| `FEET_AIR_TIME_WEIGHT` | the swing-time reward, 0.25. An option of the task, not of the ablation: 0 trains without it, and for one run `env.rewards.feet_air_time.weight=0.0` does the same without editing (`tools/slurm/submit.sh` asks) |
| `ACTUATOR_MODEL` | `"net"`: Isaac Lab's learned actuator network, fitted to a real Go1; `"pd"`: the ideal PD of `tools/sim2sim`, with randomized gains. The network damps joint velocities like `kd` 0.76 where the PD has 0.5, so the same policy moves its joints 20 - 25% faster on the PD |

Standing still is treated as a posture to hold rather than an absence of walking: the feet
all stay on the ground and the joints return to the nominal pose when there is nothing to
do, so the robot can be handed back to the controller from wherever it stops.

```bash
python isaaclab_experiments/train_rsl_rl.py --task IL4OP-Velocity-Rough-Unitree-Go1-v0 --headless
```

### Unitree Go1 with ContactNet: the ablation

**ContactNet** is a learned contact estimator (CNN and GRU over a 150-step window of 48 proprioceptive
channels) that returns a contact probability per foot, in the order `LF, RF, LH, RH`. Two proposals use
it, and each can be switched on or off independently:

| | |
|---|---|
| **CEP**, contact probability estimation | the four probabilities are **observed** by the policy (45 -> 49 inputs) |
| **CCR**, contact consistency reward | the policy is **rewarded** for agreeing with the simulated contacts: `1 - ¼ Σ|ĉᵢ - cᵢ|`, weight `0.05` |

Four tasks per terrain follow, all in `go1_locomotion/`:

| Variant | CEP | CCR | Task (`Rough`; `Flat` for flat terrain) | Policy inputs |
|---|:-:|:-:|---|:-:|
| baseline | | | `IL4OP-Velocity-Rough-Unitree-Go1-v0` | 45 |
| CEP | yes | | `IL4OP-Velocity-Rough-Unitree-Go1-CEP-v0` | 49 |
| CCR | | yes | `IL4OP-Velocity-Rough-Unitree-Go1-CCR-v0` | 45 |
| contact framework | yes | yes | `IL4OP-Velocity-Rough-Unitree-Go1-ContactFramework-v0` | 49 |

Each has a `-Play-v0` twin. The two switches are `cep` and `ccr` on `UnitreeGo1RoughEnvCfg`, and the
variants are subclasses that set them (`UnitreeGo1CEPRoughEnvCfg`, ...), which is how the PPO runners
`UnitreeGo1RoughPPORunnerCfg`, `UnitreeGo1CEPRoughPPORunnerCfg`, ... are laid out too. **Everything
else is shared** -- every reward weight, the randomization, the network and the PPO settings -- so a
difference between two variants is a difference in the switches and nothing else. Run the whole
ablation with one loop:

```bash
for variant in "" "-CEP" "-CCR" "-ContactFramework"; do
    python isaaclab_experiments/train_rsl_rl.py --task IL4OP-Velocity-Rough-Unitree-Go1${variant}-v0 --headless
done
```

Runs are logged under `logs/rsl_rl/unitree_go1_<variant>_<terrain>` (`unitree_go1_rough`,
`unitree_go1_cep_rough`, `unitree_go1_ccr_rough`, `unitree_go1_contact_framework_rough`), and each
run's `params/env.yaml` records the switches. A policy that observes the estimate (CEP, contact
framework) is exported and run in `tools/sim2sim` with `--obs_dim 49`; one that does not (baseline,
CCR) is the same 45-input policy -- CCR changes how it is trained, not what it reads, so it deploys
exactly like the baseline. `go1_locomotion/mdp/` holds the terms (`contact_probabilities`,
`contact_consistency`), and `contact_estimator.py` the estimator they share, which runs the network
once per step however many of them are on. `cnet.py` and `CNET.ckpt` are the network and its weights.

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
