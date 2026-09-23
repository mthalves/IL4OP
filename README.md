# IL4OP  
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/) [![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-green)](https://isaac-sim.github.io/IsaacLab/)

IL4OP is a modified and extended version of **IsaacLab** designed to support **online planning under uncertainty** in robotic environments. It adapts IsaacLab's flexibility to advance research in online planning, with ready-to-use components for benchmarking, testing, and experimentation.  

Publicly available to foster research! :sparkles: 
*Cite us if you use IL4OP in your work* :pray:

---

## :rocket: Main Features
- Integrated framework over **IsaacLab**;
- Support for **online planning under uncertainty**; 
- Plug-and-play **planning algorithm selection**;
- Easy experiment configuration and logging (**results + videos**);
- Open-source framework to facilitate **implementation, testing, and benchmarking**. 

---

## :gear: Installation

### Requirements
- Linux (tested on Ubuntu 22.04), an NVIDIA GPU and a driver supporting **CUDA 12.8**;
- [Miniconda](https://docs.conda.io/projects/miniconda/) (or any Python **3.11** environment) — `setup.sh` checks it and can install it with `--install-conda`;
- ~40 GB of free disk space for Isaac Sim and its asset cache.

### Quick setup
```bash
git clone git@github.com:mthalves/IL4OP.git && cd IL4OP
./setup.sh                  # add --with-robot-lab to also enable the Go2W tasks
conda activate IL4OP
```
The script verifies the conda installation, creates the `IL4OP` environment and installs
PyTorch `2.7.0+cu128`, IsaacSim `5.1.0`, the **vendored** IsaacLab `2.3.2` and this package,
checking along the way that the packages land in a Python 3.11 environment. Useful flags:

| Flag | Effect |
|---|---|
| `--with-robot-lab` | also clone and install `robot_lab` for the Go2W tasks |
| `--install-conda` | download and install Miniconda when conda is missing |
| `--use-current-env` | install into the environment that is already active |
| `--env NAME` | use another environment name (default: `IL4OP`) |
| `--dry-run` | print the commands without running them |

### Manual setup
<details>
<summary>The same steps, one by one</summary>

```bash
# 1. environment
conda create -n IL4OP python=3.11 -y && conda activate IL4OP
pip install --upgrade pip

# 2. PyTorch (CUDA 12.8), installed first so that pip keeps this exact build
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Isaac Sim
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com

# 4. the vendored IsaacLab 2.3.2 -- install it from this repository, never from pip
for ext in isaaclab isaaclab_assets isaaclab_contrib isaaclab_mimic isaaclab_rl isaaclab_tasks; do
    pip install -e IsaacLab/source/$ext
done

# 5. this project
pip install -r requirements.txt
pip install -e .
```
</details>

### Go2W tasks (optional)
The Unitree Go2W tasks build on [robot_lab](https://github.com/fan-ziqi/robot_lab), which is
not part of this repository:
```bash
git clone --branch v2.3.2 https://github.com/fan-ziqi/robot_lab.git
pip install -e robot_lab/source/robot_lab --no-deps --config-settings editable_mode=compat
```
`editable_mode=compat` is required: with the default editable install, `import robot_lab`
resolves to the empty `robot_lab/` directory of this repository instead of the package.

### Verify
```bash
python tools/check_environment.py
```
Then run a two-iteration training as an end-to-end test:
```bash
python isaaclab_experiments/train_rsl_rl.py --task IL4OP-Velocity-Flat-Unitree-Go1-v0 \
    --num_envs 32 --max_iterations 2 --headless
```
:warning: The **first** Isaac Sim start downloads shader and asset caches and can take
10-20 minutes without printing anything. This is expected, not a hang.

### What a fresh clone does not contain
| Not in the clone | Consequence |
|---|---|
| `logs/` | Training runs and checkpoints are ignored by git; copy them manually to evaluate or resume a policy |
| `robot_lab/` | The Go2W tasks stay unavailable until it is installed (the scripts warn and continue) |
| `outputs/` | Hydra run directories, not needed |

The pretrained Anymal-C navigation policies **are** included, so the planning experiments
work right after the installation.

### Troubleshooting
- `Could not load PyInstaller's embedded PKG archive ... (/root/miniconda3/_conda)` -> conda was
  installed as **root** (with `sudo`) and your user cannot read it, or the installer download was
  truncated. Remove it (`sudo rm -rf /root/miniconda3`), drop any `conda init` block that points
  there from your `~/.bashrc`, and run `./setup.sh --install-conda` **without sudo**;
- `conda: command not found` after `--install-conda` -> the batch installer does not touch the
  shell configuration; the script runs `conda init` for you, so open a new terminal (`exec $SHELL`);
- `CondaToSNonInteractiveError` (Terms of Service of the Anaconda channels not accepted) ->
  `setup.sh` creates the environment from `conda-forge` only and is not affected; if another
  conda command raises it, either accept the terms with `conda tos accept --override-channels
  --channel https://repo.anaconda.com/pkgs/main` (and the same for `.../pkgs/r`) or create the
  environment with `-c conda-forge --override-channels` and use `--use-current-env`;
- `ModuleNotFoundError: pkg_resources` when building `flatdict` -> `flatdict` has no wheel and
  its `setup.py` imports `pkg_resources`, which setuptools 82 removed; pip builds it in an
  isolated environment with the newest setuptools, so it has to be built against an older one
  (`setup.sh` does this automatically):
  ```bash
  pip install "setuptools<82" wheel
  pip install flatdict==4.0.1 --no-build-isolation
  ```
- the training and play scripts import `utils.launcher`, so they must be started from the
  repository root as `python isaaclab_experiments/<script>.py`.

## :arrow_forward: Running Experiments

### 1. Configure your agent/planning algorithm:
- Open `isaaclab_experiments/anymal_c_planning/agents/planning_cfg.py`;
- You will see the catalogue of available planners and their parameters:

  ```python
   PLANNER_CFG = {
      "astar": {},
      "despot": {
         "max_depth": 20,
         "max_it": 1000,
         "discount_factor": 0.95,
         "num_scenarios": 100,
         "lambda_reg": 0.005,
      },
      "ibpomcp": {
         "max_depth": 20,
         "max_it": 1000,
         "q": 0.2,
         "discount_factor": 0.95,
         "particle_revigoration": True,
         "k": 100,
      },
      ...
   }
  ```
- Select one key (method name), change the planning algorithm parameters as needed and have fun testing it! :smile:

  ```python
   # Select the planning method here
   METHOD = "ibpomcp"
  ```
- Discrete planners (`astar`, `despot`, `ibpomcp`, `pomcp`, `tbrhopomcp`) run with `--space discrete`;
  continuous planners (`pomcpdpw`, `pomcpow`, `pftdpw`) run with `--space continuous`.

### 2. Run your experiment:
- With the graphical launcher (recommended for single experiments):
   ```bash
   python -m app
   ```
   Pick the problem and planner, adjust the parameters and run options, and start the simulation.
   The simulator output is shown in the window and the selection is exported to
   `planning_cfg.local.json` without touching `planning_cfg.py`.
- From the command line:
   ```bash
   python isaaclab_experiments/planning.py --space discrete --log True
   ```
- For multiple experiments with screen recording:
   ```bash
   python isaaclab_experiments/run_planning_experiments.py
   ```

  - :warning: The script `run_planning_experiments.py` was tested on Ubuntu 22.04.5 LTS.
  - It has not been ported to other operating systems. Some additional requirements may still need to be installed.

### 3. Results are saved in `logs/inspection/`. This includes log files and videos for analysis.
- We let the necessary code for plotting and analysing the results ready and available in the `logs` directory. 
- Easy to run, easy to analyse. :kissing_smiling_eyes:

### 4. (Optional) Train the low-level navigation policy:
- A pretrained policy is shipped in `isaaclab_experiments/policies/`. To retrain it with RSL-RL or skrl:
   ```bash
   python isaaclab_experiments/train_rsl_rl.py --task Anymal-C-Planning-v0 --headless
   python isaaclab_experiments/train_skrl.py   --task Anymal-C-Planning-v0 --headless
   ```
- Checkpoints and TensorBoard logs are written to `logs/rsl_rl/` and `logs/skrl/`.
- To play and evaluate a trained policy, use the matching play script. It loads the latest run of the
  task (or `--checkpoint <file>`), runs the policy and reports episode statistics; `--max_steps` bounds the
  evaluation, `--video` records a rollout and `--export` (RSL-RL) writes JIT/ONNX policies next to the checkpoint:
   ```bash
   python isaaclab_experiments/play_rsl_rl.py --task Anymal-C-Planning-v0 --num_envs 32 --max_steps 1100 --headless
   python isaaclab_experiments/play_skrl.py   --task Anymal-C-Planning-v0 --checkpoint <path/to/agent.pt>
   ```

### 5. (Optional) Train the Unitree Go1 locomotion policies:
- `isaaclab_experiments/go1_locomotion/` is a standalone copy of IsaacLab's Go1 velocity task (base
  configuration, MDP terms and agents), so it can be modified without touching the vendored IsaacLab:

  | Task | Description |
  |---|---|
  | `IL4OP-Velocity-Flat-Unitree-Go1-v0` (`-Play-v0`) | flat terrain |
  | `IL4OP-Velocity-Rough-Unitree-Go1-v0` (`-Play-v0`) | rough terrain with curriculum |

   ```bash
   python isaaclab_experiments/train_rsl_rl.py --task IL4OP-Velocity-Rough-Unitree-Go1-v0 --headless
   ```

### 6. (Optional) Train the Unitree Go2W locomotion policies:
- These tasks build on [robot_lab](https://github.com/fan-ziqi/robot_lab) (`v2.3.2`), which provides the Go2W
  robot description and the base velocity-tracking task. Install it next to the repository:
   ```bash
   git clone --branch v2.3.2 https://github.com/fan-ziqi/robot_lab.git
   pip install -e robot_lab/source/robot_lab --config-settings editable_mode=compat
   ```
- Three variants are registered in `isaaclab_experiments/go2w_locomotion/`:

  | Task | Description |
  |---|---|
  | `IL4OP-Velocity-Flat-Unitree-Go2W-v0` | flat terrain |
  | `IL4OP-Velocity-Rough-Unitree-Go2W-v0` | rough terrain with curriculum |
  | `IL4OP-Velocity-Flat-Z-Unitree-Go2W-v0` | flat terrain with a **commanded base height** (0.25 - 0.40 m) |

   ```bash
   python isaaclab_experiments/train_rsl_rl.py --task IL4OP-Velocity-Flat-Z-Unitree-Go2W-v0 --headless
   ```
- The `Flat-Z` variant adds a `base_height` command to the observations and the reward terms
  `base_height_penalty`, `go2w_joint_mirror` and `wheel_position_penalty` (see `go2w_locomotion/mdp/`).


## :computer: In development & Future directions

- [x] Single agent planning using discrete world and decision models.
- [x] Support planning algorithms with continuous world and decision models.
- [ ] Extension of the single agent scenario to multi-agent problems (toilored to centralized and decentralized approaches).
- [ ] Extension of IL4OP to support dynamic world models applications.

## :book: Citation

If you use IL4OP in your research, please cite our work:

```bibtex
@misc{alves2025,
  **To appear**
}
```

