# IL4OP  
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/) [![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-green)](https://isaac-sim.github.io/IsaacLab/) [![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.2-green)](https://isaac-sim.github.io/IsaacLab/) [![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Ubuntu%20only-orange)](#requirements) [![License](https://img.shields.io/badge/license-GPL--3.0-lightgrey)](LICENSE)

> :penguin: **Linux only.** IL4OP is developed and tested on **Ubuntu** with an NVIDIA GPU, and nowhere else.
> Isaac Sim itself runs on Windows too, but nothing here has been tried there: `setup.sh`, `tools/slurm/` and the
> terminal controls of `tools/sim2sim` all assume a POSIX shell. Tested on Ubuntu 22.04 and 26.04; Isaac Sim 5.1
> also supports 24.04.

IL4OP is a modified and extended version of **IsaacLab** designed to support **online planning under uncertainty** in robotic environments. It adapts IsaacLab's flexibility to advance research in online planning, with ready-to-use components for benchmarking, testing, and experimentation.  

Publicly available to foster research! :sparkles: 
*Cite us if you use IL4OP in your work* :pray:

---

## :rocket: Main Features
- Integrated framework over **IsaacLab**;
- Support for **online planning under uncertainty**; 
- Plug-and-play **planning algorithm selection**;
- Easy experiment configuration and logging (**results + videos**);
- **Sim-to-sim validation** of trained locomotion policies in MuJoCo ([tools/sim2sim](tools/sim2sim/README.md));
- Open-source framework to facilitate **implementation, testing, and benchmarking**. 

---

## :gear: Installation

### Requirements
- **Ubuntu Linux** with an NVIDIA GPU; no other platform is tested. Isaac Sim 5.1 officially supports **Ubuntu 22.04/24.04** and lists Linux driver **580.65.06** as its tested driver version. See the [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
- This repository has also been validated on **Ubuntu 26.04 + RTX 5090 + NVIDIA 580-open** using the compatibility workaround below. Ubuntu 26.04 is **not officially supported** by Isaac Sim 5.1.
- [Miniconda](https://docs.conda.io/projects/miniconda/) (or any Python **3.11** environment) — `setup.sh` checks it and can install it with `--install-conda`;
- ~40 GB of free disk space for Isaac Sim and its asset cache.

### Quick setup
```bash
git clone git@github.com:mthalves/IL4OP.git && cd IL4OP
./setup.sh                  # add --with-robot-lab to also enable the Go2W tasks
conda activate IL4OP

# First Isaac Sim launch on RTX 5090:
isaacsim-il4op --reset-user

# Subsequent launches:
isaacsim-il4op
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
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

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

### Ubuntu 26.04 / RTX 5090 compatibility

Isaac Sim 5.1 officially supports Ubuntu 22.04/24.04. On Ubuntu 26.04,
Isaac Sim's bundled `libfbxsdk.so` expects the legacy **`libxml2.so.2`** ABI,
while the OS provides a newer libxml2 ABI. Installing `libxml2` with `apt` does
not resolve this ABI mismatch.

`setup.sh` automatically detects Ubuntu 26.04 and installs the conda-forge
build `libxml2=2.13.9=h04c0eec_0` in a temporary prefix. It then copies
`libxml2.so.2` and its required dependencies into Isaac Sim's
`asset_converter_native_bindings/libs/` directory. System libraries are left
untouched; do **not** symlink `libxml2.so.16` to `libxml2.so.2`.

This workaround follows the approach documented in
[Isaac Sim discussion #824](https://github.com/isaac-sim/IsaacSim/discussions/824).

For RTX 5090 / Blackwell, use the **NVIDIA 580-open** driver branch for this
Isaac Sim 5.1 setup. NVIDIA lists Linux driver **580.65.06** as the tested
5.1.0 baseline. This repository was successfully run on Ubuntu 26.04 with
Ubuntu's `nvidia-driver-580-open` package, version `580.178.04`.

The setup script creates an `isaacsim-il4op` launcher that passes:

```text
--/renderer/activeGpu=0
--/renderer/multiGpu/enabled=false
```

Use `isaacsim-il4op --reset-user` for the first launch, then `isaacsim-il4op`
for subsequent launches.

On Ubuntu 26.04, the ROS 2 bridge may also report that the Ubuntu version is
unsupported for automatic ROS distribution selection. This is separate from
the `libxml2` workaround; if ROS 2 is not being used, this warning can be
ignored.

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
| `unitree_ros/`, `tools/sim2sim/models/` | The sim-to-sim scenes are built from the robot descriptions on first use, by `tools/sim2sim/build_models.py` |
| `outputs/` | Hydra run directories, not needed |

The pretrained Anymal-C navigation policies **are** included, so the planning experiments
work right after the installation.

### Troubleshooting
- `ModuleNotFoundError: pkg_resources` while installing the IsaacLab extensions ->
  `pip install "setuptools<82"` and run the step again;
- `libxml2.so.2: cannot open shared object file` on Ubuntu 26.04 ->
  re-run `./setup.sh`; the compatibility library is installed automatically.
  Do **not** symlink `libxml2.so.16` to `libxml2.so.2`;
- Isaac Sim starts and then segfaults ->
  check `nvidia-smi` and use the **580-open** driver branch. Launch with
  `isaacsim-il4op --reset-user`;
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

### 4. (Optional) Retrain the navigation policy:
- The planning tasks **evaluate** planning algorithms; they do not train anything. The high-level
  navigation policy they drive ships with the repository
  (`isaaclab_experiments/policies/anymal_c_navigation.jit.pt`) and the low-level locomotion policy comes
  from the Isaac Lab asset server, so the planning experiments run right after the installation.
- That navigation policy is trained with **Isaac Lab's own navigation task**, then converted to the
  TorchScript file the planning environment loads:
   ```bash
   python isaaclab_experiments/train_skrl.py --task Isaac-Navigation-Flat-Anymal-C-v0 --headless
   python isaaclab_experiments/policies/convert_pt2jit.py --headless \
       --task Isaac-Navigation-Flat-Anymal-C-v0 \
       --checkpoint logs/skrl/<run>/checkpoints/best_agent.pt
   ```
- See **[isaaclab_experiments/README.md](isaaclab_experiments/README.md)** for where the policy goes and
  what it has to match.

### 5. (Optional) Train a locomotion policy:
- The same scripts train the legged-locomotion tasks this repository adds for the **Unitree Go1** (plain,
  and two variants that use a learned contact estimator) and the **Unitree Go2W** (including a commanded
  base height). They are listed, with what each one changes, in
  **[isaaclab_experiments/README.md](isaaclab_experiments/README.md)**:
   ```bash
   python isaaclab_experiments/train_rsl_rl.py --task IL4OP-Velocity-Rough-Unitree-Go1-v0 --headless
   python isaaclab_experiments/play_rsl_rl.py  --task IL4OP-Velocity-Rough-Unitree-Go1-v0 --export
   ```
- The Go2W tasks additionally need [robot_lab](https://github.com/fan-ziqi/robot_lab) (`v2.3.2`); without
  it they are not registered and the scripts carry on without them.

### 6. (Optional) Validate a trained policy sim-to-sim:
- Before a locomotion policy goes anywhere near the hardware, run it in a **different simulator**:
  `tools/sim2sim/` plays an exported policy in MuJoCo, on a model built from the robot's own URDF,
  through the sequence a real deployment follows (folded on the floor, fixed-gain stand-up, then the
  legs handed over). The robot can also be teleoperated, and walked without any policy at all.

   ```bash
   pip install "mujoco>=3.1"
   git clone https://github.com/unitreerobotics/unitree_ros.git ~/unitree_ros   # Go1 description
   python tools/sim2sim/build_models.py          # URDFs -> MuJoCo scenes
   python tools/sim2sim/play.py --robot go1 --teleop \
       --policy logs/rsl_rl/unitree_go1_rough/<run>/exported/policy.pt
   ```
- See **[tools/sim2sim/README.md](tools/sim2sim/README.md)** for the exported-policy requirement,
  the keyboard controls, and what the results do and do not tell you.


## :computer: In development & Future directions

- [x] Single agent planning using discrete world and decision models.
- [x] Support planning algorithms with continuous world and decision models.
- [ ] Extension of the single agent scenario to multi-agent problems (toilored to centralized and decentralized approaches).
- [ ] Extension of IL4OP to support dynamic world models applications.

## :handshake: Acknowledgements

IL4OP does not implement or propose a whole new simulator: it unifies and extends works that others published, and it would not
exist without it. Please respect the licence of each project and cite it alongside IL4OP.

| Project | How IL4OP uses it | Licence |
|---|---|---|
| [**Isaac Lab**](https://github.com/isaac-sim/IsaacLab) 2.3.2 (NVIDIA, ETH Zurich) | vendored in `IsaacLab/` and extended: the planning environments, the locomotion tasks and the training/play scripts are all built on its manager-based environments | BSD-3-Clause |
| [**Isaac Sim**](https://developer.nvidia.com/isaac/sim) 5.1.0 (NVIDIA Omniverse) | the simulator and the renderer underneath Isaac Lab, installed from pip and covered by its own NVIDIA licence | NVIDIA Omniverse licence |
| [**robot_lab**](https://github.com/fan-ziqi/robot_lab) 2.3.2 (Ziqi Fan) | the Go2W description and the wheeled-legged velocity task that `isaaclab_experiments/go2w_locomotion/` builds on | Apache-2.0 |
| [**unitree_ros**](https://github.com/unitreerobotics/unitree_ros) (Unitree Robotics) | the Go1 description and the fixed-gain posture control that `tools/sim2sim/` reproduces | BSD-3-Clause |
| [**MuJoCo**](https://github.com/google-deepmind/mujoco) (Google DeepMind) | the second simulator the policies are validated in | Apache-2.0 |
| [**RSL-RL**](https://github.com/leggedrobotics/rsl_rl) (ETH Zurich) and [**skrl**](https://github.com/Toni-SM/skrl) (Antonio Serrano-Muñoz) | the PPO implementations the policies are trained with | BSD-3-Clause / MIT |

```bibtex
@article{mittal2025isaaclab,
  title   = {Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author  = {Mittal, Mayank and Roth, Pascal and Tigue, James and Richard, Antoine and others},
  journal = {arXiv preprint arXiv:2511.04831},
  year    = {2025},
  url     = {https://arxiv.org/abs/2511.04831}
}

@software{fan-ziqi2024robot_lab,
  author = {Ziqi Fan},
  title  = {robot_lab: RL Extension Library for Robots, Based on IsaacLab.},
  url    = {https://github.com/fan-ziqi/robot_lab},
  year   = {2024}
}
```

## :book: Citation

If you use IL4OP in your research, please cite our work:

```bibtex
@misc{alves2025,
  **To appear**
}
```

