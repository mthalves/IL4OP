# IL4OP  
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-green)](https://isaac-sim.github.io/IsaacLab/)

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

1. Follow the IsaacSim [`5.1.0`]/IsaacLab [`2.3.2`] installation steps here:  
   :point_right: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html  
   - We recommend installing IsaacSim via `pip` and using `conda` as the virtual environment.  
   - The repository already contains the correct working version of IsaacLab (`2.3.2`), so **skip cloning IsaacLab manually**.  

2. Install the `isaaclab-experiment` folder using `pip install -e .`

3. You’re ready to run our environment :sunglasses:

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
         "alpha": 0.5,
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
- For a single experiment:
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

