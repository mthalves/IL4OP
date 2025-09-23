# IL4OP  
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-green)](https://isaac-sim.github.io/IsaacLab/)

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

1. Follow the IsaacSim [`5.0.0`]/IsaacLab [`2.2.1`] installation steps here:  
   :point_right: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html  
   - We recommend installing IsaacSim via `pip` and using `conda` as the virtual environment.  
   - The repository already contains the correct working version of IsaacLab (`2.2.1`), so **skip cloning IsaacLab manually**.  

2. Install the `isaaclab-experiment` folder using `pip install -e .`

3. You’re ready to run our environment :sunglasses:

## :arrow_forward: Running Experiments

1. Configure your setup:
- Open `isaaclab_experiments/anymal_c_planning/events.py:234`;
- Change the planning algorithm and problem definitions as needed. Take a look in an example:
  
  ```python
   params={
      "planning_method":{
            "name":"ibpomcp",
            "args":{
               "max_depth":20,
               "max_it":1000,
               "kwargs":{},
            },
            ###
            # Possibilities
            ###
            # 1) A-STAR
            #"name":"astar",
            #"args":{},
            #
            # 2) POMCP
            #"name":"pomcp",
            #"args":{
            #    "max_depth":20,
            #    "max_it":1000,
            #    "kwargs":{},
            #},
            #
            # 3) TB rho-POMCP
            #"name":"tbrhopomcp",
            #"args":{
            #    "max_depth":20,
            #    "max_it":1000,
            #    "kwargs":{'time_budget':2.0,'smallbag_size':10},
            #},
      },
      "problem": {
            "name":"inspection",
            "args": {
               "map_size_w": (17, 17),
               "z_min":0.1, "z_max":1.0,
               "resolution":               1., 
               "confirm_threshold":        2,
               "inscribed_radius":         0.1, 
               "inflation_radius":         0.0, 
               "cost_scaling_factor":      0.5,
               "visibility_radius":        5.,
               "max_inspection":           1,
               "max_inspection_distance":  2.9,
               "tasks": ['box_1','box_2','box_3'],
            },
      },
      "command_name": 'pose_commands',
      "robot_cfg": SceneEntityCfg("robot"),
      "lidar_cfg": SceneEntityCfg("lidar_sensor"),
   },
  ``` 

2. Run your experiment:
- For a single experiment:
   ```bash
   python isaaclab_experiments/planning.py
   ```
- For multiple experiments with screen recording:
   ```bash
   python isaaclab_experiments/run_planning_experiments.py
   ```

  - :warning: The script `run_planning_experiments.py` was tested on Ubuntu 22.04.5 LTS.
  - It has not been ported to other operating systems. Some additional requirements may still need to be installed.

3. Results are saved in `logs/inspection/`. This includes log files and videos for analysis.

## :book: Citation

If you use IL4OP in your research, please cite our work:

```bibtex
@misc{alves2025,
  **To appear**
}
```

