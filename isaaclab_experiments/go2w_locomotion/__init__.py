"""Unitree Go2W locomotion tasks.

Requires ``robot_lab`` (https://github.com/fan-ziqi/robot_lab, v2.3.2), which
provides the Go2W robot description and the base velocity-tracking task.

Three variants are registered:

* ``IL4OP-Velocity-Flat-Unitree-Go2W-v0``   - flat terrain
* ``IL4OP-Velocity-Rough-Unitree-Go2W-v0``  - rough terrain with curriculum
* ``IL4OP-Velocity-Flat-Z-Unitree-Go2W-v0`` - flat terrain with a commanded base height
"""

import gymnasium as gym

try:
    import robot_lab.tasks  # noqa: F401
except ImportError as error:
    raise ImportError(
        "isaaclab_experiments.go2w_locomotion requires robot_lab. Install it with:\n"
        "  git clone --branch v2.3.2 https://github.com/fan-ziqi/robot_lab.git\n"
        "  pip install -e robot_lab/source/robot_lab --config-settings editable_mode=compat"
    ) from error

from . import agents

##
# Register Gym environments.
##

FLAT_ENV_ID = "IL4OP-Velocity-Flat-Unitree-Go2W-v0"
ROUGH_ENV_ID = "IL4OP-Velocity-Rough-Unitree-Go2W-v0"
FLAT_Z_ENV_ID = "IL4OP-Velocity-Flat-Z-Unitree-Go2W-v0"

gym.register(
    id=FLAT_ENV_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2WFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WFlatPPORunnerCfg",
    },
)

gym.register(
    id=ROUGH_ENV_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Go2WRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WRoughPPORunnerCfg",
    },
)

gym.register(
    id=FLAT_Z_ENV_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_z_env_cfg:Go2WFlatZEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WFlatZPPORunnerCfg",
    },
)
