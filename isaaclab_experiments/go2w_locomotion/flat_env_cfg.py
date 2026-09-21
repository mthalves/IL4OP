"""Unitree Go2W flat-terrain locomotion, as trained for IL4OP.

Stock robot_lab configuration with the settings used by the
``unitree_go2w_flat`` training runs (``logs/rsl_rl/unitree_go2w_flat``).
"""

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.flat_env_cfg import (
    UnitreeGo2WFlatEnvCfg,
)

from .rough_env_cfg import apply_il4op_settings


@configclass
class Go2WFlatEnvCfg(UnitreeGo2WFlatEnvCfg):
    def __post_init__(self):
        # post init of parent (rough -> flat)
        super().__post_init__()
        apply_il4op_settings(self)

        # ------------------------------Commands------------------------------
        # forward/backward and yaw only
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Go2WFlatEnvCfg":
            self.disable_zero_weight_rewards()
