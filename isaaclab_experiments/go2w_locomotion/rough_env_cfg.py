"""Unitree Go2W rough-terrain locomotion, as trained for IL4OP.

Stock robot_lab configuration with the settings used by the
``unitree_go2w_rough`` training runs (``logs/rsl_rl/unitree_go2w_rough``).
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.rough_env_cfg import (
    UnitreeGo2WRoughEnvCfg,
)

import isaaclab_experiments.go2w_locomotion.mdp as mdp


def apply_il4op_settings(cfg: UnitreeGo2WRoughEnvCfg):
    """Settings shared by every IL4OP Go2W variant, on top of stock robot_lab."""
    cfg.sim.enable_scene_query_support = True
    # spawn upright: no roll/pitch randomization at reset
    cfg.events.randomize_reset_base.params["pose_range"]["roll"] = (0.0, 0.0)
    cfg.events.randomize_reset_base.params["pose_range"]["pitch"] = (0.0, 0.0)
    # terminate on base or hip contact (disabled in stock robot_lab)
    cfg.terminations.illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[cfg.base_link_name, ".*_hip"]),
            "threshold": 1.0,
        },
    )


@configclass
class Go2WRoughEnvCfg(UnitreeGo2WRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        apply_il4op_settings(self)

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)

        # ------------------------------Observations------------------------------
        # the base configuration removes the linear velocity from the policy; restore it
        self.observations.policy.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-100.0, 100.0), scale=2.0
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Go2WRoughEnvCfg":
            self.disable_zero_weight_rewards()
