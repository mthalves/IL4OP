from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2WFlatPPORunnerCfg,
    UnitreeGo2WRoughPPORunnerCfg,
)


@configclass
class Go2WRoughPPORunnerCfg(UnitreeGo2WRoughPPORunnerCfg):
    pass


@configclass
class Go2WFlatPPORunnerCfg(UnitreeGo2WFlatPPORunnerCfg):
    pass


@configclass
class Go2WFlatZPPORunnerCfg(UnitreeGo2WFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 100000
        self.experiment_name = "unitree_go2w_flat_z"
