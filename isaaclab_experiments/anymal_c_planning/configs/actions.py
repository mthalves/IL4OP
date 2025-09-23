from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.navigation.mdp as mdp

from isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg import AnymalCFlatEnvCfg

LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()

@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)