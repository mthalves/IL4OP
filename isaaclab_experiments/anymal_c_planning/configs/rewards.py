from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""