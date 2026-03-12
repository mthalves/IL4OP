
import torch

from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    @property
    def max_time(self):
        return 900.

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )