
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup

import isaaclab_tasks.manager_based.navigation.mdp as mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import torch

def get_high_level_obs(obs):
    hlobs = torch.zeros(len(obs),10,device=obs.device)
    hlobs[:,0:3] = obs[:,0:3]   # base lin vel
    hlobs[:,3:6] = obs[:,6:9]   # project grav
    hlobs[:,6:9] = obs[:,12:15] # pose x, y, z
    hlobs[:,9]   = obs[:,15]    # pose heading
    return hlobs

def get_low_level_obs(obs, llcmd):
    llobs = torch.zeros(len(obs),48,device=obs.device)
    llobs[:,0:3]  = obs[:,0:3]      # base lin vel
    llobs[:,3:6]  = obs[:,3:6]      # base ang vel
    llobs[:,6:9]  = obs[:,6:9]      # project grav
    llobs[:,9:12] = llcmd           # desired vel
    llobs[:,12:36]  = obs[:,16:40]  # joint info (pos and vel)
    llobs[:,36:]  = obs[:,40:]      # last action
    return llobs

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # :3
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        # 3:6
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # 6:9
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 9:12
        base_velocity = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 12:16
        pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_commands"})
        # 16:28
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # 28:40
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # 40:52
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
