import gymnasium as gym

from isaaclab_experiments.anymal_c_planning import agents
from isaaclab_experiments.anymal_c_planning import env
from isaaclab_experiments.anymal_c_planning.configs.observations import get_high_level_obs, get_low_level_obs

import torch
from isaaclab_experiments.anymal_c_planning import agents
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, read_file

##
# Register Gym environments.
##
ENV_ID = "Anymal-C-Planning-v0"

HIGH_POLICY_PATH = './isaaclab_experiments/policies/anymal_c_navigation.jit.pt'
LL_POLICY_PATH = read_file(f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt")

def get_policies(hl_path=None, ll_path=None, device='cpu'):
    hl_policy = torch.jit.load(hl_path) if hl_path else torch.jit.load(HIGH_POLICY_PATH)
    ll_policy = torch.jit.load(ll_path) if ll_path else torch.jit.load(LL_POLICY_PATH)

    if hl_policy.training:
        hl_policy = hl_policy.to(device).eval()
    if ll_policy.training:
        ll_policy = ll_policy.to(device).eval()

    return hl_policy, ll_policy
##
# Register Gym environments.
##

gym.register(
    id=ENV_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:AnymalCPlanningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCPlanningEnvPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
