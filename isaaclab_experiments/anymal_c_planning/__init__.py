import gymnasium as gym

from isaaclab_experiments.anymal_c_planning import agents
from isaaclab_experiments.anymal_c_planning import env
from isaaclab_experiments.anymal_c_planning.configs.observations import get_high_level_obs, get_low_level_obs

import torch
from isaaclab_experiments.anymal_c_planning import agents
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, read_file

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

##
# Register Gym environments.
##

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
DISCRETE_ENV_ID = "Anymal-C-Planning-v0"
gym.register(
    id=DISCRETE_ENV_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:AnymalCPlanningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCPlanningEnvPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

CONTINUOUS_ENV_ID = "Continuous-Anymal-C-Planning-v0"
gym.register(
    id=CONTINUOUS_ENV_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:ContinuousAnymalCPlanningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCPlanningEnvPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

def get_env_id(space_representation):
    if space_representation == 'discrete':
        return DISCRETE_ENV_ID 
    elif space_representation == 'continuous':
        return CONTINUOUS_ENV_ID
    else:
        print('Invalid space representation',space_representation,' -- options: "discrete" or "continuous".')
        exit(1)

def init_environment(env_id, num_envs, disable_fabric=False, video=False, ml_framework='torch', device='cpu'):
    env_cfg = parse_env_cfg(env_id, device=device, num_envs=num_envs, use_fabric=not disable_fabric)
    env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array" if video else None)
    env = SkrlVecEnvWrapper(env, ml_framework=ml_framework)
    obs, _ = env.reset()
    return env, obs