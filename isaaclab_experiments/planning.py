###
# loading app
###
from utils.launcher import PlanningApp
app = PlanningApp()
simulation_app = app.simulation_app

import skrl
app.check_skrl_version(skrl)
args_cli = app.args_cli
if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner
if args_cli.ml_framework.startswith("jax"):
    skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

###
# environment setup
###
def init_environment(env_id, num_envs, disable_fabric=False, video=False, ml_framework='torch', device='cpu'):
    env_cfg = parse_env_cfg(env_id, device=device, num_envs=num_envs, use_fabric=not disable_fabric)
    env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array" if video else None)
    env = SkrlVecEnvWrapper(env, ml_framework=ml_framework)
    obs, _ = env.reset()
    return env, obs
###
# main routine - start
###
import gymnasium as gym
import time

from isaaclab_experiments.anymal_c_planning import ENV_ID, get_policies, get_low_level_obs, get_high_level_obs
HLPOLICY, LLPOLICY = get_policies(device=args_cli.device)

from utils.camera import init_camera, update_camera
from utils.log import LogFile

CAMERA_FOLLOW_ROBOT = True
def main():
    env, obs = init_environment(ENV_ID, args_cli.num_envs, args_cli.disable_fabric,
                           args_cli.video, args_cli.ml_framework, args_cli.device)
    
    env_shape = env.event_manager.cfg.planning.func.get_map_size()
    init_camera(env,size=env_shape)
    env.reset()
    
    header = ['Time','Reward','Time to reason']
    problem_name = env_shape = env.event_manager.cfg.planning.func.problem_name
    scenario_name = 'ushaped'

    planner_name = env_shape = env.event_manager.cfg.planning.func.planner_name

    log = LogFile(problem_name,scenario_name,planner_name,args_cli.exp_num,header)

    start_t = time.time()
    while simulation_app.is_running() and (time.time() - start_t) < 900.:
        hlobs = get_high_level_obs(obs)
        llcmd = HLPOLICY(hlobs)
        
        # forcing straight walk
        env.command_manager._terms['base_velocity'].vel_command_b = llcmd
        llobs = get_low_level_obs(obs, llcmd)
        actions = LLPOLICY(llobs)

        obs, rewards, terminated, truncated, extras = \
            env.step(actions.to(env.device))

        if env.event_manager.cfg.planning.func.update:
            cur_time = (time.time() - start_t)
            last_reward = env.event_manager.cfg.planning.func.last_reward \
                if not terminated and not truncated else -1
            last_time2reason = env.event_manager.cfg.planning.func.last_time2reason
            data = {'time':cur_time,
                    'reward':last_reward,
                    'time2reason':last_time2reason}
            log.write(data)

        if CAMERA_FOLLOW_ROBOT:
            update_camera(env)
        
        problem_end = env.event_manager.cfg.planning.func.problem_env.completed_all_tasks()
        if terminated or truncated or problem_end:
            break
    print('Experiment finished after %.1f seconds'%(time.time() - start_t))
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
    import sys
    sys.exit(0)