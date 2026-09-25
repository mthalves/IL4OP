###
# loading app
###
from utils.launcher import PlaySkrlApp
app = PlaySkrlApp()
simulation_app = app.simulation_app

import skrl
app.check_skrl_version(skrl)
args_cli = app.args_cli
if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner
else:
    from skrl.utils.runner.torch import Runner
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()
if args_cli.ml_framework.startswith("jax"):
    skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

###
# main routine - start
###
import inspect
import os
import time

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaaclab_experiments  # noqa: F401  (registers the tasks)
from isaaclab_tasks.utils.hydra import hydra_task_config

from utils.evaluation import EpisodeStats


def set_evaluation_mode(agent):
    """Switch the agent to inference (skrl 1.x and 2.x)."""
    if hasattr(agent, "set_running_mode"):
        agent.set_running_mode("eval")
    else:
        agent.enable_training_mode(False)


def act(agent, obs):
    """Deterministic action of the agent; returns ``(actions, outputs)`` for skrl 1.x and 2.x."""
    if "states" in inspect.signature(agent.act).parameters:
        outputs = agent.act(obs, None, timestep=0, timesteps=0)  # skrl 2.x: (observations, states, ...)
    else:
        outputs = agent.act(obs, timestep=0, timesteps=0)  # skrl 1.x
    return outputs[0], outputs[-1]


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play and evaluate a skrl agent."""
    env_cfg, agent_cfg, resume_path = app.auto_config(args_cli, env_cfg, agent_cfg)
    if resume_path is None:
        return
    log_dir = env_cfg.log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    dt = getattr(env, "step_dt", None) or env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording a video of the rollout.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner, then load the checkpoint
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    set_evaluation_mode(runner.agent)

    stats = EpisodeStats(env.num_envs, env.unwrapped.device)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            sampled, outputs = act(runner.agent, obs)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[a].get("mean_actions", sampled[a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs.get("mean_actions", sampled)
            obs, rewards, terminated, truncated, _ = env.step(actions)
        if not hasattr(env, "possible_agents"):
            stats.update(rewards, torch.logical_or(terminated, truncated))
        timestep += 1

        # exit the play loop after recording one video or reaching the step budget
        if args_cli.video and timestep == args_cli.video_length:
            break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    stats.report(timestep)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
