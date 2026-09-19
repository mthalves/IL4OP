import argparse
import sys
import os
import random

from packaging import version
from datetime import datetime

import importlib.metadata as metadata
import platform

from isaaclab.app import AppLauncher

import logging

# local imports
from isaaclab_experiments.utils.cli_args import add_rsl_rl_args, update_rsl_rl_cfg


def str2bool(value):
    """Parse boolean command-line values such as ``--log True`` or ``--log false``."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")

class TrainRslRLApp:

    def __init__(self):
        # parsing arguments
        parser = self.parse_args()

        # append RSL-RL cli arguments
        add_rsl_rl_args(parser)
        # append AppLauncher cli args
        AppLauncher.add_app_launcher_args(parser)
        self.args_cli, hydra_args = parser.parse_known_args()
        # always enable cameras to record video
        if self.args_cli.video:
            self.args_cli.enable_cameras = True
        # clear out sys.argv for Hydra
        sys.argv = [sys.argv[0]] + hydra_args

        # launch omniverse app
        self.app_launcher = AppLauncher(self.args_cli)
        self.simulation_app = self.app_launcher.app
        self.check_rsl_rl_version()

        # import logger
        self.logger = logging.getLogger(__name__)

    
    def parse_args(self):
        # add argparse arguments
        parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
        parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
        parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
        parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
        parser.add_argument("--num_envs", type=int, default=2024, help="Number of environments to simulate.")
        parser.add_argument("--task", type=str, default=None, help="Name of the task.")
        parser.add_argument(
            "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
        )
        parser.add_argument("--seed", type=int, default=2017, help="Seed used for the environment")
        parser.add_argument("--max_iterations", type=int, default=100000, help="RL Policy training iterations.")
        parser.add_argument(
            "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
        )
        parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
        parser.add_argument(
            "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
        )
        return parser
    
    def check_rsl_rl_version(self):
        """Check for minimum supported RSL-RL version."""
        # check minimum supported rsl-rl version
        RSL_RL_VERSION = "3.0.1"
        installed_version = metadata.version("rsl-rl-lib")
        if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
            if platform.system() == "Windows":
                cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
            else:
                cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
            print(
                f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
                f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
                f"\n\n\t{' '.join(cmd)}\n"
            )
            exit(1)
    
    def auto_config(self, args_cli, env_cfg, agent_cfg):
        # override configurations with non-hydra CLI arguments
        agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        # check for invalid combination of CPU device with distributed training
        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError(
                "Distributed training is not supported when using CPU device. "
                "Please use GPU device (e.g., --device cuda) for distributed training."
            )

        # multi-gpu training configuration
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{self.app_launcher.local_rank}"
            agent_cfg.device = f"cuda:{self.app_launcher.local_rank}"

            # set seed to have diversity in different threads
            seed = agent_cfg.seed + self.app_launcher.local_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        # specify directory for logging runs: {time-stamp}_{run_name}
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
        # change it (see PR #2346, comment-2819298849)
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        return env_cfg, agent_cfg, log_dir, log_root_path


class TrainSkrlApp:

    def __init__(self):
        # parsing arguments
        parser = self.parse_args()

        # append AppLauncher cli args
        AppLauncher.add_app_launcher_args(parser)
        self.args_cli, hydra_args = parser.parse_known_args()
        # always enable cameras to record video
        if self.args_cli.video:
            self.args_cli.enable_cameras = True
        # clear out sys.argv for Hydra
        sys.argv = [sys.argv[0]] + hydra_args

        # launch omniverse app
        self.app_launcher = AppLauncher(self.args_cli)
        self.simulation_app = self.app_launcher.app


    def parse_args(self):
        # add argparse arguments
        parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
        parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
        parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
        parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
        parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
        parser.add_argument("--task", type=str, default=None, help="Name of the task.")
        parser.add_argument(
            "--agent",
            type=str,
            default=None,
            help=(
                "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
                "--algorithm is used to determine the default agent configuration entry point."
            ),
        )
        parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
        parser.add_argument(
            "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
        )
        parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
        parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
        parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
        parser.add_argument(
            "--ml_framework",
            type=str,
            default="torch",
            choices=["torch", "jax", "jax-numpy"],
            help="The ML framework used for training the skrl agent.",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            default="PPO",
            choices=["AMP", "PPO", "IPPO", "MAPPO"],
            help="The RL algorithm used for training the skrl agent.",
        )
        parser.add_argument(
            "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
        )
        return parser
    
    def check_skrl_version(self, skrl):
        SKRL_VERSION = "1.4.3"
        if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
            skrl.logger.error(
                f"Unsupported skrl version: {skrl.__version__}. "
                f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
            )
            exit()
    
    def auto_config(self, args_cli, env_cfg, agent_cfg, algorithm):
        # override configurations with non-hydra CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
        # check for invalid combination of CPU device with distributed training
        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError(
                "Distributed training is not supported when using CPU device. "
                "Please use GPU device (e.g., --device cuda) for distributed training."
            )
    
        # multi-gpu training config
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{self.app_launcher.local_rank}"
        # max iterations for training
        if args_cli.max_iterations:
            agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        agent_cfg["trainer"]["close_environment_at_exit"] = False
    
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
    
        # set the agent and environment seed from command line
        # note: certain randomization occur in the environment initialization so we set the seed here
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
        env_cfg.seed = agent_cfg["seed"]
    
        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        # specify directory for logging runs: {time-stamp}_{run_name}
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
        # The Ray Tune workflow extracts experiment name using the logging line below, hence,
        # do not change it (see PR #2346, comment-2819298849)
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg["agent"]["experiment"]["experiment_name"]:
            log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
        # set directory into agent config
        agent_cfg["agent"]["experiment"]["directory"] = log_root_path
        agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
        # update log_dir
        log_dir = os.path.join(log_root_path, log_dir)
    
        # dump the configuration into log-directory
        from isaaclab.utils.io import dump_yaml
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
        # get checkpoint path (to resume training)
        from isaaclab.utils.assets import retrieve_file_path
        resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
        return env_cfg, agent_cfg, log_dir, resume_path


class PlayApp:

    def __init__(self):
        # parsing arguments
        parser = self.parse_args()

        # append AppLauncher cli args
        AppLauncher.add_app_launcher_args(parser)
        self.args_cli = parser.parse_args()
        # always enable cameras to record video
        if self.args_cli.video:
            self.args_cli.enable_cameras = True
        self.algorithm = self.args_cli.algorithm.lower()

        # launch omniverse app
        self.app_launcher = AppLauncher(self.args_cli)
        self.simulation_app = self.app_launcher.app

    
    def parse_args(self):
        # add argparse arguments
        parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
        parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
        parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
        parser.add_argument(
            "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
        )
        parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
        parser.add_argument("--task", type=str, default=None, help="Name of the task.")
        parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
        parser.add_argument(
            "--use_pretrained_checkpoint",
            action="store_true",
            help="Use the pre-trained checkpoint from Nucleus.",
        )
        parser.add_argument(
            "--ml_framework",
            type=str,
            default="torch",
            choices=["torch", "jax", "jax-numpy"],
            help="The ML framework used for training the skrl agent.",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            default="PPO",
            choices=["AMP", "PPO", "IPPO", "MAPPO"],
            help="The RL algorithm used for training the skrl agent.",
        )
        parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
        return parser
    
    def check_skrl_version(self, skrl):
        SKRL_VERSION = "1.4.2"
        if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
            skrl.logger.error(
                f"Unsupported skrl version: {skrl.__version__}. "
                f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
            )
            exit()
    
    def init_runner(self, env, experiment_cfg, resume_path, Runner):
        runner = Runner(env, experiment_cfg)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        # set agent to evaluation mode
        runner.agent.set_running_mode("eval")
        return runner

    def init_log(self, experiment_cfg):
        from isaaclab_tasks.utils import get_checkpoint_path
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
        # specify directory for logging experiments (load checkpoint)
        log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        # get checkpoint path
        if self.args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("skrl", self.args_cli.task)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif self.args_cli.checkpoint:
            resume_path = os.path.abspath(self.args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(
                log_root_path, run_dir=f".*_{self.algorithm}_{self.args_cli.ml_framework}", other_dirs=["checkpoints"]
            )
        log_dir = os.path.dirname(os.path.dirname(resume_path))
        return log_dir, resume_path

class PlanningApp:

    def __init__(self):
        # parsing arguments
        parser = self.parse_args()

        # append AppLauncher cli args
        AppLauncher.add_app_launcher_args(parser)
        self.args_cli = parser.parse_args()
        # always enable cameras to record video
        if self.args_cli.video:
            self.args_cli.enable_cameras = True

        # launch omniverse app
        self.app_launcher = AppLauncher(self.args_cli)
        self.simulation_app = self.app_launcher.app
    
    def parse_args(self):
        # add argparse arguments
        parser = argparse.ArgumentParser(description="Play planning of an RL agent from skrl.")
        # simulation
        parser.add_argument("--exp_num", type=int, default=0, help="Experiment ID.")
        parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
        parser.add_argument("--task", type=str, default='inspection', help="Name of the task.")
        parser.add_argument("--space", type=str, default='discrete', help="Space representation mode (discrete or continuous).")
        # policy
        parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
        parser.add_argument(
            "--use_pretrained_checkpoint",
            action="store_true",
            help="Use the pre-trained checkpoint from Nucleus.",
        )
        parser.add_argument(
            "--ml_framework",
            type=str,
            default="torch",
            choices=["torch", "jax", "jax-numpy"],
            help="The ML framework used for training the skrl agent.",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            default="PPO",
            choices=["AMP", "PPO", "IPPO", "MAPPO"],
            help="The RL algorithm used for training the skrl agent.",
        )
        parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")
        #video
        parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
        parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
        parser.add_argument(
            "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
        )
        parser.add_argument("--follow_camera", type=str2bool, default=True, help="Make the camera follow the robot.")
        # log
        parser.add_argument("--log", type=str2bool, default=False, help="Log info from the experiments.")
        return parser
    
    def check_skrl_version(self, skrl):
        SKRL_VERSION = "1.4.3"
        if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
            skrl.logger.error(
                f"Unsupported skrl version: {skrl.__version__}. "
                f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
            )
            exit()