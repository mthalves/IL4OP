import argparse
import os
from packaging import version

from isaaclab.app import AppLauncher

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
        parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
        return parser
    
    def check_skrl_version(self, skrl):
        SKRL_VERSION = "1.4.2"
        if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
            skrl.logger.error(
                f"Unsupported skrl version: {skrl.__version__}. "
                f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
            )
            exit()