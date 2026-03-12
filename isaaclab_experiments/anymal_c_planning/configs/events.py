
import torch

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_experiments.anymal_c_planning.configs.planning.discrete import OnlinePlanning as DiscretePlanning
from isaaclab_experiments.anymal_c_planning.configs.planning.continuous import OnlinePlanning as ContinuousPlanning
from isaaclab_experiments.anymal_c_planning.agents.planning_cfg import DISCRETE_AGENT, CONTINUOUS_AGENT

def reset_command( 
 env: ManagerBasedEnv, env_ids: torch.Tensor | None,
 command_name: str = 'pose_commands', robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    

    robot = env.scene[robot_cfg.name]
    root_pos = robot.data.default_root_state[0]

    cmd = env.command_manager._terms[command_name]
    cmd.pos_command_w[:, 0] = float(root_pos[0])
    cmd.pos_command_w[:, 1] = float(root_pos[1])
    cmd.pos_command_w[:, 2] = float(root_pos[2])

    cmd.heading_command_w[:] = robot.data.heading_w[0]

    target_vec_b = cmd.pos_command_w - robot.data.root_pos_w[:, :3]
    cmd.pos_command_b[:] = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), target_vec_b)
    cmd.heading_command_b[:] = wrap_to_pi(cmd.heading_command_w - robot.data.heading_w)


@configclass
class BaseEventCfg:
    reset_game = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset"
    )

    reset_command = EventTerm(
        func=reset_command,
        mode="reset",
        params={
            "command_name": 'pose_commands',
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class DiscreteEventCfg(BaseEventCfg):
    """Configuration for planning."""

    planning = EventTerm(
        func=DiscretePlanning,
        mode="interval",
        interval_range_s=(1.,1.),
        params={
            "planning_method":DISCRETE_AGENT,
            "problem": {
                "name":"inspection.discrete",
                "args": {
                    "map_size_w": (17, 17),
                    "z_min":0.1, "z_max":1.0,
                    "resolution":               1., 
                    "confirm_threshold":        2,
                    "inscribed_radius":         0.1, 
                    "inflation_radius":         0.0, 
                    "cost_scaling_factor":      0.5,
                    "visibility_radius":        5.,
                    "max_inspection":           1,
                    "max_inspection_distance":  2.9,
                    "tasks": ['box_1','box_2','box_3'],
                },
            },
            "command_name": 'pose_commands',
            "robot_cfg": SceneEntityCfg("robot"),
            "lidar_cfg": SceneEntityCfg("lidar_sensor"),
        },
    )



@configclass
class ContinuousEventCfg:
    """Configuration for planning."""

    planning = EventTerm(
        func=ContinuousPlanning,
        mode="interval",
        interval_range_s=(1.,1.),
        params={
            "planning_method":CONTINUOUS_AGENT,
            "problem": {
                "name":"inspection.continuous",
                "args": {
                    "map_size_w": (17, 17),
                    "z_min":0.1, "z_max":1.0,
                    "resolution":               1., 
                    "confirm_threshold":        2,
                    "inscribed_radius":         0.1, 
                    "inflation_radius":         0.0, 
                    "cost_scaling_factor":      0.5,
                    "visibility_radius":        5.,
                    "max_inspection":           1,
                    "max_inspection_distance":  2.9,
                    "tasks": ['box_1','box_2','box_3'],
                },
            },
            "command_name": 'pose_commands',
            "robot_cfg": SceneEntityCfg("robot"),
            "lidar_cfg": SceneEntityCfg("lidar_sensor"),
        },
    )