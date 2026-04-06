
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


def avoid_colision(
 env: ManagerBasedEnv, env_ids: torch.Tensor | None,
 command_name: str = 'pose_commands', 
 robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
 lidar_cfg: SceneEntityCfg = SceneEntityCfg("lidar_sensor"),
 safe_distance: float = 0.25):
    
    robot = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[0, :3]
    lidar_hits = env.scene[lidar_cfg.name].data.ray_hits_w[0]
    cmd = env.command_manager._terms[command_name]

    # --- compute vectors from robot to hits ---
    rel = lidar_hits - robot_pos.unsqueeze(0)   # (N, 3)
    dists = torch.norm(rel, dim=1)              # (N,)

    # --- filter valid hits (ignore invalid / far / zero) ---
    mask = (dists > 1e-3) & (dists < safe_distance)

    if not torch.any(mask):
        return

    # if there is an obstacle closer, stop the robot
    print("Avoiding collision...")
    rel = rel[mask]
    dists = dists[mask]

    # --- normalize directions ---
    dirs = rel / (dists.unsqueeze(1) + 1e-6)

    # --- repulsion weights (closer = stronger) ---
    weights = (safe_distance - dists) / safe_distance
    weights = weights.unsqueeze(1)

    # --- compute repulsion vector (AWAY from obstacles) ---
    repulsion = -torch.sum(weights * dirs, dim=0)

    # --- ignore vertical component ---
    repulsion[2] = 0.0

    # normalize
    norm = torch.norm(repulsion) + 1e-6
    repulsion = repulsion / norm

    # --- step away from obstacle ---
    step_size = 0.5
    new_target = robot_pos.clone()
    new_target[:2] += step_size * repulsion[:2]

    print("> Repulsion:", repulsion[:2])

    # --- update command ---
    cmd.pos_command_w[:, 0] = new_target[0]
    cmd.pos_command_w[:, 1] = new_target[1]
    cmd.pos_command_w[:, 2] = robot_pos[2]

    # --- convert to base frame ---
    target_vec_b = cmd.pos_command_w - robot.data.root_pos_w[:, :3]
    cmd.pos_command_b[:] = quat_apply_inverse(
        yaw_quat(robot.data.root_quat_w),
        target_vec_b
    )

    print("New target waypoint:", cmd.pos_command_w)

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
                    "map_size_w"                : (17, 17),     # in meters
                    "z_min":    0.1, "z_max"    : 1.0,
                    "resolution"                : 0.2, 
                    "confirm_threshold"         : 2,            # obstacle ray hitting confirmation thereshold
                    "robot_radius"              : 0.5,          # robot radius (in meters)
                    "inflation_radius"          : 1.5,          # inflation for obstacle avoidance (in meters)
                    "cost_scaling_factor"       : 3.0,          # inflation cost scaling factor (approximately the length of navigation)
                    "visibility_radius"         : 5.0,          # robot visibility radius 
                    "max_inspection"            : 1,
                    "max_inspection_distance"   : 3.0,
                    "tasks": ['box_1','box_2','box_3'],
                },
            },
            "command_name": 'pose_commands',
            "robot_cfg": SceneEntityCfg("robot"),
            "lidar_cfg": SceneEntityCfg("lidar_sensor"),
        },
    )

    obstacle_avoidance = EventTerm(
        func=avoid_colision,
        mode="interval",
        interval_range_s=(0.3,0.3),
        params={
            "command_name": 'pose_commands',
            "robot_cfg": SceneEntityCfg("robot"),
            "lidar_cfg": SceneEntityCfg("lidar_sensor"),
            "safe_distance": 0.5,
        }
    )