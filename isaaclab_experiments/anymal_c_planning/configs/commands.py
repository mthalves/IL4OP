import math
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    pose_commands = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(3600., 3600.),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(0,0), pos_y=(0,0), heading=(0,0)),
        goal_pose_visualizer_cfg=RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/pose_goal"),
    )


    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3600., 3600.),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )
