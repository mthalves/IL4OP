# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_experiments.anymal_c_planning.configs.actions import ActionsCfg
from isaaclab_experiments.anymal_c_planning.configs.rewards import RewardsCfg
from isaaclab_experiments.anymal_c_planning.configs.observations import ObservationsCfg
from isaaclab_experiments.anymal_c_planning.configs.commands import CommandsCfg
from isaaclab_experiments.anymal_c_planning.configs.terminations import TerminationsCfg

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG as MY_ROBOT_CFG

from isaaclab_experiments.anymal_c_planning.configs.scene import UShapedCfg as MySceneCfg
from isaaclab_experiments.anymal_c_planning.configs.events import DiscreteEventCfg, ContinuousEventCfg

@configclass
class AnymalCPlanningEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the planning environment."""

    # Scene and event settings (scenario and continuous vs discrete)
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=2.5)
    events: DiscreteEventCfg = DiscreteEventCfg()

    # Standard settings
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self):
        # post init of parent
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 60.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.lidar_sensor.update_period = self.sim.dt * self.decimation

        # switch robot to anymal-d
        self.scene.robot = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = self.scene.init_pos

        self.commands.pose_commands.ranges.pos_x = (self.scene.init_pos[0],self.scene.init_pos[0])
        self.commands.pose_commands.ranges.pos_y = (self.scene.init_pos[1],self.scene.init_pos[1])
        
        # visual config
        self.commands.pose_commands.goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)

@configclass
class ContinuousAnymalCPlanningEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the planning environment."""

    # Scene and event settings (scenario and continuous vs discrete)
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=2.5)
    events: ContinuousEventCfg = ContinuousEventCfg()

    # Standard settings
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self):
        # post init of parent
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 60.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.lidar_sensor.update_period = self.sim.dt * self.decimation

        # switch robot to anymal-d
        self.scene.robot = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = self.scene.init_pos

        self.commands.pose_commands.ranges.pos_x = (self.scene.init_pos[0],self.scene.init_pos[0])
        self.commands.pose_commands.ranges.pos_y = (self.scene.init_pos[1],self.scene.init_pos[1])
        
        # visual config
        self.commands.pose_commands.goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
    