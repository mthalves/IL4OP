# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copied from IsaacLab 2.3.2 (isaaclab_tasks/manager_based/locomotion/velocity/config/go1/rough_env_cfg.py) for local modification.
import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_experiments.go1_locomotion.mdp as mdp
from isaaclab_experiments.go1_locomotion.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip

USE_SENSORS = False  # set to True to use height scan sensor for rough terrain locomotion

FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

@configclass
class UnitreeGo1RewardsCfg(RewardsCfg):
    """Rewards of the base task plus gait-quality and hardware-safety terms."""

    # -- gait quality
    # do not drag the feet along the ground (wears the rubber, trips on rough terrain)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    # keep the abduction joints near their nominal +-0.1 rad, i.e. each foot under its own
    # hip. Every leg is compared to its own default, so the two body sides stay independent
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    # lower bound on the stance width (nominal stance: 0.32 m front, 0.36 m rear)
    feet_stance_width = RewTerm(
        func=mdp.feet_stance_width,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "min_distance": 0.28,
        },
    )

    # -- hardware safety and motor stress
    # stay below 90% of the 30 rad/s motor speed
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=-0.1, params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot")}
    )
    # penalize asking for more torque than the 23.7 Nm the motors can deliver
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class UnitreeGo1TerminationsCfg(TerminationsCfg):
    """Terminations of the base task plus a tip-over guard."""

    # stop the episode before the robot tips over instead of letting it learn to crawl
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)})


@configclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: UnitreeGo1RewardsCfg = UnitreeGo1RewardsCfg()
    terminations: UnitreeGo1TerminationsCfg = UnitreeGo1TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if USE_SENSORS:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        else:
            # no height scan
            self.scene.height_scanner = None
            self.observations.policy.height_scan = None

        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # -- domain randomization (sim-to-real)
        # ground friction varies widely on the real robot; a single value makes the
        # policy rely on the contact behaviour of this simulation only
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)
        # battery and payload shift the centre of mass of the trunk
        self.events.base_com.params["asset_cfg"].body_names = "trunk"
        self.events.base_com.params["com_range"] = {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "z": (-0.02, 0.02)}
        # recover from pushes instead of only from the nominal state
        self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # -- gait quality
        # a swing-time reward of 0.01 barely pays for lifting a foot, which is what
        # produces the shuffling, narrow-stance gait; 0.25 is the value used on flat
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.4
        # -- hardware safety and motor stress
        # knees and thighs should never take the impacts
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
                "threshold": 1.0,
            },
        )
        # stay off the joint stops
        self.rewards.dof_pos_limits.weight = -1.0

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"


@configclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
