# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copied from IsaacLab 2.3.2 (isaaclab_tasks/manager_based/locomotion/velocity/config/go1/rough_env_cfg.py) for local modification.
import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_experiments.go1_locomotion_contact_framework.mdp as mdp
from isaaclab_experiments.go1_locomotion_contact_framework.cnet import GO1_FEET as CNET_FEET
from isaaclab_experiments.go1_locomotion_contact_framework.velocity_env_cfg import (
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

#: The pose the policy is trained around, which is also the pose the robot is handed over
#: in. The thigh and calf angles are those the built-in Unitree controller stands at
#: (``unitree_ros``, body.cpp: 0.67 and -1.3), so a policy that holds its nominal pose
#: holds the posture the robot was stood up in; the abduction joints keep the 0.1 rad of
#: the Isaac Lab default, which widens the stance from 0.25 m to 0.32 m and costs only
#: 10 mm of ride height. Changing this changes the action offset and ``joint_pos_rel``, so
#: a policy trained with it must be run with it: tools/sim2sim has the same pose.
STAND_POSE = {
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    ".*_thigh_joint": 0.67,
    ".*_calf_joint": -1.3,
}
#: Height of the base above the feet at ``STAND_POSE``, measured on the model (metres).
STAND_HEIGHT = 0.329

#: Longest lag between a decision and the motors acting on it, in physics steps of 5 ms.
#: The robot needs a few milliseconds to pack the command, put it on the bus and have the
#: motor controllers pick it up; a policy trained at zero latency reacts faster than the
#: robot can. Set to 0 to command the joints the instant the policy decides.
ACTION_DELAY_STEPS = 2
# ContactNet orders the feet (LF, RF, LH, RH)
FEET_CNET = list(CNET_FEET)

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
    # lower bound on the stance width (the nominal stance of STAND_POSE is 0.32 m)
    feet_stance_width = RewTerm(
        func=mdp.feet_stance_width,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "min_distance": 0.28,
        },
    )

    # -- CNET
    # reward the agent when the contacts estimated by CNET match the simulated ones
    contact_consistency = RewTerm(
        func=mdp.contact_consistency,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_CNET, preserve_order=True),
            "threshold": 1.0,
            "use_probabilities": True,
        },
    )

    # -- posture, so that the policy is deployable from and back to the standing pose
    # hold the ride height the robot was stood up in, measured from the feet on the ground
    base_height = RewTerm(
        func=mdp.base_height_above_feet,
        weight=-50.0,
        params={
            "target_height": STAND_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True),
        },
    )
    # stand on four feet, not three: a lifted leg carries nothing and recovers nothing
    feet_airborne_when_still = RewTerm(
        func=mdp.feet_airborne_when_still,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True),
        },
    )
    # and come back to the nominal pose when there is nothing to do
    stand_still_posture = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
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
        if ACTION_DELAY_STEPS:
            # the same action term, reaching the motors as late as it does on the robot
            self.actions.joint_pos = mdp.DelayedJointPositionActionCfg(
                asset_name="robot",
                joint_names=[".*"],
                scale=self.actions.joint_pos.scale,
                use_default_offset=True,
                max_delay=ACTION_DELAY_STEPS,
            )

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

        # the policy is trained around the pose the robot is handed over in
        self.scene.robot.init_state.joint_pos = dict(STAND_POSE)

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
        # The actuator network carries the motor, but not what sits between it and the
        # link: the rotor turns with the joint through the gearbox, and the joint has
        # friction in it. The USD says neither, so both are set here, and varied because no
        # two robots are the same. The armature is the rotor inertia of the motor
        # (1.47e-4 kg m^2) seen through the reduction, 6.33 at the hip and the thigh and
        # 6.33 x 1.55 at the knee, where a belt stage follows the gearbox.
        self.events.joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "friction_distribution_params": (0.0, 0.04),
                "operation": "abs",
                "distribution": "uniform",
            },
        )
        self.events.joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint"]),
                "armature_distribution_params": (0.0047, 0.0071),
                "operation": "abs",
                "distribution": "uniform",
            },
        )
        self.events.knee_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_calf_joint"),
                "armature_distribution_params": (0.0113, 0.0169),
                "operation": "abs",
                "distribution": "uniform",
            },
        )

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
