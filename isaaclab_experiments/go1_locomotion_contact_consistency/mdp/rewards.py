# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copied from IsaacLab 2.3.2 (isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py) for local modification.
"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def feet_stance_width(env, asset_cfg: SceneEntityCfg, min_distance: float) -> torch.Tensor:
    """Penalize a stance narrower than ``min_distance``.

    The lateral (base-frame y) separation of the front and of the rear foot pair is
    compared against ``min_distance``, and only stances narrower than that are penalized.
    The term therefore sets a lower bound on the stance width — it does not prescribe a
    gait, tie the two body sides together or reward any particular foot placement above
    the bound.

    ``asset_cfg.body_names`` must list the four feet as
    (front-left, front-right, rear-left, rear-right) with ``preserve_order=True``.
    """
    asset = env.scene[asset_cfg.name]
    # feet position relative to the base, expressed in the base frame
    offset_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids] - asset.data.root_link_pos_w.unsqueeze(1)
    quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, offset_w.shape[1], -1)
    pos_b = quat_apply_inverse(quat.reshape(-1, 4), offset_w.reshape(-1, 3)).reshape(offset_w.shape)

    front = torch.abs(pos_b[:, 0, 1] - pos_b[:, 1, 1])
    rear = torch.abs(pos_b[:, 2, 1] - pos_b[:, 3, 1])
    return (min_distance - front).clip(min=0.0) + (min_distance - rear).clip(min=0.0)


class contact_consistency(ManagerTermBase):
    """Reward agreement between the CNET contact estimate and the simulated contacts.

    .. math::  R_{cc} = 1 - \\frac{1}{4}\\sum_{i=1}^{4} |\\hat{c}_i - c_i| \\in [0, 1]

    where :math:`c` are the contacts measured by the contact sensor and :math:`\\hat c`
    are the contacts estimated by CNET from a window of proprioceptive history
    (joint positions and velocities, foot positions and velocities in the base frame).

    The term keeps a rolling window per environment, so it must be evaluated every step;
    with ``every_n_steps > 1`` the network runs less often and the previous estimate is
    reused, which trades accuracy for speed on large environment counts.

    Args:
        asset_cfg: the articulation, used for the joint and foot states.
        sensor_cfg: the contact sensor, whose ``body_names`` must resolve to the feet.
        threshold: contact force above which a foot counts as in contact (N).
        use_probabilities: compare against the marginal contact probabilities instead of
            the binary decision, which makes the reward continuous.
        every_n_steps: run the network every n steps and hold the estimate in between.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        from isaaclab_experiments.go1_locomotion_contact_consistency import cnet as cnet_module

        self._cnet_module = cnet_module
        self.model = cnet_module.load_cnet(device=env.device)
        self.window_size = cnet_module.WINDOW_SIZE

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]

        # CNET expects leg-major channels, while IsaacLab orders the joints by joint type
        joint_names = [
            f"{cnet_module.LEG_TO_GO1[leg]}_{joint}_joint"
            for leg in cnet_module.LEG_ORDER
            for joint in cnet_module.JOINT_ORDER
        ]
        self.joint_ids = self.asset.find_joints(joint_names, preserve_order=True)[0]
        self.foot_ids = self.asset.find_bodies(list(cnet_module.GO1_FEET), preserve_order=True)[0]

        self.history = torch.zeros(env.num_envs, self.window_size, cnet_module.INPUT_CHANNELS, device=env.device)
        # after a reset the window holds no past, so it is filled with the first sample
        self.needs_fill = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.estimate = torch.zeros(env.num_envs, cnet_module.NUM_FEET, device=env.device)
        self.step = 0

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.history[env_ids] = 0.0
        self.needs_fill[env_ids] = True

    def _features(self) -> torch.Tensor:
        """Assemble the 48 channels of the current step: joint pos/vel and foot pos/vel."""
        data = self.asset.data
        joint_pos = data.joint_pos[:, self.joint_ids]
        joint_vel = data.joint_vel[:, self.joint_ids]

        # foot pose and velocity relative to the base, expressed in the base frame
        root_pos = data.root_link_pos_w.unsqueeze(1)
        root_quat = data.root_link_quat_w.unsqueeze(1).expand(-1, len(self.foot_ids), -1).reshape(-1, 4)
        foot_pos = (data.body_link_pos_w[:, self.foot_ids] - root_pos).reshape(-1, 3)
        foot_vel = (data.body_lin_vel_w[:, self.foot_ids] - data.root_lin_vel_w.unsqueeze(1)).reshape(-1, 3)
        foot_pos = quat_apply_inverse(root_quat, foot_pos).reshape(joint_pos.shape[0], -1)
        foot_vel = quat_apply_inverse(root_quat, foot_vel).reshape(joint_pos.shape[0], -1)

        features = torch.cat([joint_pos, joint_vel, foot_pos, foot_vel], dim=-1)
        return (features - self._cnet_module.INPUT_MEAN) / self._cnet_module.INPUT_STD

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        threshold: float = 1.0,
        use_probabilities: bool = True,
        every_n_steps: int = 1,
    ) -> torch.Tensor:
        # 1. push the current sample into the rolling window
        features = self._features()
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1, :] = features
        if self.needs_fill.any():
            self.history[self.needs_fill] = features[self.needs_fill].unsqueeze(1).expand(-1, self.window_size, -1)
            self.needs_fill[:] = False

        # 2. estimate the contacts, reusing the previous estimate in between runs
        if self.step % every_n_steps == 0:
            with torch.inference_mode():
                logits = self.model(self.history)
                if use_probabilities:
                    self.estimate = self._cnet_module.contact_probabilities(logits).clone()
                else:
                    # hard decision: the most likely of the 16 contact states
                    self.estimate = self._cnet_module.class_to_contacts(logits.argmax(dim=-1)).clone()
        self.step += 1

        # 3. contacts measured in simulation, in the same foot order as CNET
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        contacts = (forces > threshold).float()

        return 1.0 - torch.abs(self.estimate - contacts).mean(dim=-1)
