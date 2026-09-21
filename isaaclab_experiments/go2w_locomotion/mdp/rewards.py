"""Reward terms for the Go2W locomotion task with a commanded base height.

These terms were reconstructed from the resolved configuration of the
``unitree_go2w_flat_z`` training runs (``logs/rsl_rl/unitree_go2w_flat_z/*/params/env.yaml``)
after the original source was lost. Function names and parameters match the
runs exactly; the bodies follow the conventions of ``robot_lab``'s reward terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _upright_scale(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fade a penalty out as the robot tips over, as done by robot_lab's reward terms."""
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def base_height_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
    target_height: float | None = None,
    command_name: str | None = None,
) -> torch.Tensor:
    """Squared error between the base height and its target.

    The target is ``target_height`` when given, otherwise the value of the
    ``command_name`` command (see :class:`~.commands.UniformBaseHeightCommand`).
    With a height scanner the target is measured from the terrain under the base,
    so the term also works on rough terrain.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if target_height is not None:
        target = torch.full((env.num_envs,), target_height, device=env.device)
    else:
        target = env.command_manager.get_command(command_name)[:, 0]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            target = asset.data.root_pos_w[:, 2]
        else:
            target = target + torch.mean(ray_hits, dim=1)

    reward = torch.square(asset.data.root_pos_w[:, 2] - target)
    return reward * _upright_scale(env)


def go2w_joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    hip_reference_weight: float = 0.5,
    min_symmetry_scale: float = 0.25,
    yaw_max: float = 1.0,
) -> torch.Tensor:
    """Diagonal leg symmetry penalty for Go2W, relaxed while turning.

    Thigh and calf joints of diagonal legs (FR/RL and FL/RR) are penalised for
    differing, as in robot_lab's ``joint_mirror``. Hip joints are instead kept
    close to their default position, weighted by ``hip_reference_weight``, since
    mirrored hips are not meaningful for a wheeled stance. The whole penalty is
    scaled down linearly with the commanded yaw rate, from 1 at zero yaw to
    ``min_symmetry_scale`` at ``yaw_max`` and above.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    cache_name = "go2w_joint_mirror_cache"
    if getattr(env, cache_name, None) is None:
        pairs = [
            (asset.find_joints(["FR_thigh_joint", "FR_calf_joint"])[0], asset.find_joints(["RL_thigh_joint", "RL_calf_joint"])[0]),
            (asset.find_joints(["FL_thigh_joint", "FL_calf_joint"])[0], asset.find_joints(["RR_thigh_joint", "RR_calf_joint"])[0]),
        ]
        hips = asset.find_joints(".*_hip_joint")[0]
        setattr(env, cache_name, (pairs, hips))
    pairs, hips = getattr(env, cache_name)

    joint_pos = asset.data.joint_pos
    mirror_error = torch.zeros(env.num_envs, device=env.device)
    for left, right in pairs:
        mirror_error += torch.sum(torch.square(joint_pos[:, left] - joint_pos[:, right]), dim=-1)
    mirror_error /= len(pairs)

    hip_error = torch.sum(torch.square(joint_pos[:, hips] - asset.data.default_joint_pos[:, hips]), dim=-1)

    yaw_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    symmetry_scale = torch.clamp(1.0 - yaw_cmd / yaw_max, min=min_symmetry_scale, max=1.0)

    reward = (mirror_error + hip_reference_weight * hip_error) * symmetry_scale
    return reward * _upright_scale(env)


def wheel_position_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Keep each wheel under its hip while the base height changes.

    Penalises the horizontal (base-frame x/y) offset between every wheel body in
    ``asset_cfg.body_names`` and the hip body of the same leg, so that lowering
    or raising the base folds the legs vertically instead of splaying the wheels.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    cache_name = "wheel_position_penalty_cache"
    if getattr(env, cache_name, None) is None:
        wheel_ids, wheel_names = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
        hip_names = [name.replace("_foot", "_hip") for name in wheel_names]
        hip_ids = asset.find_bodies(hip_names, preserve_order=True)[0]
        setattr(env, cache_name, (wheel_ids, hip_ids))
    wheel_ids, hip_ids = getattr(env, cache_name)

    offset_w = asset.data.body_link_pos_w[:, wheel_ids] - asset.data.body_link_pos_w[:, hip_ids]
    quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, offset_w.shape[1], -1)
    offset_b = math_utils.quat_apply_inverse(quat.reshape(-1, 4), offset_w.reshape(-1, 3)).reshape(offset_w.shape)

    reward = torch.sum(torch.square(offset_b[..., :2]), dim=(1, 2))
    return reward * _upright_scale(env)
