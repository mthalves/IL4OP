"""Command terms for the Go2W locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformBaseHeightCommand(CommandTerm):
    """Samples a target base height uniformly within a range.

    The command is a single value per environment, resampled every
    ``resampling_time_range`` seconds. It is meant to be consumed by
    :func:`~isaaclab_experiments.go2w_locomotion.mdp.rewards.base_height_penalty`
    and exposed to the policy through ``mdp.generated_commands``.
    """

    cfg: UniformBaseHeightCommandCfg

    def __init__(self, cfg: UniformBaseHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "UniformBaseHeightCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tHeight range: ({self.cfg.ranges.min_height}, {self.cfg.ranges.max_height})"
        )

    @property
    def command(self) -> torch.Tensor:
        """Target base height. Shape is (num_envs, 1)."""
        return self.height_command

    def _update_metrics(self):
        self.metrics["error_height"] = torch.abs(self.height_command[:, 0] - self.robot.data.root_pos_w[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        self.height_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.ranges.min_height, self.cfg.ranges.max_height
        )

    def _update_command(self):
        pass


@configclass
class UniformBaseHeightCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformBaseHeightCommand`."""

    class_type: type = UniformBaseHeightCommand

    asset_name: str = MISSING
    """Name of the articulation whose base height is commanded."""

    @configclass
    class Ranges:
        min_height: float = MISSING
        max_height: float = MISSING

    ranges: Ranges = MISSING
    """Uniform sampling range of the target height (in meters)."""
