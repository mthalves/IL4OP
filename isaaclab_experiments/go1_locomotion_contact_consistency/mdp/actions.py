# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for the locomotion environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedJointPositionAction(JointPositionAction):
    """Joint position targets that reach the motors a few milliseconds late, as they do.

    A policy does not act the instant it decides: the command is packed, sent over the bus
    and picked up by the motor controllers on their own schedule. The few milliseconds that
    costs are a few milliseconds the robot keeps moving on the previous command, and a
    policy trained without them learns to react faster than the robot can. The lag varies
    between robots and between runs, so each environment is given its own, resampled when
    it resets.

    The lag is counted in physics steps, which is the rate the targets are written at, so
    one step is ``sim.dt`` -- 5 ms at the 200 Hz this task simulates.
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if not 0 <= cfg.min_delay <= cfg.max_delay:
            raise ValueError(f"a delay of {cfg.min_delay}..{cfg.max_delay} steps is not a range")
        self._history = torch.zeros(cfg.max_delay + 1, self.num_envs, self.action_dim, device=self.device)
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._envs = torch.arange(self.num_envs, device=self.device)
        self._cursor = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        index = slice(None) if env_ids is None else env_ids
        # a robot that has just been reset has no history: it is holding where it stands
        self._history[:, index] = self._asset.data.joint_pos[index][:, self._joint_ids]
        count = self.num_envs if env_ids is None else len(env_ids)
        self._delay[index] = torch.randint(
            self.cfg.min_delay, self.cfg.max_delay + 1, (count,), device=self.device
        )

    def apply_actions(self):
        self._cursor = (self._cursor + 1) % self._history.shape[0]
        self._history[self._cursor] = self._processed_actions
        late = (self._cursor - self._delay) % self._history.shape[0]
        self._asset.set_joint_position_target(self._history[late, self._envs], joint_ids=self._joint_ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration of :class:`DelayedJointPositionAction`."""

    class_type: type[ActionTerm] = DelayedJointPositionAction

    min_delay: int = 0
    """Shortest lag, in physics steps."""

    max_delay: int = 2
    """Longest lag, in physics steps. Each environment draws its own, uniformly."""
