# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copied from IsaacLab 2.3.2 (isaaclab_tasks/manager_based/locomotion/velocity/config/go1/flat_env_cfg.py) for local modification.
from typing import ClassVar

from isaaclab.utils import configclass

from .rough_env_cfg import FEET_AIR_TIME_WEIGHT, PlayMixin, UnitreeGo1RoughEnvCfg


@configclass
class UnitreeGo1FlatEnvCfg(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = FEET_AIR_TIME_WEIGHT

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


##
# The ablation, on flat terrain: see ``UnitreeGo1RoughEnvCfg`` for what the switches are.
##


@configclass
class UnitreeGo1CEPFlatEnvCfg(UnitreeGo1FlatEnvCfg):
    """CEP only: the policy observes the contact probabilities."""

    cep: ClassVar[bool] = True


@configclass
class UnitreeGo1CCRFlatEnvCfg(UnitreeGo1FlatEnvCfg):
    """CCR only: the policy is rewarded for agreeing with ContactNet."""

    ccr: ClassVar[bool] = True


@configclass
class UnitreeGo1ContactFrameworkFlatEnvCfg(UnitreeGo1FlatEnvCfg):
    """The contact framework: CEP and CCR together."""

    cep: ClassVar[bool] = True
    ccr: ClassVar[bool] = True


@configclass
class UnitreeGo1FlatEnvCfg_PLAY(PlayMixin, UnitreeGo1FlatEnvCfg):
    pass


@configclass
class UnitreeGo1CEPFlatEnvCfg_PLAY(PlayMixin, UnitreeGo1CEPFlatEnvCfg):
    pass


@configclass
class UnitreeGo1CCRFlatEnvCfg_PLAY(PlayMixin, UnitreeGo1CCRFlatEnvCfg):
    pass


@configclass
class UnitreeGo1ContactFrameworkFlatEnvCfg_PLAY(PlayMixin, UnitreeGo1ContactFrameworkFlatEnvCfg):
    pass
