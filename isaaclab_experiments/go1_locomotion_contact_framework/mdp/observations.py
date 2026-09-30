"""Observation terms specific to the contact framework."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_experiments.go1_locomotion_contact_framework.contact_estimator import get_contact_estimator


def contact_probabilities(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Contact probability of each foot estimated by ContactNet.

    Shape is ``(num_envs, 4)``, ordered (LF, RF, LH, RH), with values in [0, 1]. The
    estimator is shared with the ``contact_consistency`` reward, so the network runs once
    per step regardless of how many terms use it.
    """
    return get_contact_estimator(env, asset_cfg.name).update(env)
