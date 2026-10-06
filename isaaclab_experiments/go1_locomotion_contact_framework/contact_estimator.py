"""Shared ContactNet estimator: one evaluation per environment step.

Both the ``contact_probabilities`` observation and the ``contact_consistency`` reward
need the estimate of the same step. They therefore share a single estimator per
environment instance, which keeps the rolling window, runs the network once per step and
caches the result for every other consumer of that step.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse

from . import cnet as cnet_module

#: run the network every n steps and hold the estimate in between (1 = every step)
EVERY_N_STEPS = 1


class ContactEstimator:
    """Rolling window of proprioceptive features and the ContactNet estimate built from it."""

    def __init__(self, env, asset_name: str = "robot", every_n_steps: int = EVERY_N_STEPS):
        self.model = cnet_module.load_cnet(device=env.device)
        self.window_size = cnet_module.WINDOW_SIZE
        self.every_n_steps = max(1, every_n_steps)
        self.asset = env.scene[asset_name]

        # ContactNet expects leg-major channels, while IsaacLab orders joints by joint type
        joint_names = [
            f"{cnet_module.LEG_TO_GO1[leg]}_{joint}_joint"
            for leg in cnet_module.LEG_ORDER
            for joint in cnet_module.JOINT_ORDER
        ]
        self.joint_ids = self.asset.find_joints(joint_names, preserve_order=True)[0]
        self.foot_ids = self.asset.find_bodies(list(cnet_module.GO1_FEET), preserve_order=True)[0]

        self.history = torch.zeros(
            env.num_envs, self.window_size, cnet_module.INPUT_CHANNELS, device=env.device
        )
        #: soft contact probability per foot, (num_envs, 4) ordered (LF, RF, LH, RH)
        self.probabilities = torch.zeros(env.num_envs, cnet_module.NUM_FEET, device=env.device)
        #: contacts of the most likely of the 16 states, same shape and order
        self.contacts = torch.zeros_like(self.probabilities)

        # a window filled with the past of a previous episode would be meaningless
        self.needs_fill = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._episode_length = torch.zeros_like(env.episode_length_buf)
        self._last_update = -1

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.history[env_ids] = 0.0
        self.needs_fill[env_ids] = True

    def _features(self) -> torch.Tensor:
        """The 48 channels of the current step: z = [q, qd, p_f, v_f]."""
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
        return cnet_module.normalize(features)

    def update(self, env) -> torch.Tensor:
        """Advance the window and return the contact probabilities of the current step.

        Calling this several times within one environment step returns the cached estimate,
        so the network runs once no matter how many terms consume it.
        """
        step = int(env.common_step_counter)
        if step == self._last_update:
            return self.probabilities

        # an episode that restarted has no usable past: refill its window with the new state
        episode_length = env.episode_length_buf
        self.needs_fill |= episode_length < self._episode_length
        self._episode_length = episode_length.clone()

        features = self._features()
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1, :] = features
        if self.needs_fill.any():
            self.history[self.needs_fill] = (
                features[self.needs_fill].unsqueeze(1).expand(-1, self.window_size, -1)
            )
            self.needs_fill[:] = False

        if step % self.every_n_steps == 0:
            # no_grad rather than inference_mode: the observation reaches the policy, and
            # inference tensors cannot be saved for backward
            with torch.no_grad():
                logits = self.model(self.history)
                self.probabilities = cnet_module.contact_probabilities(logits)
                self.contacts = cnet_module.class_to_contacts(logits.argmax(dim=-1))
        self._last_update = step
        return self.probabilities


def get_contact_estimator(env, asset_name: str = "robot") -> ContactEstimator:
    """The estimator of this environment, created on first use."""
    estimator = getattr(env, "_contact_estimator", None)
    if estimator is None:
        estimator = ContactEstimator(env, asset_name=asset_name)
        env._contact_estimator = estimator
    return estimator
