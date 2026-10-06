"""ContactNet, the contact estimator some of the policies observe."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]

class ContactNet:
    """The contact estimator the contact-framework policy expects among its observations.

    It rebuilds, from the MuJoCo state, the window of proprioceptive features ContactNet was
    trained on (joint positions and velocities, foot positions and velocities in the base
    frame) and returns one contact probability per foot, ordered (LF, RF, LH, RH).
    """

    def __init__(self, sim: "Sim2Sim"):
        sys.path.insert(0, str(REPO / "isaaclab_experiments" / "go1_locomotion_contact_framework"))
        import cnet as cnet_module

        self.cnet = cnet_module
        self.model = cnet_module.load_cnet()
        self.sim = sim

        joint_names = [f"{cnet_module.LEG_TO_GO1[leg]}_{j}_joint"
                       for leg in cnet_module.LEG_ORDER for j in cnet_module.JOINT_ORDER]
        self.qpos_idx = np.array([sim.model.jnt_qposadr[sim._joint(n)] for n in joint_names])
        self.qvel_idx = np.array([sim.model.jnt_dofadr[sim._joint(n)] for n in joint_names])
        self.default = np.array([sim.cfg["default"][n] for n in joint_names])
        self.foot_ids = [mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f)
                         for f in cnet_module.GO1_FEET]

        self.window = np.zeros((cnet_module.WINDOW_SIZE, cnet_module.INPUT_CHANNELS))
        self.filled = False

    def _features(self) -> np.ndarray:
        data, rot = self.sim.data, self.sim.base_rotation()
        base_pos = data.xpos[self.sim.base_id]
        base_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.sim.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.sim.base_id, base_vel, 0)

        foot_pos, foot_vel = [], []
        for foot in self.foot_ids:
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.sim.model, data, mujoco.mjtObj.mjOBJ_BODY, foot, velocity, 0)
            foot_pos.append(rot.T @ (data.xpos[foot] - base_pos))
            foot_vel.append(rot.T @ (velocity[3:] - base_vel[3:]))

        return np.concatenate([
            data.qpos[self.qpos_idx] - self.default,
            data.qvel[self.qvel_idx],
            np.concatenate(foot_pos),
            np.concatenate(foot_vel),
        ])

    def probabilities(self) -> np.ndarray:
        features = self._features()
        if not self.filled:                       # a fresh episode has no past to show
            self.window[:] = features
            self.filled = True
        else:
            self.window = np.roll(self.window, -1, axis=0)
            self.window[-1] = features
        with torch.no_grad():
            logits = self.model(torch.from_numpy(self.window).float().unsqueeze(0))
            return self.cnet.contact_probabilities(logits).squeeze(0).numpy()
