"""Where a leg can put its foot, and which joint angles put it there."""

from __future__ import annotations

import math

import mujoco
import numpy as np

def euler_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation of the trunk from roll, pitch and yaw, applied in that order."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


class Legs:
    """The four legs of a standing robot: where they reach, and what reaches there.

    Everything is read off the model at the standing pose instead of being written down,
    so the same foot target means the same thing on both robots.
    """

    JOINT_MARGIN = 0.05           # how close to a joint stop a pose may go
    IK_STEPS = 12                 # Newton steps of the solver
    IK_DAMPING = 1e-4
    IK_TOLERANCE = 2e-4           # 0.2 mm at the foot is close enough
    LEGS = ("FL", "FR", "RL", "RR")

    def __init__(self, sim, stand: np.ndarray):
        self.sim = sim
        model, names = sim.model, sim.cfg["joints"]
        # a pose must never ask for a joint past its stop: the motor would sit there
        # pushing against the mechanical limit with everything it has
        limits = model.jnt_range[[sim._joint(n) for n in names]]
        limited = model.jnt_limited[[sim._joint(n) for n in names]].astype(bool)
        self.pose_low = np.where(limited, limits[:, 0] + self.JOINT_MARGIN, -np.inf)
        self.pose_high = np.where(limited, limits[:, 1] - self.JOINT_MARGIN, np.inf)

        depth = {}
        for body in range(model.nbody):
            parent = model.body_parentid[body]
            depth[body] = 0 if parent == body else depth[parent] + 1

        def body_name(body):
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""

        # the deepest body of a leg is the one that touches the ground: a foot or a wheel
        self.feet = {leg: max((b for b in range(model.nbody) if body_name(b).startswith(leg)),
                              key=lambda b: depth[b]) for leg in self.LEGS}
        # the three joints that place the foot; a wheel turns under it and does not
        self.joints = {leg: [i for i, n in enumerate(names)
                             if n.startswith(leg) and n.split("_")[1] in ("hip", "thigh", "calf")]
                       for leg in self.LEGS}
        self.dofs = {leg: np.array([model.jnt_dofadr[sim._joint(names[i])] for i in self.joints[leg]])
                     for leg in self.LEGS}

        self.data = mujoco.MjData(model)
        self.stance = self.places(stand)

    def places(self, pose: np.ndarray) -> dict[str, np.ndarray]:
        """Where each foot sits, seen from the trunk, when the joints hold this pose."""
        sim, data = self.sim, self.data
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0          # the trunk at the origin: its frame is the world
        data.qpos[sim.qpos_idx] = pose
        mujoco.mj_kinematics(sim.model, data)
        return {leg: (data.xpos[body] - data.xpos[sim.base_id]).copy()
                for leg, body in self.feet.items()}

    def solve(self, targets: dict[str, np.ndarray], start: np.ndarray) -> np.ndarray:
        """Joint angles that put each foot where it is wanted, seen from the trunk.

        A leg has three joints and a foot has three coordinates, so this is the ordinary
        inverse kinematics of a leg, solved on the model itself with a damped Newton step
        per iteration -- damped because a leg with a straight knee is badly conditioned.
        It is warm-started from the pose last commanded, usually a millimetre away, so it
        leaves after an iteration or two.
        """
        sim, model = self.sim, self.sim.model
        data = self.data
        pose = start.copy()
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        jacp = np.zeros((3, model.nv))

        for _ in range(self.IK_STEPS):
            data.qpos[sim.qpos_idx] = pose
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)      # the Jacobians are built on top of this
            worst = 0.0
            for leg, body in self.feet.items():
                error = targets[leg] - (data.xpos[body] - data.xpos[sim.base_id])
                worst = max(worst, float(np.linalg.norm(error)))
                mujoco.mj_jacBody(model, data, jacp, None, body)
                jac = jacp[:, self.dofs[leg]]
                step = jac.T @ np.linalg.solve(jac @ jac.T + self.IK_DAMPING * np.eye(3), error)
                index = self.joints[leg]
                pose[index] = np.clip(pose[index] + np.clip(step, -0.3, 0.3),
                                      self.pose_low[index], self.pose_high[index])
            if worst < self.IK_TOLERANCE:
                break
        return pose
