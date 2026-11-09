"""The simulated robot: the model, the motors, and the observation a policy reads."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from .contact import ContactNet
from .specs import ROBOTS

class Sim2Sim:
    def __init__(self, robot: str, policy_path: Path | None, obs_dim: int | None = None,
                 nominal: str = "current", joint_damping: float | None = None):
        self.cfg = ROBOTS[robot]
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg["scene"]))
        self.data = mujoco.MjData(self.model)
        self.policy = None
        if policy_path is not None:
            self._load_policy(policy_path)
            # the policy knows how much it reads, which says what it observes: 45 numbers are the
            # robot's own, 49 add ContactNet's four probabilities. Asking for another size is an
            # error, not a guess
            self.policy_inputs = self._policy_inputs()
            if obs_dim is not None and obs_dim != self.policy_inputs:
                raise SystemExit(f"{policy_path} reads {self.policy_inputs} numbers, not the {obs_dim} "
                                 f"given by --obs_dim; leave it out and the harness takes it from the policy")
            obs_dim = self.policy_inputs
        elif obs_dim is None:
            obs_dim = self._layout_size(self.cfg["obs"])
        self.obs_dim = obs_dim

        names = self.cfg["joints"]
        self.qpos_idx = np.array([self.model.jnt_qposadr[self._joint(n)] for n in names])
        self.qvel_idx = np.array([self.model.jnt_dofadr[self._joint(n)] for n in names])
        self.act_idx = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names])
        # the pose a policy was trained around is the zero of its action and of the joint
        # positions it reads, so it has to be the one it is run with
        pose = self.cfg["default"] if nominal == "current" else self.cfg.get(f"{nominal}_default")
        if pose is None:
            raise SystemExit(f"'{robot}' has no '{nominal}' nominal pose")
        self.nominal = nominal
        self.default = np.array([pose[n] for n in names])
        self.is_wheel = np.array([n in self.cfg["wheels"] for n in names])

        # Viscous damping of the leg joints, on top of the kd of the controller. Zero is an
        # ideal PD. Isaac Lab drives the Go1 through a learned actuator whose response to a
        # velocity is that of kd = 0.76, not the 0.5 it is commanded: 0.26 reproduces it here.
        self.joint_damping = self.cfg.get("joint_damping", 0.0) if joint_damping is None else joint_damping
        self.model.dof_damping[self.qvel_idx[~self.is_wheel]] = self.joint_damping

        def per_joint(value):
            return np.array([value[n] for n in names]) if isinstance(value, dict) else np.full(len(names), value)

        self.action_scale = per_joint(self.cfg["action_scale"])
        self.kp = per_joint(self.cfg["kp"])
        self.kd = per_joint(self.cfg["kd"])
        self.stand_kp = per_joint(self.cfg["stand_kp"])
        self.stand_kd = per_joint(self.cfg["stand_kd"])
        self.damping_kd = per_joint(self.cfg["damping_kd"])
        self.rest_kp = per_joint(self.cfg["rest_kp"])
        self.rest_kd = per_joint(self.cfg["rest_kd"])
        self.obs_spec = self.cfg.get(f"obs_{obs_dim}", self.cfg["obs"])
        self._check_layout(policy_path)
        self.height_command = 0.0
        # only built when a policy asks for it: it runs a CNN+GRU at every step
        self.contact_net = ContactNet(self) if self.policy is not None and any(
            name == "contact_probabilities" for name, _ in self.obs_spec) else None
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.cfg["base_body"])

        self.actions = np.zeros(len(names))
        # What the real robot has and the model does not, off by default and set by whoever tests
        # a policy against it (evaluate.py): an encoder whose zero is not the model's, in rad
        # per joint, read by the policy and by the controller alike; and a command that reaches
        # the motors this many control steps (1 ms) late.
        self.joint_bias = np.zeros(len(names))
        self.latency_steps = 0
        self._target = None
        self.torque_limit = self.model.actuator_ctrlrange[self.act_idx, 1]

    def _load_policy(self, policy_path: Path):
        try:
            self.policy = torch.jit.load(str(policy_path)).eval()
        except RuntimeError as error:
            raise SystemExit(
                f"'{policy_path}' is not a TorchScript policy ({error}).\n"
                "This harness needs the exported policy, not a training checkpoint. Export one with:\n"
                f"  python isaaclab_experiments/play_rsl_rl.py --task <TASK> "
                f"--checkpoint {policy_path} --export --headless --max_steps 2\n"
                f"and pass {policy_path.parent / 'exported' / 'policy.pt'}"
            ) from None

    #: how many numbers each observation term is made of; the joints count once per joint
    TERM_SIZE = {"base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3,
                 "height_command": 1, "contact_probabilities": 4}

    def _layout_size(self, spec) -> int:
        joints = len(self.cfg["joints"])
        return sum(self.TERM_SIZE.get(name, joints) for name, _ in spec)

    def _check_layout(self, policy_path):
        """The layout picked for this many inputs has to add up to it, or the harness cannot feed it."""
        if self._layout_size(self.obs_spec) == self.obs_dim:
            return
        known = sorted({self._layout_size(self.cfg["obs"])} | {
            self._layout_size(v) for k, v in self.cfg.items() if k.startswith("obs_") and isinstance(v, list)})
        raise SystemExit(f"{policy_path} reads {self.obs_dim} numbers and the harness knows no layout of that "
                         f"size for the {self.cfg['scene'].stem.split('_')[0]} (it knows {known}): "
                         "add one to robot/specs.py")

    def _policy_inputs(self) -> int:
        """Size of the observation the loaded policy reads: the input of its first layer."""
        for _, weight in self.policy.named_parameters():
            if weight.ndim == 2:
                return int(weight.shape[1])
        raise SystemExit("could not tell how many inputs the policy reads; pass --obs_dim")

    def _saturate(self, torque: np.ndarray) -> np.ndarray:
        """Clip to what the motors can actually deliver, as the driver does on the robot."""
        return np.clip(torque, -self.torque_limit, self.torque_limit)

    def _joint(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint '{name}' is not in {self.cfg['scene'].name}")
        return jid

    def reset(self):
        if getattr(self, "contact_net", None) is not None:
            self.contact_net.filled = False
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_idx] = self.default
        self.data.qpos[2] = self.cfg["spawn_height"]   # drop it just above the floor
        self.actions[:] = 0.0
        self._target = None
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ states
    def base_rotation(self) -> np.ndarray:
        return self.data.xmat[self.base_id].reshape(3, 3)

    def base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Linear and angular velocity of the base, both in the base frame.

        For a free joint MuJoCo stores the linear velocity in the world frame and the
        angular velocity already in the body frame.
        """
        rot = self.base_rotation()
        return rot.T @ self.data.qvel[0:3], self.data.qvel[3:6].copy()

    def observation(self, command: np.ndarray) -> torch.Tensor:
        lin_vel_b, ang_vel_b = self.base_velocity()
        gravity_b = self.base_rotation().T @ np.array([0.0, 0.0, -1.0])

        joint_pos = self.data.qpos[self.qpos_idx] - self.joint_bias - self.default   # joint_pos_rel
        # a wheel angle carries no information, so Isaac Lab zeroes it in the observation
        joint_pos = np.where(self.is_wheel, 0.0, joint_pos)

        terms = {
            "base_lin_vel": lin_vel_b,
            "base_ang_vel": ang_vel_b,
            "projected_gravity": gravity_b,
            "velocity_commands": command,
            "height_command": np.array([self.height_command]),
            "joint_pos": joint_pos,
            "joint_vel": self.data.qvel[self.qvel_idx],            # the default velocity is zero
            "actions": self.actions,
        }
        if self.contact_net is not None:
            terms["contact_probabilities"] = self.contact_net.probabilities()
        obs = np.concatenate([terms[name] * scale for name, scale in self.obs_spec])
        if obs.size != self.obs_dim:
            raise ValueError(f"observation is {obs.size}, the policy expects {self.obs_dim}")
        return torch.from_numpy(obs).float().unsqueeze(0)

    # ------------------------------------------------------------------- loop
    def drive(self, target: np.ndarray, kp: np.ndarray, kd: np.ndarray):
        """One turn of the position control, the loop that runs at 1 kHz on the robot."""
        error = target - (self.data.qpos[self.qpos_idx] - self.joint_bias)
        self.data.ctrl[self.act_idx] = self._saturate(
            kp * error - kd * self.data.qvel[self.qvel_idx])
        mujoco.mj_step(self.model, self.data)

    def hold(self, target: np.ndarray, kp: np.ndarray, kd: np.ndarray):
        """Drive the joints to a fixed pose, as the stand-up controller of the robot does."""
        for _ in range(self.cfg["decimation"]):
            self.drive(target, kp, kd)

    def attitude(self) -> tuple[float, float]:
        """Roll and pitch of the trunk, which is what an IMU gives the controller."""
        rot = self.base_rotation()
        return float(math.atan2(rot[2, 1], rot[2, 2])), float(-math.asin(np.clip(rot[2, 0], -1.0, 1.0)))

    def lie_down(self):
        """Put the robot on the ground the way it is found before a run: switched off.

        The base is lowered until the robot just touches the floor, instead of being
        dropped from a guessed height, and it is then left under joint damping until it
        comes to rest. With the motors loose a quadruped settles onto its belly with the
        legs splayed, which is the pose the stand-up routine starts from on the robot.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_idx] = np.array([self.cfg["folded"][n] for n in self.cfg["joints"]])
        self.actions[:] = 0.0
        if getattr(self, "contact_net", None) is not None:
            self.contact_net.filled = False

        # lowest height at which nothing is pushed into the floor
        low, high = 0.0, 0.8
        for _ in range(40):
            self.data.qpos[2] = 0.5 * (low + high)
            mujoco.mj_forward(self.model, self.data)
            penetration = min((c.dist for c in self.data.contact[: self.data.ncon]), default=1.0)
            if penetration < 0.0:
                low = self.data.qpos[2]
            else:
                high = self.data.qpos[2]
        self.data.qpos[2] = high + 0.002
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # hold the folded pose until it is at rest: a session starts with the robot folded
        # and the controller holding it, not with the motors loose
        folded = np.array([self.cfg["folded"][n] for n in self.cfg["joints"]])
        for _ in range(int(3.0 / (self.model.opt.timestep * self.cfg["decimation"]))):
            self.hold(folded, self.rest_kp, self.rest_kd)

    def step(self, command: np.ndarray):
        with torch.no_grad():
            self.actions = self.policy(self.observation(command)).squeeze(0).numpy()

        target = self.default + self.action_scale * self.actions
        late = self._target if self._target is not None else target
        for k in range(self.cfg["decimation"]):
            q = self.data.qpos[self.qpos_idx] - self.joint_bias
            dq = self.data.qvel[self.qvel_idx]
            # a command that is still on its way is the previous one
            goal = late if k < self.latency_steps else target
            # legs follow a position target, wheels a velocity target (their stiffness is 0)
            torque = np.where(
                self.is_wheel,
                self.kd * (self.action_scale * self.actions - dq),
                self.kp * (goal - q) - self.kd * dq,
            )
            self.data.ctrl[self.act_idx] = self._saturate(torque)
            mujoco.mj_step(self.model, self.data)
        self._target = target
