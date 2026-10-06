"""The controller of the robot: the states it goes through and the poses it holds."""

from __future__ import annotations

import numpy as np

from .gait import locomotion
from .kinematics import Legs, euler_matrix


class Controller:
    """The states a Unitree robot goes through, reproduced in simulation.

    On the real robot the policy never starts from a robot lying on the floor: the
    built-in controller first brings it to its standing pose with fixed gains, and only
    then is the control handed over. The same sequence here keeps the sim-to-sim test
    close to what deployment looks like.

        DAMPING -> STAND -> POLICY -> SIT -> DAMPING

    From the stand the legs can go either way: to the trained policy, or to the walking
    this harness does without one, which is the ``WALK`` state.
    """

    FOLDED, DAMPING, STAND, POLICY, SIT = "folded", "damping", "stand", "policy", "sit"
    WALK = "walk"
    #: the states in which the robot moves on a velocity command
    DRIVEN = (POLICY, WALK)
    SIT_TIME = 1.5
    HANDBACK_TIME = 1.0                           # seconds to take the legs back
    HANDOVER_TIME = 1.0                           # seconds to square up before the policy
    #: what counts as standing square: a posture command this small, and no joint further
    #: than this from the standing pose. The fixed gains sag a little under gravity, so a
    #: joint is never exactly where it was put
    POSTURE_NEUTRAL = np.array([0.005, 0.005, 0.005, 0.02, 0.02, 0.02])
    HANDOVER_TOLERANCE = 0.15
    #: how fast the trunk follows a posture command: m/s of travel, rad/s of turn
    POSTURE_RATE = np.array([0.08, 0.08, 0.08, 0.35, 0.35, 0.45])

    def __init__(self, sim: "Sim2Sim"):
        self.sim = sim
        self.state = self.FOLDED
        self.standing = np.array([sim.cfg["default"][n] for n in sim.cfg["joints"]])
        self.folded = np.array([sim.cfg["folded"][n] for n in sim.cfg["joints"]])
        self.poses = {
            "standing": self.standing,
            "folded": self.folded,
            "fix_stand": np.array([sim.cfg["fix_stand"][n] for n in sim.cfg["joints"]]),
        }
        self.ramp_from = self.folded.copy()
        self.wheel_hold = self.folded.copy()
        #: where the trunk is asked to be while the Unitree controller holds the stand:
        #: (x, y, z) in metres and (roll, pitch, yaw) in radians, off the standing pose
        self.posture_target = np.zeros(6)
        self.posture = np.zeros(6)
        self.speed = 1.0            # how much of the driving range the keyboard may use
        self.legs = Legs(sim, self.poses["fix_stand"])
        self.gait = locomotion(self.legs)
        self.ik_pose = self.poses["fix_stand"].copy()   # warm start of the leg solver
        self.waypoints: list[tuple[str, float]] = []
        self.pending: str | None = None           # state to enter once the ramp is over
        self.waypoint = 0
        self.progress = 0.0
        self.dt = sim.model.opt.timestep * sim.cfg["decimation"]

    def stand_pose(self) -> np.ndarray:
        """The joint angles that hold the trunk at the commanded posture.

        The feet stay where they are planted and the trunk is asked to move off its
        standing pose: lower or higher, forwards, sideways, and turned about any of its
        three axes. Each foot is therefore wanted at a different place as seen from the
        trunk, which the leg solver turns into joint angles. This is the body posture
        control of the Unitree demos, under the same fixed-gain position control.
        """
        offset, rotation = self.posture[:3], euler_matrix(*self.posture[3:])
        targets = {leg: rotation.T @ (foot - offset) for leg, foot in self.legs.stance.items()}
        self.ik_pose = self.legs.solve(targets, self.ik_pose)
        pose = self.poses["fix_stand"].copy()
        for index in self.legs.joints.values():
            pose[index] = self.ik_pose[index]
        return pose

    @property
    def ready_to_walk(self) -> bool:
        return (self.state == self.STAND and self.waypoint >= len(self.waypoints) - 1
                and self.progress >= 1.0)

    @property
    def command_limits(self) -> tuple[float, float, float]:
        """What the robot may be asked for, which depends on what has the legs.

        The gait has the range it was tuned for. A policy has the range it was trained on,
        of which driving by hand uses a moderate part: the whole of it is far more speed
        than the gait can give, and the two then behave nothing like each other.
        """
        if self.state == self.WALK:
            return self.gait.LIMITS
        trained = np.array(self.sim.cfg["command_limits"])
        return tuple(np.minimum(np.array(self.sim.cfg["drive_limits"]) * self.speed, trained))

    @property
    def squared_up(self) -> bool:
        """True when the robot holds the standing pose the policy was trained around."""
        if np.any(np.abs(self.posture) > self.POSTURE_NEUTRAL):
            return False
        error = self.sim.data.qpos[self.sim.qpos_idx] - self.poses["fix_stand"]
        return bool(np.all(np.abs(error[~self.sim.is_wheel]) < self.HANDOVER_TOLERANCE))

    def request(self, state: str) -> str:
        """Accept a state change if the robot can legally make it, as the FSM does."""
        taking_over = state == self.STAND and self.state in self.DRIVEN
        if state == self.POLICY and self.sim.policy is None:
            return "no policy was loaded: pass --policy to hand the legs over"
        if state in self.DRIVEN and not self.ready_to_walk:
            return "stand up first (walking starts from a standing robot)"
        if state in self.DRIVEN and not self.squared_up:
            # the policy was trained around the standing pose: bring the trunk back to it
            # before handing the legs over, instead of starting it from a leaning robot
            self.posture_target = np.zeros(6)
            self.posture = np.zeros(6)
            self.ramp_from = self.sim.data.qpos[self.sim.qpos_idx].copy()
            self.progress = 0.0
            self.waypoint = 0
            self.waypoints = [("fix_stand", self.HANDOVER_TIME)]
            self.pending = state
            self.state = self.STAND
            return f"squaring up to the standing pose, then {state} takes over"
        if state in (self.STAND, self.SIT):
            self.ramp_from = self.sim.data.qpos[self.sim.qpos_idx].copy()
            self.progress = 0.0
            self.waypoint = 0
            if state == self.SIT:
                self.waypoints = [("folded", self.SIT_TIME)]
            elif taking_over:
                # the robot is on its feet already: go straight to the stand pose instead
                # of through the folded one the stand-up routine starts from
                self.waypoints = [("fix_stand", self.HANDBACK_TIME)]
            else:
                self.waypoints = list(self.sim.cfg["stand_waypoints"])
        if state == self.POLICY:
            self.sim.actions[:] = 0.0          # the policy starts from a zero action
        if state == self.WALK:
            self.gait.reset()
            self.ik_pose = self.poses["fix_stand"].copy()
        self.pending = None
        self.state = state
        if state == self.STAND and taking_over:
            return "state: stand (the Unitree controller has the legs back)"
        return f"state: {state}"

    def update(self, command: np.ndarray) -> str | None:
        """One control cycle in the current state; returns a message when it changes by itself."""
        cfg = self.sim.cfg
        # walk the posture to the commanded one rather than stepping it there, so the
        # legs are never asked for a jump in position
        step = self.POSTURE_RATE * self.dt
        self.posture += np.clip(self.posture_target - self.posture, -step, step)
        if self.state == self.FOLDED:
            # the robot waits in its folded pose, held by the controller
            self.sim.hold(self.folded, self.sim.rest_kp, self.sim.rest_kd)
        elif self.state == self.DAMPING:
            # the Passive state of the robot: the motors are released, so it flops down
            self.sim.hold(self.sim.data.qpos[self.sim.qpos_idx].copy(),
                          np.zeros_like(self.sim.damping_kd), self.sim.damping_kd)
        elif self.state in (self.STAND, self.SIT):
            name, duration = self.waypoints[self.waypoint]
            self.progress = min(1.0, self.progress + self.dt / duration)
            goal = self.stand_pose() if name == "fix_stand" else self.poses[name]
            self.sim.hold(self.ramp_from + (goal - self.ramp_from) * self.progress,
                          self.sim.stand_kp, self.sim.stand_kd)
            if self.progress >= 1.0:
                if self.waypoint < len(self.waypoints) - 1:
                    self.waypoint += 1
                    self.progress = 0.0
                    self.ramp_from = self.sim.data.qpos[self.sim.qpos_idx].copy()
                elif self.state == self.SIT:
                    self.state = self.FOLDED
                    return "state: folded (the robot is down)"
                elif self.pending is not None:
                    wanted, self.pending = self.pending, None
                    return self.request(wanted)
        elif self.state == self.POLICY:
            self.sim.step(command)
            if self.sim.data.qpos[2] < cfg["fall_height"]:
                self.state = self.DAMPING
                return "the robot fell: motors released"
        elif self.state == self.WALK:
            self.ik_pose = self.gait.cycle(self.sim, command, self.poses["fix_stand"], self.ik_pose)
            if self.sim.data.qpos[2] < cfg["fall_height"]:
                self.state = self.DAMPING
                return "the robot fell: motors released"
        else:
            # never drive the legs from a state that is not known
            raise RuntimeError(f"unknown controller state '{self.state}'")
        return None
