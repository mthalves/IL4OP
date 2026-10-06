"""Walking without a learned policy: a trot, in the spirit of the built-in controller.

Unitree does not ship the walking controller of these robots. What the SDK contains is the
client that asks for it -- ``sport_client.Move(vx, vy, vyaw)``, ``SwitchGait(1)`` -- and the
service that answers runs closed on the robot itself; ``unitree_ros/unitree_controller`` has
only the posture demos. So the gait here is written for this harness rather than taken from
Unitree, and it is the classical one: the diagonal pairs of legs take turns, a swinging foot
is thrown to where the commanded velocity says it should land, and the standing feet are
drawn backwards underneath the robot, which is what carries the body forwards.

It places the legs through the same inverse kinematics and holds them with the same
fixed-gain position control the controller stands with, so what the motors are asked for is
honest. What it does not have is any force control, and it never looks at where the ground
is, so it is a flat-floor gait at moderate speed and nothing more.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .kinematics import euler_matrix


class Locomotion:
    """What the ways of getting around have in common: a command, and what came of it.

    Neither of them can make the robot travel exactly as asked on the first try -- feet
    slip through a stance, wheels scrub through a turn -- so both watch what the robot
    actually did and add the difference back into the command.
    """

    #: what it may be asked for: m/s forwards, m/s sideways, rad/s of turn
    LIMITS = (0.0, 0.0, 0.0)
    VELOCITY_GAIN = 1.0
    #: how far the command may be trimmed to get what was asked for
    VELOCITY_TRIM = np.array([0.3, 0.3, 0.5])


    def __init__(self, legs):
        self.legs = legs
        self.trim = np.zeros(3)

    def reset(self):
        self.trim = np.zeros(3)

    @staticmethod
    def measured(sim) -> np.ndarray:
        """How fast the robot is really travelling and turning, in its own frame."""
        linear, angular = sim.base_velocity()
        return np.array([linear[0], linear[1], angular[2]])

    def asked_for(self, command: np.ndarray, sim, dt: float) -> np.ndarray:
        """The command to drive with: the one wanted, plus what it has been short by."""
        wanted = np.clip(command, [-l for l in self.LIMITS], list(self.LIMITS))
        # while a motor is at its limit the robot cannot deliver more than it already is,
        # so the trim is left where it is rather than wound further into the saturation
        if not np.any(np.abs(sim.data.ctrl[sim.act_idx]) >= sim.torque_limit - 1e-6):
            short = self.VELOCITY_GAIN * (wanted - self.measured(sim))
            self.trim = np.clip(self.trim + short * dt, -self.VELOCITY_TRIM, self.VELOCITY_TRIM)
        return wanted + self.trim

    def cycle(self, sim, command: np.ndarray, stand: np.ndarray, start: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Trot(Locomotion):
    """The gait: a phase, four feet, and where each of them is going."""

    PERIOD = 0.3                  # seconds of a full cycle
    DUTY = 0.5                    # of it, the fraction a foot spends on the ground
    STEP_HEIGHT = 0.06            # how far a foot is lifted, in metres
    #: the diagonal pairs move together, half a cycle apart
    OFFSET = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}
    LIMITS = (0.6, 0.3, 0.8)
    #: a foot is thrown this much further than the velocity alone suggests, which is what
    #: keeps the robot from walking out from under itself
    REACH = 1.1
    #: how hard the standing legs push the trunk back towards level
    ATTITUDE_GAIN = 1.5


    def __init__(self, legs, stance: dict[str, np.ndarray] | None = None):
        super().__init__(legs)
        self.nominal = {leg: foot.copy() for leg, foot in (stance or legs.stance).items()}
        self.reset()

    def reset(self):
        super().reset()
        self.phase = 0.0
        self.foot = {leg: foot.copy() for leg, foot in self.nominal.items()}
        self.lift_from = {leg: foot.copy() for leg, foot in self.nominal.items()}
        self.airborne = dict.fromkeys(self.nominal, False)

    @property
    def stance_time(self) -> float:
        return self.PERIOD * self.DUTY

    def _landing(self, leg: str, command: np.ndarray) -> np.ndarray:
        """Where a foot should come down to carry the commanded velocity for a stance.

        Half a stance of travel ahead of where the leg stands, so that the foot spends the
        stance sweeping from in front of the hip to behind it: the step of Raibert.
        """
        nominal = self.nominal[leg]
        travel = np.array([command[0], command[1], 0.0]) * self.stance_time * 0.5 * self.REACH
        turn = np.cross([0.0, 0.0, command[2]], nominal) * self.stance_time * 0.5 * self.REACH
        return nominal + travel + turn

    def tick(self, command: np.ndarray, dt: float, attitude: tuple[float, float]) -> dict:
        """Advance the gait by ``dt`` and say where every foot is wanted.

        A gait made of nothing but foot positions has no reason to stay level, so the whole
        foot pattern is turned with the measured tilt of the trunk: the standing legs then
        push the trunk back the other way, which is a proportional correction of its
        attitude through the geometry of the legs. Rotating the pattern the opposite way is
        the frame transform that keeps the legs plumb, and it makes the robot walk nose-up
        at fourteen degrees instead.
        """
        # what the command may have grown to once the trim has been added to it
        reach = np.array(self.LIMITS) + self.VELOCITY_TRIM
        command = np.clip(command, -reach, reach)
        self.phase = (self.phase + dt / self.PERIOD) % 1.0
        swing_time = self.PERIOD * (1.0 - self.DUTY)

        for leg, nominal in self.nominal.items():
            phase = (self.phase - self.OFFSET[leg]) % 1.0
            if phase < self.DUTY:
                # on the ground: the foot is carried backwards under the body, which is the
                # same motion as the body travelling forwards over a planted foot
                turn = euler_matrix(0.0, 0.0, -command[2] * dt)
                self.foot[leg] = turn @ (self.foot[leg] - np.array([command[0], command[1], 0.0]) * dt)
                self.airborne[leg] = False
            else:
                if not self.airborne[leg]:
                    self.lift_from[leg] = self.foot[leg].copy()
                    self.airborne[leg] = True
                swing = (phase - self.DUTY) / (1.0 - self.DUTY)
                # smooth in and out of the step, so the foot leaves and lands slowly
                eased = 0.5 - 0.5 * math.cos(math.pi * swing)
                landing = self._landing(leg, command)
                self.foot[leg] = self.lift_from[leg] + (landing - self.lift_from[leg]) * eased
                self.foot[leg][2] = nominal[2] + self.STEP_HEIGHT * math.sin(math.pi * swing)

        roll, pitch = attitude
        level = euler_matrix(self.ATTITUDE_GAIN * roll, self.ATTITUDE_GAIN * pitch, 0.0)
        return {leg: level @ foot for leg, foot in self.foot.items()}

    def cycle(self, sim, command: np.ndarray, stand: np.ndarray, start: np.ndarray) -> np.ndarray:
        """One control cycle: the feet move the whole way through it.

        The gait is stepped at the rate the position control runs at, not once per cycle,
        because a foot in the air covers a good part of its step in between.
        """
        timestep = sim.model.opt.timestep
        pose = start
        for _ in range(sim.cfg["decimation"]):
            asked = self.asked_for(command, sim, timestep)
            solved = self.legs.solve(self.tick(asked, timestep, sim.attitude()), pose)
            pose = stand.copy()
            for index in self.legs.joints.values():
                pose[index] = solved[index]
            sim.drive(pose, sim.stand_kp, sim.stand_kd)
        return pose


class WheelDrive(Locomotion):
    """Driving a wheeled-legged robot, which is what its own controller does.

    A Go2W does not trot to get around: it stands on its legs and drives its wheels, so
    that is what this does. The legs hold the standing pose under the fixed gains and the
    wheels are given the speeds that add up to the commanded travel and turn, the ordinary
    differential drive. Sideways travel is not among them, as the wheels do not steer.
    """

    #: m/s forwards, no sideways, rad/s of turn. Turning is what limits this robot: four
    #: wheels that do not steer have to scrub their way around a turn, and the wheel
    #: motors run out of torque doing it before the geometry runs out of turn
    LIMITS = (1.0, 0.0, 0.4)
    VELOCITY_TRIM = np.array([0.3, 0.0, 1.0])

    def __init__(self, legs, stance: dict[str, np.ndarray] | None = None):
        super().__init__(legs)
        sim = legs.sim
        self.stance = stance or legs.stance
        self.left = np.array([n.startswith(("FL", "RL")) for n in sim.cfg["joints"]]) & sim.is_wheel
        self.right = np.array([n.startswith(("FR", "RR")) for n in sim.cfg["joints"]]) & sim.is_wheel
        self.track = float(np.mean([abs(foot[1]) for foot in self.stance.values()])) * 2.0
        self.radius = self._wheel_radius()

    def _wheel_radius(self) -> float:
        """The radius of a wheel, measured on the model."""
        sim = self.legs.sim
        model = sim.model
        wheel = sim.cfg["wheels"][0].replace("_joint", "")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wheel)
        geoms = range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body])
        return float(max(model.geom_size[g][0] for g in geoms))

    def cycle(self, sim, command: np.ndarray, stand: np.ndarray, start: np.ndarray) -> np.ndarray:
        timestep = sim.model.opt.timestep
        speed = np.zeros(len(sim.cfg["joints"]))
        for _ in range(sim.cfg["decimation"]):
            asked = self.asked_for(command, sim, timestep)
            speed[self.left] = (asked[0] - asked[2] * self.track / 2.0) / self.radius
            speed[self.right] = (asked[0] + asked[2] * self.track / 2.0) / self.radius
            error = stand - sim.data.qpos[sim.qpos_idx]
            velocity_error = speed - sim.data.qvel[sim.qvel_idx]
            # the legs are held in place, the wheels are asked for a speed
            torque = np.where(sim.is_wheel,
                              sim.stand_kd * velocity_error,
                              sim.stand_kp * error - sim.stand_kd * sim.data.qvel[sim.qvel_idx])
            sim.data.ctrl[sim.act_idx] = sim._saturate(torque)
            mujoco.mj_step(sim.model, sim.data)
        return stand


def locomotion(legs):
    """The way this robot gets around: a trot on legs, wheels where it has them."""
    return WheelDrive(legs) if legs.sim.cfg["wheels"] else Trot(legs)
