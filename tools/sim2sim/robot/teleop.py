"""Driving the robot from the keyboard, and the key map that says how."""

from __future__ import annotations

import math
import time

import numpy as np

class Teleop:
    """Keyboard driving, like a joystick on the real robot.

    The commands are clamped to the range the policy was trained on: asking for more than
    that is outside its experience and says nothing about the robot.

    Three ways to drive, chosen with ``--control``:

    ``sticky``
        every press adds a step to the command, which then stays until it is changed.
        Convenient to hold one velocity for a long run.
    ``hold``
        a key sends a fixed command while it is pressed, and zero when it is let go.
    ``ramp``
        the command grows towards the trained limit while the key is pressed and falls
        back to zero when it is let go, which is what a joystick feels like.

    ``hold`` and ``ramp`` tell a held key from a released one by its auto-repeat: the
    window system keeps resending a key that is down, and the command is let go once the
    repeats stop. The first repeat only arrives after the auto-repeat delay, so a short
    tap is a short pulse of the command.
    """

    MODES = ("sticky", "hold", "ramp")
    HOLD_GRACE = 0.6          # keep the command this long after a press, for the repeat delay
    MIN_RELEASE = 0.12        # but never let go sooner than this
    RELEASE_AFTER = 3.0       # a key is released once this many repeats are missed
    RAMP_TIME = 1.2           # seconds from zero to the full command, in 'ramp'
    RETURN_TIME = 0.4         # seconds back to zero once the key is released

    KEYS = {
        87:  ("vx", +0.5),     83:  ("vx", -0.5),      # W / S
        265: ("vx", +0.5),     264: ("vx", -0.5),      # up / down
        65:  ("vy", +0.5),     68:  ("vy", -0.5),      # A / D
        81:  ("wz", +0.5),     69:  ("wz", -0.5),      # Q / E
        263: ("wz", +0.5),     262: ("wz", -0.5),      # left / right
        82:  ("height", +1), 70: ("height", -1),  # R / F
    }
    POSTURE_KEYS = {
        87: (0, +1), 83: (0, -1),      # W / S   trunk forwards / backwards
        265: (0, +1), 264: (0, -1),    # up / down
        65: (1, +1), 68: (1, -1),      # A / D   trunk left / right
        82: (2, +1), 70: (2, -1),      # R / F   trunk up / down
        89: (3, +1), 72: (3, -1),      # Y / H   roll left / right
        84: (4, +1), 71: (4, -1),      # T / G   pitch down / up
        81: (5, +1), 69: (5, -1),      # Q / E   turn left / right
        263: (5, +1), 262: (5, -1),    # left / right
    }
    STEP = {"vx": 0.1, "vy": 0.1, "wz": 0.1, "height": 0.01}
    #: one press of a posture key, and how far the trunk may be asked to go: metres of
    #: travel along (x, y, z) and radians of turn about them
    POSTURE_STEP = np.array([0.01, 0.01, 0.01, 0.03, 0.03, 0.04])
    POSTURE_LIMIT = np.array([0.08, 0.05, 0.07, 0.25, 0.30, 0.40])

    #: 1 2 3 4 5 6
    STATE_KEYS = {49: "damping", 50: "stand", 51: "policy", 52: "sit", 53: "folded", 54: "walk"}

    def __init__(self, cfg, height: float, mode: str = "sticky"):
        if mode not in self.MODES:
            raise ValueError(f"control mode '{mode}' is not one of {self.MODES}")
        self.mode = mode
        self.limits = cfg["command_limits"]
        self.height_limits = cfg["height_limits"]
        self.command = np.zeros(3)
        self.height = height
        self.stop = False
        self.reset = False
        self.changed = True
        self.requested_state = None
        self.posture = np.zeros(6)
        self.posture_mode = False       # set while the Unitree controller holds the stand
        self.pressed = np.zeros(3)      # direction of the key held on each axis
        self.last_event = np.full(3, -np.inf)
        self.repeat = np.full(3, np.inf)  # measured auto-repeat interval of each axis

    def key_callback(self, keycode: int):
        if keycode in self.STATE_KEYS:        # 1-4: the states of the controller
            self.requested_state = self.STATE_KEYS[keycode]
            self.release()
        elif keycode == 32:                   # space: stand still, or trunk back to neutral
            self.posture[:] = 0.0
            self.release()
        elif keycode == 48:                   # 0: back to the start pose
            self.reset = True
        elif keycode in (256, 88):            # escape or X: quit
            self.stop = True
        elif self.posture_mode and keycode in self.POSTURE_KEYS:
            axis, sign = self.POSTURE_KEYS[keycode]
            self.posture[axis] = float(np.clip(self.posture[axis] + sign * self.POSTURE_STEP[axis],
                                               -self.POSTURE_LIMIT[axis], self.POSTURE_LIMIT[axis]))
        elif keycode in self.KEYS:
            axis, sign = self.KEYS[keycode]
            if axis == "height":
                if self.height_limits is None:
                    return
                low, high = self.height_limits
                self.height = float(np.clip(self.height + sign * self.STEP[axis], low, high))
            elif self.mode == "sticky":
                index = {"vx": 0, "vy": 1, "wz": 2}[axis]
                limit = self.limits[index]
                self.command[index] = float(np.clip(
                    self.command[index] + sign * self.STEP[axis], -limit, limit))
            else:
                self.press({"vx": 0, "vy": 1, "wz": 2}[axis], sign)
        else:
            return
        self.changed = True

    def press(self, index: int, sign: float):
        """Record a key event of a driving axis, with the interval to the previous one.

        Only the direction of the key matters here: how big a step it is worth belongs to
        'sticky', while a key held down in the other modes always asks for the full command.
        """
        sign = math.copysign(1.0, sign)
        now = time.monotonic()
        gap = now - self.last_event[index]
        if self.pressed[index] != sign:
            gap = np.inf                 # a different direction: not a repeat of this key
        self.pressed[index] = sign
        self.last_event[index] = now
        # only an interval short enough to be an auto-repeat says the key is still down
        self.repeat[index] = gap if gap < self.HOLD_GRACE else np.inf

    def release(self):
        """Let every axis go and zero the command."""
        self.command[:] = 0.0
        self.pressed[:] = 0.0
        self.last_event[:] = -np.inf
        self.repeat[:] = np.inf

    def update(self, dt: float):
        """Advance a continuously driven command; does nothing in 'sticky' mode."""
        if self.mode == "sticky":
            return
        now = time.monotonic()
        for index in range(3):
            limit = self.limits[index]
            interval = self.repeat[index]
            timeout = (self.HOLD_GRACE if not np.isfinite(interval)
                       else min(self.HOLD_GRACE, max(self.MIN_RELEASE, self.RELEASE_AFTER * interval)))
            held = (now - self.last_event[index]) < timeout
            target = self.pressed[index] * limit if held else 0.0
            if self.mode == "hold":
                value = target
            else:
                rate = limit / (self.RAMP_TIME if target != 0.0 else self.RETURN_TIME)
                value = self.command[index] + np.clip(target - self.command[index], -rate * dt, rate * dt)
            value = float(np.clip(value, -limit, limit))
            if abs(value - self.command[index]) > 1e-9:
                self.command[index] = value
                self.changed = True

    def banner(self) -> str:
        if self.posture_mode:
            x, y, z = self.posture[:3] * 100.0
            roll, pitch, yaw = (math.degrees(a) for a in self.posture[3:])
            return (f"  trunk  x {x:+3.0f} y {y:+3.0f} z {z:+3.0f} cm   "
                    f"roll {roll:+3.0f} pitch {pitch:+3.0f} yaw {yaw:+3.0f}°   (Unitree control)")
        height = f"   height {self.height:.2f} m" if self.height_limits else ""
        return (f"  vx {self.command[0]:+.2f}   vy {self.command[1]:+.2f}   "
                f"wz {self.command[2]:+.2f}{height}")


HELP = """
  controller                              (as on the robot: stand up before walking)
    1   damping    release the motors (the robot flops down, as a real one does)
    5   folded     hold the folded pose it starts in
    2   stand      fixed-gain stand-up, like the built-in controller
    3   policy     hand the legs over to the trained policy, and back again
    6   walk       move around without a policy, as pressing start on the robot does
    4   sit        fold the legs back down

  the trunk, under the Unitree position control of the stand (state 2)
    W / S  or  up / down     move forwards / backwards
    A / D                    move left / right
    R / F                    move up / down
    Y / H                    roll left / right
    T / G                    pitch down / up
    Q / E  or  left / right  turn left / right
    space                    back to the standing pose

  teleoperation, once the legs are walking (state 3 or state 6)
    W / S  or  up / down     forward speed
    A / D                    sideways speed
    Q / E  or  left / right  turn
    R / F                    base height        (Go2W only)
    space                    stand still
    0                        reset the robot
    X or escape              quit

  keys are read from this terminal, so the shortcuts of the viewer stay out of the way:
  keep the terminal focused while driving (--keys viewer reads the viewer window instead)

  the speeds asked for are clamped to what the one in charge of the legs can take: what
  the gait of state 6 was tuned for, or, for a policy, a moderate part of the range it was
  trained on, which --speed opens up

  --control sticky   each press adds a step; the command stays until it is changed
  --control hold     a key commands a fixed speed while it is held, zero when let go
  --control ramp     the speed grows while the key is held and falls back when let go
"""
