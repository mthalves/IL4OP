"""Sim-to-sim validation: run an IL4OP policy in MuJoCo.

The policy is the TorchScript file exported by ``play_rsl_rl.py --export``. This harness
rebuilds, in MuJoCo, exactly the observation the policy saw in Isaac Lab and closes the
loop with the same control rate, so a policy that behaves here is one step closer to the
robot. What differs on purpose is the physics engine and the actuator model, which is the
point of the exercise.

    python tools/sim2sim/play.py --robot go1 \\
        --policy logs/rsl_rl/unitree_go1_rough/20k_no_air_feet/exported/policy.pt \\
        --command 1.0 0 0 --duration 10

Add ``--viewer`` to watch it (needs a display), or ``--teleop`` to drive the robot
yourself through the states a Unitree robot goes through: it starts folded on the floor,
the controller stands it up and holds whatever posture it is given, and the legs are
handed over only when asked -- to the trained policy, or to the walking this harness does
without one. Neither needs the other, so ``--teleop`` runs without a policy at all:

    python tools/sim2sim/play.py --robot go1 --teleop --control ramp

The parts live in ``robot/``: ``specs`` what each robot is, ``simulation`` the model and
its motors, ``kinematics`` where a leg can reach, ``controller`` the states the robot goes
through, ``gait`` getting around without a policy, ``contact`` the contact estimator, and
``teleop``, ``keyboard`` and ``session`` the driving.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robot import ROBOTS, Sim2Sim, Teleop, teleoperate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="go1", choices=sorted(ROBOTS))
    parser.add_argument("--policy", type=Path,
                        help="exported policy.pt; without it only the Unitree controller runs")
    parser.add_argument("--obs_dim", type=int, default=48, help="input size of the policy")
    parser.add_argument("--height", type=float, default=0.33,
                        help="base height command of the Go2W flat-z policies (0.25 - 0.40)")
    parser.add_argument("--command", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        metavar=("VX", "VY", "WZ"), help="base velocity command")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds to simulate")
    parser.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer")
    parser.add_argument("--teleop", action="store_true",
                        help="drive the robot from the keyboard")
    parser.add_argument("--control", default="sticky", choices=Teleop.MODES,
                        help="how the keyboard drives the robot (see the key map)")
    parser.add_argument("--keys", default="terminal", choices=("terminal", "viewer"),
                        help="where the keys are read from; the viewer binds most letters itself")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="scale the speeds the keyboard may ask a policy for, up to the"
                             " range it was trained on (1.0 is near what the gait does)")
    args = parser.parse_args()

    if args.policy is None and not args.teleop:
        parser.error("--policy is needed for a scripted run; --teleop works without one")

    sim = Sim2Sim(args.robot, args.policy, args.obs_dim)
    sim.height_command = args.height
    sim.reset()

    if args.teleop:
        teleoperate(sim, args.height, args.control, args.keys, args.speed)
        return

    command = np.array(args.command)

    dt = sim.model.opt.timestep * sim.cfg["decimation"]
    steps = int(args.duration / dt)
    heights, vel_error, torques, yaw_rate = [], [], [], []
    start_xy = sim.data.qpos[:2].copy()
    fell_at = None

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    for i in range(steps):
        sim.step(command)
        lin_vel_b, ang_vel_b = sim.base_velocity()
        heights.append(sim.data.qpos[2])
        vel_error.append(abs(lin_vel_b[0] - command[0]))
        yaw_rate.append(ang_vel_b[2])
        torques.append(np.abs(sim.data.ctrl[sim.act_idx]).mean())
        if fell_at is None and sim.data.qpos[2] < sim.cfg["fall_height"]:
            fell_at = i * dt
        if viewer is not None:
            viewer.sync()

    if viewer is not None:
        viewer.close()

    print(f"\npolicy   : {args.policy}")
    print(f"robot    : {args.robot}   command: vx={command[0]} vy={command[1]} wz={command[2]}")
    print(f"simulated: {steps * dt:.1f} s at {1 / dt:.0f} Hz")
    print("-" * 56)
    if fell_at is None:
        print(f"  stayed up for the whole run")
    else:
        print(f"  FELL at {fell_at:.2f} s (base below {sim.cfg['fall_height']} m)")
    print(f"  base height   mean {np.mean(heights):.3f} m   min {np.min(heights):.3f} m")
    print(f"  |vx - cmd|    mean {np.mean(vel_error):.3f} m/s")
    print(f"  |torque|      mean {np.mean(torques):.2f} Nm   max {np.max(torques):.2f} Nm")
    travelled = sim.data.qpos[:2] - start_xy
    print(f"  travelled     {travelled[0]:+.2f} m forward, {travelled[1]:+.2f} m sideways")
    print(f"  yaw rate      mean {np.mean(yaw_rate):+.3f} rad/s (command {command[2]:+.2f})")


if __name__ == "__main__":
    main()
