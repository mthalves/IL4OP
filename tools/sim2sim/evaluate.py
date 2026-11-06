"""Put an exported policy through a battery of commands and of things that differ on the robot.

    python tools/sim2sim/evaluate.py --policy <run>/exported/policy.pt [--obs_dim 49]

Every condition runs the same commands -- standing, forwards, sideways, turning and a mix -- on a
flat floor and reports the worst case over them, so that policies, and variants of one policy, are
compared on the same numbers. The conditions are the ones a real Go1 differs from the model by:

  nominal    the harness as it is (an ideal PD at the published gains)
  damped     + 0.26 N m s/rad of joint damping, the response of the actuator network the policy is
             trained with; compare against the numbers of Isaac Lab in this one
  heavy      damped, with 2 kg more on the trunk (a battery or a payload)
  slippery   damped, on a floor with a friction coefficient of 0.4
  offset     damped, with encoders whose zero is 0.03 rad off the model's, a sign per joint
  late       damped, with the command reaching the motors 20 ms late (training: up to 10 ms)

A condition PASSES if the robot stays up through every command. Nothing else is judged: the other
columns are there to be read, and to be compared between policies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from robot import Sim2Sim

COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "fwd0.5": (0.5, 0.0, 0.0),
    "fwd1.0": (1.0, 0.0, 0.0),
    "back0.5": (-0.5, 0.0, 0.0),
    "lat0.4": (0.0, 0.4, 0.0),
    "yaw0.8": (0.0, 0.0, 0.8),
    "diag": (0.5, 0.3, 0.5),
}

DAMPING = 0.26
CONDITIONS = {
    "nominal": dict(damping=0.0),
    "damped": dict(damping=DAMPING),
    "heavy": dict(damping=DAMPING, extra_mass=2.0),
    "slippery": dict(damping=DAMPING, friction=0.4),
    "offset": dict(damping=DAMPING, offset=0.03),
    "late": dict(damping=DAMPING, latency_ms=20),
}


def run(policy, obs_dim, nominal, command, duration, damping=0.0, extra_mass=0.0, friction=None,
        offset=0.0, latency_ms=0, seed=0):
    """One rollout. Returns the numbers that describe it."""
    sim = Sim2Sim("go1", policy, obs_dim, nominal, joint_damping=damping)
    if extra_mass:
        sim.model.body_mass[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")] += extra_mass
    if friction is not None:
        sim.model.geom_friction[:, 0] = friction
    if offset:
        sim.joint_bias = offset * np.random.default_rng(seed).choice([-1.0, 1.0], size=len(sim.joint_bias))
    sim.latency_steps = latency_ms
    sim.reset()

    command = np.array(command)
    dt = sim.model.opt.timestep * sim.cfg["decimation"]
    foot = [mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot") for leg in ("FL", "FR", "RL", "RR")]
    heights, tilts, track, torques, peak, speed, rates, jitter, airborne = [], [], [], [], [], [], [], [], []
    previous, q_hist, fell = np.zeros(12), [], None
    for i in range(int(duration / dt)):
        sim.step(command)
        linear, _ = sim.base_velocity()
        gravity = sim.base_rotation().T @ np.array([0.0, 0.0, -1.0])
        tilts.append(np.degrees(np.arccos(np.clip(-gravity[2], -1.0, 1.0))))
        heights.append(sim.data.qpos[2])
        track.append(np.linalg.norm(linear[:2] - command[:2]))
        applied = np.abs(sim.data.ctrl[sim.act_idx])
        torques.append(applied.mean())
        peak.append((applied / sim.torque_limit).max())
        speed.append(np.abs(sim.data.qvel[sim.qvel_idx]).max())
        rates.append(np.square(sim.actions - previous).sum())
        previous = sim.actions.copy()
        q_hist.append(sim.data.qpos[sim.qpos_idx].copy())
        down = {foot.index(sim.model.geom_bodyid[g]) for c in sim.data.contact[: sim.data.ncon]
                for g in (c.geom1, c.geom2) if sim.model.geom_bodyid[g] in foot}
        airborne.append(4 - len(down))
        if sim.data.qpos[2] < sim.cfg["fall_height"] or tilts[-1] > 60.0:
            fell = i * dt
            break
    q = np.array(q_hist)
    return dict(
        fell=fell, height=np.mean(heights), tilt=np.max(tilts), tracking=np.mean(track),
        torque=np.mean(torques), peak_torque=np.max(peak), joint_speed=np.max(speed),
        action_rate=np.mean(rates), jitter=1e3 * np.abs(np.diff(q, n=2, axis=0)).mean() if len(q) > 3 else 0.0,
        airborne=np.mean(airborne),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", type=Path, required=True, help="exported policy.pt")
    parser.add_argument("--obs_dim", type=int, default=45, help="49 for a policy that observes ContactNet")
    parser.add_argument("--nominal", default="current", choices=("current", "legacy"))
    parser.add_argument("--duration", type=float, default=8.0, help="seconds per command")
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--commands", nargs="*", default=list(COMMANDS), choices=list(COMMANDS))
    parser.add_argument("--detail", action="store_true", help="one line per command instead of the worst case")
    args = parser.parse_args()

    print(f"policy: {args.policy}   inputs: {args.obs_dim}   {args.duration:g} s per command")
    header = (f"{'':10s}{'fell':>6s}{'tilt max':>9s}{'|dv| m/s':>9s}{'torque':>7s}{'peak/lim':>9s}"
              f"{'dq max':>7s}{'dA2':>6s}{'jitter':>7s}{'air':>5s}")
    verdicts = {}
    for name in args.conditions:
        print(f"\n{name}: " + ", ".join(f"{k} {v}" for k, v in CONDITIONS[name].items()))
        print(header)
        results = {}
        for command in args.commands:
            results[command] = run(args.policy, args.obs_dim, args.nominal, COMMANDS[command],
                                   args.duration, **CONDITIONS[name])
        falls = [c for c, r in results.items() if r["fell"] is not None]

        def row(label, rs):
            fell = ",".join(f"{r['fell']:.1f}s" for r in rs if r["fell"] is not None) or "-"
            return (f"{label:10s}{fell:>6s}{max(r['tilt'] for r in rs):9.1f}"
                    f"{np.mean([r['tracking'] for r in rs]):9.3f}{np.mean([r['torque'] for r in rs]):7.2f}"
                    f"{max(r['peak_torque'] for r in rs):9.2f}{max(r['joint_speed'] for r in rs):7.1f}"
                    f"{np.mean([r['action_rate'] for r in rs]):6.2f}{np.mean([r['jitter'] for r in rs]):7.2f}"
                    f"{np.mean([r['airborne'] for r in rs]):5.2f}")

        if args.detail:
            for command, r in results.items():
                print(row(command, [r]))
        print(row("worst/mean", list(results.values())))
        verdicts[name] = not falls
        if falls:
            print(f"  FELL on: {', '.join(falls)}")

    print("\n" + "  ".join(f"{n}: {'PASS' if ok else 'FAIL'}" for n, ok in verdicts.items()))


if __name__ == "__main__":
    main()
