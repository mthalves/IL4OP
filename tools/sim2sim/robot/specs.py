"""The two robots this harness knows, and what it takes to drive each of them.

Everything here is read off the robot or its deployment configuration: the joint order
Isaac Lab uses, the poses and gains of the built-in controller, and the observation each
trained policy expects.

A policy is trained over a wide range of commands, ``command_limits``, and the top of that
range is a lot of speed to hand someone at a keyboard: ``drive_limits`` is what driving it
asks for instead, near what the gait of the controller does, so that the robot behaves much
the same whoever has the legs. ``--speed`` opens it up again, never past what was trained.
"""

from __future__ import annotations

from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"

GO1_JOINTS = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
GO1_DEFAULT = {  # UNITREE_GO1_CFG.init_state.joint_pos
    "FL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
}

# Go2W keeps the leg joints first and the four wheels last, in the order robot_lab uses
GO2W_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
]
GO2W_WHEELS = GO2W_JOINTS[12:]
GO2W_DEFAULT = {j: (0.0 if "hip" in j or "foot" in j else 0.8 if "thigh" in j else -1.5) for j in GO2W_JOINTS}

ROBOTS = {
    "go1": {
        "scene": MODELS / "go1_scene.xml",
        "joints": GO1_JOINTS,
        "default": GO1_DEFAULT,
        "action_scale": 0.25,          # actions.joint_pos.scale
        "base_body": "trunk",
        # Isaac Lab drives the Go1 through a learned actuator network; MuJoCo gets the PD
        # gains Unitree publishes for the same joints, which is part of the sim-to-sim gap
        "kp": 20.0,
        "kd": 0.5,
        "decimation": 20,              # 0.001 s * 20 = 50 Hz policy, 1 kHz control
        "fall_height": 0.18,
        "spawn_height": 0.42,
        "wheels": [],
        "obs": [("base_lin_vel", 1.0), ("base_ang_vel", 1.0), ("projected_gravity", 1.0),
                ("velocity_commands", 1.0), ("joint_pos", 1.0), ("joint_vel", 1.0), ("actions", 1.0)],
        # 52 inputs: the contact-framework policy also sees the ContactNet estimate
        "obs_52": [("base_lin_vel", 1.0), ("base_ang_vel", 1.0), ("projected_gravity", 1.0),
                   ("velocity_commands", 1.0), ("joint_pos", 1.0), ("joint_vel", 1.0),
                   ("contact_probabilities", 1.0), ("actions", 1.0)],
        "command_limits": (1.0, 1.0, 1.0),      # the ranges the policy was trained on
        "drive_limits": (0.6, 0.3, 0.8),        # and what driving it by hand asks for
        "height_limits": None,
        # the folded pose and the stand-up routine of the Unitree controller, taken from
        # unitree_rl_lab/deploy/robots/go2/config/config.yaml (FixStand: qs, kp, kd, ts)
        "folded": {j: (0.0 if "hip" in j else 1.36 if "thigh" in j else -2.65) for j in GO1_JOINTS},
        # stand(): a 2 s ramp to [0, 0.67, -1.3] with Kp 70/180/300 and Kd 3/8/15
        "fix_stand": {j: (0.0 if "hip" in j else 0.67 if "thigh" in j else -1.3) for j in GO1_JOINTS},
        "stand_kp": {j: (70.0 if "hip" in j else 180.0 if "thigh" in j else 300.0) for j in GO1_JOINTS},
        "stand_kd": {j: (3.0 if "hip" in j else 8.0 if "thigh" in j else 15.0) for j in GO1_JOINTS},
        "damping_kd": {j: 3.0 for j in GO1_JOINTS},
        # paramInit() holds the measured position with the same gains
        "rest_kp": {j: (70.0 if "hip" in j else 180.0 if "thigh" in j else 300.0) for j in GO1_JOINTS},
        "rest_kd": {j: (3.0 if "hip" in j else 8.0 if "thigh" in j else 15.0) for j in GO1_JOINTS},
        "lying_height": 0.09,
        "stand_waypoints": [("fix_stand", 2.0)],   # moveAllPosition(pos, 2*1000)
    },
    "go2w": {
        "scene": MODELS / "go2w_scene.xml",
        "joints": GO2W_JOINTS,
        "default": GO2W_DEFAULT,
        # hips move half as much as the other leg joints; the wheels are velocity targets
        "action_scale": {**{j: 0.125 if "hip" in j else 0.25 for j in GO2W_JOINTS[:12]},
                         **{j: 5.0 for j in GO2W_WHEELS}},
        "base_body": "base",
        "kp": {**{j: 25.0 for j in GO2W_JOINTS[:12]}, **{j: 0.0 for j in GO2W_WHEELS}},
        "kd": 0.5,
        "decimation": 20,
        "fall_height": 0.20,
        "spawn_height": 0.45,
        "wheels": GO2W_WHEELS,
        # 61 inputs: the flat-z policy also sees its base-height command
        "obs": [("base_lin_vel", 2.0), ("base_ang_vel", 0.25), ("projected_gravity", 1.0),
                ("velocity_commands", 1.0), ("height_command", 1.0), ("joint_pos", 1.0),
                ("joint_vel", 0.05), ("actions", 1.0)],
        # 60 inputs: the rough policy sees the base velocity but has no height command
        "obs_60": [("base_lin_vel", 2.0), ("base_ang_vel", 0.25), ("projected_gravity", 1.0),
                   ("velocity_commands", 1.0), ("joint_pos", 1.0), ("joint_vel", 0.05), ("actions", 1.0)],
        # 57 inputs: the stock flat policy sees neither base velocity nor height
        "obs_57": [("base_ang_vel", 0.25), ("projected_gravity", 1.0), ("velocity_commands", 1.0),
                   ("joint_pos", 1.0), ("joint_vel", 0.05), ("actions", 1.0)],
        "command_limits": (1.5, 1.0, 1.0),
        "drive_limits": (1.0, 0.4, 0.8),
        "height_limits": (0.25, 0.40),
        # same routine, from unitree_rl_lab/deploy/robots/go2w/config/config.yaml
        "folded": {j: (0.0 if "hip" in j or "foot" in j else 1.36 if "thigh" in j else -2.65)
                   for j in GO2W_JOINTS},
        "fix_stand": {j: (0.0 if "hip" in j or "foot" in j else 0.8 if "thigh" in j else -1.5)
                      for j in GO2W_JOINTS},
        # the wheels are never position controlled: the controller only brakes them
        "stand_kp": {j: (0.0 if "foot" in j else 60.0 if "hip" in j else 80.0) for j in GO2W_JOINTS},
        "stand_kd": {j: (3.0 if "foot" in j else 5.0 if "hip" in j else 4.0) for j in GO2W_JOINTS},
        "damping_kd": {j: 3.0 for j in GO2W_JOINTS},
        "rest_kp": {j: (0.0 if "foot" in j else 60.0 if "hip" in j else 80.0) for j in GO2W_JOINTS},
        "rest_kd": {j: (3.0 if "foot" in j else 5.0 if "hip" in j else 4.0) for j in GO2W_JOINTS},
        "lying_height": 0.12,
        "stand_waypoints": [("folded", 1.0), ("fix_stand", 1.0)],
    },
}
