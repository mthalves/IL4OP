"""Build MuJoCo scenes for the sim-to-sim validation of the IL4OP policies.

The robot descriptions shipped with unitree_ros (Go1) and robot_lab (Go2W) are URDF
files meant for Gazebo, so they need three fixes before MuJoCo accepts them as a
free-floating robot on a floor:

* the visual meshes are COLLADA, which MuJoCo cannot read -> the URDF importer is told
  to keep only the collision geometry (boxes, cylinders and spheres here);
* ``package://`` paths and, for Go2W, several ``<material>`` elements inside one
  ``<visual>`` are rejected by the parser;
* the root link of a URDF is welded to the world, so the saved MJCF gets a free joint,
  a floor, a light and one torque actuator per joint.

Run it once:  python tools/sim2sim/build_models.py

The descriptions are not part of this repository. Go1 comes from unitree_ros:

    git clone https://github.com/unitreerobotics/unitree_ros.git ~/unitree_ros

which is looked for in a few usual places, or wherever ``$UNITREE_ROS`` says. Go2W comes
from robot_lab, installed next to the repository for its Isaac Lab tasks. Whichever
description is present is built, and the other is reported and skipped.
"""

from __future__ import annotations

import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "models"

def unitree_ros() -> Path:
    """Where unitree_ros is, which is wherever it was cloned."""
    env = os.environ.get("UNITREE_ROS")
    candidates = [Path(env)] if env else []
    candidates += [
        Path.home() / "unitree_ros",
        Path.home() / "unitree_rl_lab/unitree_ros",     # cloned beside unitree_rl_lab
        REPO / "unitree_ros",
        REPO.parent / "unitree_ros",
    ]
    for path in candidates:
        if (path / "robots/go1_description/urdf/go1.urdf").is_file():
            return path
    return candidates[0]        # report this one as missing


ROBOTS = {
    "go1": {
        "urdf": unitree_ros() / "robots/go1_description/urdf/go1.urdf",
        "package": "package://go1_description/meshes/",
        "base_height": 0.40,
        "effort_limit": 23.7,
    },
    "go2w": {
        "urdf": REPO / "robot_lab/source/robot_lab/data/Robots/unitree/go2w_description/urdf/go2w_description.urdf",
        "package": "package://go2w_description/meshes/",
        "base_height": 0.45,
        "effort_limit": 23.5,
    },
}

COMPILER = '<mujoco><compiler meshdir="{meshdir}" strippath="true" discardvisual="true" ' \
           'fusestatic="false" balanceinertia="true"/></mujoco>'


def clean_urdf(path: Path, package_prefix: str) -> str:
    """URDF text MuJoCo can parse: one material per visual, no package:// and a compiler block."""
    text = path.read_text(encoding="utf-8")

    # MuJoCo rejects a second <material> in the same <visual>
    def one_material(match: re.Match) -> str:
        body = re.sub(r"(<material\b.*?</material>)(\s*<material\b.*?</material>)+", r"\1", match.group(1), flags=re.S)
        return body + match.group(2)

    text = re.sub(r"(<visual>.*?)(</visual>)", one_material, text, flags=re.S)
    text = text.replace(package_prefix, "")
    meshdir = (path.parent.parent / "meshes").as_posix()
    return re.sub(r"(<robot[^>]*>)", r"\1\n  " + COMPILER.format(meshdir=meshdir), text, count=1)


#: rotor inertia of the motor seen at the joint, in kg m^2: the inertia of the rotor of
#: the Unitree A1 motor (1.47e-4) times the square of the reduction, 6.33 at the hip and
#: the thigh and 6.33 * 1.55 at the knee, where a belt stage follows the gearbox.
ARMATURE = {"hip": 0.0059, "thigh": 0.0059, "calf": 0.0141}


def to_scene(mjcf: str, base_height: float, effort_limit: float) -> str:
    """Add the free joint, the floor and the actuators the URDF cannot express.

    ``effort_limit`` is what a motor of this robot can deliver; a joint that carries its
    own limit keeps it, which is how the knee of a Go1 gets the 35.55 Nm its belt stage
    buys it instead of the 23.7 Nm of the motor itself.
    """
    root = ET.fromstring(mjcf)

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.001")       # the low-level control of the robot runs at 1 kHz
    option.set("integrator", "implicitfast")

    worldbody = root.find("worldbody")
    robot = worldbody.find("body")
    robot.set("pos", f"0 0 {base_height}")
    robot.insert(0, ET.Element("freejoint", {"name": "floating_base"}))

    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "0 0 0.05",
        "rgba": "0.35 0.37 0.40 1", "friction": "1.0 0.005 0.0001",
    })
    ET.SubElement(worldbody, "light", {"pos": "0 0 3", "dir": "0 0 -1", "directional": "true"})

    # torque actuators: the PD loop of the controller lives in the harness, like on hardware
    actuator = ET.SubElement(root, "actuator")
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name and joint.get("type") not in ("free", "ball"):
            # the rotor of the motor turns with the joint and the URDF says nothing about
            # it, yet it is a large part of what a joint feels: without it the stiff gains
            # of the stand-up controller are unstable at 1 kHz, which hardware does not do
            armature = ARMATURE.get(next((k for k in ARMATURE if k in name), None))
            if armature is not None:
                joint.set("armature", f"{armature:g}")
            limit = effort_limit
            own = joint.get("actuatorfrcrange")
            if own:
                limit = float(own.split()[1])
            ET.SubElement(actuator, "motor", {
                "name": name, "joint": name, "ctrlrange": f"-{limit} {limit}", "gear": "1",
            })
    return ET.tostring(root, encoding="unicode")


HINTS = {
    "go1": "git clone https://github.com/unitreerobotics/unitree_ros.git ~/unitree_ros"
           " (or point $UNITREE_ROS at an existing clone)",
    "go2w": "git clone --branch v2.3.2 https://github.com/fan-ziqi/robot_lab.git"
            " next to this repository",
}


def build(name: str, spec: dict) -> Path:
    urdf = Path(spec["urdf"])
    if not urdf.is_file():
        raise FileNotFoundError(f"{urdf} not found\n        get it with: {HINTS[name]}")

    model = mujoco.MjModel.from_xml_string(clean_urdf(urdf, spec["package"]))
    tmp = tempfile.mktemp(suffix=".xml")
    mujoco.mj_saveLastXML(tmp, model)
    scene = to_scene(Path(tmp).read_text(), spec["base_height"], spec["effort_limit"])
    os.remove(tmp)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}_scene.xml"
    out.write_text(scene)

    check = mujoco.MjModel.from_xml_path(str(out))
    joints = [mujoco.mj_id2name(check, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(check.njnt)]
    print(f"{name:5s} -> {out.relative_to(REPO)}")
    print(f"        nq={check.nq} nv={check.nv} actuators={check.nu} geoms={check.ngeom}")
    print(f"        joints: {[j for j in joints if j != 'floating_base']}")
    return out


if __name__ == "__main__":
    built = 0
    for robot, spec in ROBOTS.items():
        try:
            build(robot, spec)
            built += 1
        except FileNotFoundError as error:
            # one description missing is no reason not to build the other
            print(f"{robot:5s} -- skipped: {error}")
    if not built:
        raise SystemExit("no robot description was found, so no scene could be built")
