"""Report whether the IL4OP environment is complete.

Versions are read from the package metadata instead of importing the packages,
because ``isaaclab`` and ``isaacsim`` can only be imported once the simulator
application is running. Run it any time with::

    python tools/check_environment.py
"""

import importlib.metadata as metadata
import importlib.util
import sys

# (distribution name, why it is needed, required?)
PACKAGES = [
    ("torch", "simulation and learning backend", True),
    ("isaacsim", "Isaac Sim 5.1", True),
    ("isaaclab", "vendored IsaacLab (IsaacLab/source/isaaclab)", True),
    ("isaaclab_assets", "robot and object assets", True),
    ("isaaclab_contrib", "contributed extensions", True),
    ("isaaclab_mimic", "imitation learning extension", True),
    ("isaaclab_rl", "RL framework wrappers", True),
    ("isaaclab_tasks", "task library", True),
    ("isaaclab-experiments", "this project (pip install -e .)", True),
    ("rsl-rl-lib", "RSL-RL training", True),
    ("skrl", "skrl training", True),
    ("PySide6", "experiment launcher (python -m app)", True),
    ("matplotlib", "result plots", True),
    ("pandas", "result analysis", True),
    ("scipy", "result analysis", True),
    ("robot_lab", "Go2W tasks (optional)", False),
]

EXPECTED_PYTHON = (3, 11)


def main() -> int:
    print(f"{'package':22s} {'version':16s} status")
    print("-" * 72)

    missing_required = []
    for name, purpose, required in PACKAGES:
        try:
            version = metadata.version(name)
            status = "ok"
        except metadata.PackageNotFoundError:
            version = "-"
            if required:
                status = f"MISSING  ({purpose})"
                missing_required.append(name)
            else:
                status = f"not installed  ({purpose})"
        print(f"{name:22s} {version:16s} {status}")

    print("-" * 72)

    python_version = sys.version_info[:2]
    print(f"python                 {'.'.join(map(str, python_version)):16s}", end="")
    print("ok" if python_version == EXPECTED_PYTHON else f"expected {'.'.join(map(str, EXPECTED_PYTHON))}")

    # torch is safe to import and tells us whether the GPU is usable
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            print(f"cuda                   {torch.version.cuda:16s} ok ({torch.cuda.get_device_name(0)})")
        else:
            print(f"cuda                   {'-':16s} NOT AVAILABLE (Isaac Sim needs a GPU)")

    if missing_required:
        print(f"\nMissing required packages: {', '.join(missing_required)}")
        print("Run ./setup.sh (or see the installation section of the README).")
        return 1

    print("\nEnvironment is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
