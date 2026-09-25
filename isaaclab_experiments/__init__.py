"""IL4OP experiments.

Importing this package registers every task in gymnasium, so it must be imported
*after* the Isaac Sim application has been launched (the task configurations pull in
``isaaclab``, which needs the running simulator).
"""

__version__ = "1.0.0"

from isaaclab_experiments import anymal_c_planning  # noqa: F401
from isaaclab_experiments import go1_locomotion  # noqa: F401
from isaaclab_experiments import go1_locomotion_contact_consistency  # noqa: F401

# the Go2W tasks depend on robot_lab, which is not part of this repository
try:
    from isaaclab_experiments import go2w_locomotion  # noqa: F401
except ImportError as error:
    print(f"[WARN] Go2W tasks unavailable: {error}")
