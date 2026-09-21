"""MDP terms of the Go2W locomotion task, on top of robot_lab's velocity MDP."""

from robot_lab.tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
