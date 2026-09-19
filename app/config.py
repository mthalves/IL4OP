"""Export of the launcher selection to the format consumed by ``planning_cfg.py``."""

import json

from app.registry import PLANNING_CFG

# Written next to planning_cfg.py and ignored by git (*.local.json)
LOCAL_CFG = PLANNING_CFG.with_name("planning_cfg.local.json")

ENV_VAR = "IL4OP_PLANNER_CFG"


def write_agent_config(planner_id, parameters, path=LOCAL_CFG):
    """Write ``{"name": ..., "args": {"kwargs": ...}}`` and return the file path."""
    agent = {"name": planner_id, "args": {"kwargs": dict(parameters)}}
    path.write_text(json.dumps(agent, indent=4) + "\n", encoding="utf-8")
    return path
