"""
Planner configuration for the IL4OP experiments.

Select the planning method by setting ``METHOD`` to one of the keys in
``PLANNER_CFG`` and adjust its parameters as needed. Every planner is
constructed as ``Planner(kwargs)``; any parameter left out falls back to
the default defined in the planner's ``__init__``.

Space compatibility:
    discrete   : astar, despot, ibpomcp, pomcp, tbrhopomcp
    continuous : pomcpdpw, pomcpow, pftdpw

The experiment launcher (``python -m app``) exports its selection to a JSON
file and points ``IL4OP_PLANNER_CFG`` at it, which overrides ``AGENT`` below.
"""

import json
import os

PLANNER_CFG = {
    # ---------------------------------------------------------------
    # Discrete planners
    # ---------------------------------------------------------------
    "astar": {},
    "despot": {
        "max_depth": 20,
        "max_it": 1000,
        "discount_factor": 0.95,
        "num_scenarios": 100,
        "lambda_reg": 0.005,
    },
    "ibpomcp": {
        "max_depth": 20,
        "max_it": 1000,
        "q": 0.2,
        "discount_factor": 0.95,
        "particle_revigoration": True,
        "k": 100,
    },
    "pomcp": {
        "max_depth": 20,
        "max_it": 1000,
        "exploration_weight": 0.5,
        "discount_factor": 0.95,
        "particle_revigoration": True,
        "k": 100,
    },
    "tbrhopomcp": {
        "max_depth": 20,
        "max_it": 1000,
        "discount_factor": 0.95,
        "particle_revigoration": True,
        "k": 100,
        "smallbag_size": 10,
        "time_budget": 2.0,
    },
    # ---------------------------------------------------------------
    # Continuous planners
    # ---------------------------------------------------------------
    "pomcpdpw": {
        "max_depth": 20,
        "max_it": 1000,
        "discount_factor": 0.95,
        "particle_revigoration": True,
        "k": 100,
        "c": 50,
        "ka": 15.0,
        "alpha_a": 0.03,
        "ko": 4.0,
        "alpha_o": 0.01,
    },
    "pomcpow": {
        "max_depth": 20,
        "max_it": 1000,
        "discount_factor": 0.95,
        "particle_revigoration": True,
        "k": 100,
        "c": 50,
        "ka": 15.0,
        "alpha_a": 0.03,
        "ko": 4.0,
        "alpha_o": 0.01,
    },
    "pftdpw": {
        "max_depth": 20,
        "max_it": 1000,
        "discount_factor": 0.95,
        "particle_revigoration": True,
        "k": 100,
        "c": 50,
        "ka": 15.0,
        "alpha_a": 0.03,
        "ko": 4.0,
        "alpha_o": 0.01,
    },
}

# Select the planning method here
METHOD = "ibpomcp"

AGENT = {
    "name": METHOD,
    "args": {"kwargs": PLANNER_CFG[METHOD]},
}

_override = os.environ.get("IL4OP_PLANNER_CFG")
if _override:
    with open(_override, encoding="utf-8") as _file:
        AGENT = json.load(_file)
