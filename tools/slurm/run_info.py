#!/usr/bin/env python3
"""Describe the training runs under logs/rsl_rl, one tab-separated line each, for submit.sh.

    run_info.py [LOGS_DIR]

Fields: run directory, experiment, last checkpoint iteration, weight of the feet air time reward,
weight of the contact consistency reward, seed, cep, ccr, terrain, run name. A field a run does not
say is "-". Only the standard library is used, so any Python on the submit host will do.
"""

import re
import sys
from pathlib import Path


def block_weight(text: str, term: str) -> str:
    """Weight of a reward term in a saved env.yaml, or "-" if the run did not have the term."""
    block = re.search(rf"^  {term}:\n((?:    .*\n)+)", text, re.M)
    if not block:
        return "-"
    weight = re.search(r"weight: (\S+)", block.group(1))
    return weight.group(1) if weight else "-"


def top(text: str, key: str) -> str:
    found = re.search(rf"^{key}: (\S+)", text, re.M)
    return found.group(1) if found and found.group(1) not in ("''", '""') else "-"


def describe(run: Path) -> str | None:
    iterations = [int(m.group(1)) for f in run.glob("model_*.pt") if (m := re.fullmatch(r"model_(\d+)\.pt", f.name))]
    if not iterations:
        return None
    env = (run / "params" / "env.yaml").read_text() if (run / "params" / "env.yaml").is_file() else ""
    agent = (run / "params" / "agent.yaml").read_text() if (run / "params" / "agent.yaml").is_file() else ""
    terrain = re.search(r"terrain_type: (\w+)", env)
    fields = [
        str(run), run.parent.name, str(max(iterations)),
        block_weight(env, "feet_air_time"), block_weight(env, "contact_consistency"),
        top(agent, "seed"), top(env, "cep"), top(env, "ccr"),
        {"generator": "rough", "plane": "flat"}.get(terrain.group(1), "-") if terrain else "-",
        top(agent, "run_name").strip("'\""),
    ]
    return "\t".join(fields)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/rsl_rl")
    for run in sorted(root.glob("*/*")):
        # folders that start with "_" or "." are somebody's scratch space or a backup
        if run.is_dir() and not run.parent.name.startswith(("_", ".")) and run.parent.name != "backup":
            line = describe(run)
            if line:
                print(line)


if __name__ == "__main__":
    main()
