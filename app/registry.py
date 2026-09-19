"""Discovery of problems and planners from the ``isaaclab_experiments`` sources.

Everything is read statically (``ast``) so the launcher never imports the
simulation stack.
"""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "isaaclab_experiments"
PROBLEMS_DIR = EXPERIMENTS_DIR / "src" / "problems"
PLANNERS_DIR = EXPERIMENTS_DIR / "src" / "planning_algorithms"
PLANNING_CFG = EXPERIMENTS_DIR / "anymal_c_planning" / "agents" / "planning_cfg.py"
SPACE_CFG_DIR = EXPERIMENTS_DIR / "anymal_c_planning" / "configs" / "planning"
PLANNING_SCRIPT = EXPERIMENTS_DIR / "planning.py"


# ----------------------------------------------------------------------------
# Static source inspection
# ----------------------------------------------------------------------------

def _parse(path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None


def _python_files(directory):
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix == ".py" and not p.name.startswith("_")
    )


def read_constant(path, name):
    """Return the literal assigned to ``name`` anywhere in ``path`` (module or class body)."""
    tree = _parse(path)
    if tree is None:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    return None
    return None


def defines_function(path, name):
    tree = _parse(path)
    if tree is None:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in ast.walk(tree)
    )


def read_kwargs(path):
    """Parameters read as ``kwargs.get("name", default)`` inside ``__init__``, as ``{name: default}``."""
    tree = _parse(path)
    if tree is None:
        return {}

    parameters = {}
    for node in ast.walk(tree):
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"):
            continue
        for call in ast.walk(node):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "get"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "kwargs"
                and len(call.args) >= 2
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
            ):
                continue
            try:
                parameters[call.args[0].value] = ast.literal_eval(call.args[1])
            except (ValueError, TypeError):
                continue
    return parameters


# ----------------------------------------------------------------------------
# Registries
# ----------------------------------------------------------------------------

class Problems:
    """Problems under ``src/problems``; each module of a problem package is a scenario.

    ``items[problem_id] = {"name", "path", "scenarios": {scenario_id: {"name", "path"}}}``
    """

    def __init__(self):
        self.items = {}
        for path in sorted(PROBLEMS_DIR.iterdir()):
            if not path.is_dir() or path.name.startswith("_"):
                continue
            scenarios = {
                f.stem: {"name": read_constant(f, "NAME") or f.stem, "path": f}
                for f in _python_files(path)
            }
            self.items[path.name] = {
                "name": read_constant(path / "__init__.py", "NAME") or path.name,
                "path": path,
                "scenarios": scenarios,
            }


class Planners:
    """Planners under ``src/planning_algorithms`` (modules defining a ``plan`` function).

    ``items[planner_id] = {"name", "path", "parameters": {name: default}, "spaces": {space, ...}}``

    Parameter defaults come from the planner's ``__init__`` and are overridden by the
    ``PLANNER_CFG`` catalogue. Space compatibility comes from the
    ``available_planning_algorithms`` registry of each ``configs/planning/<space>.py``.
    """

    def __init__(self):
        catalogue = read_constant(PLANNING_CFG, "PLANNER_CFG") or {}

        spaces = {}
        for path in _python_files(SPACE_CFG_DIR):
            for planner_id in read_constant(path, "available_planning_algorithms") or {}:
                spaces.setdefault(planner_id, set()).add(path.stem)

        self.items = {}
        for path in _python_files(PLANNERS_DIR):
            if not defines_function(path, "plan"):
                continue
            parameters = read_kwargs(path)
            parameters.update(catalogue.get(path.stem, {}))
            self.items[path.stem] = {
                "name": read_constant(path, "NAME") or path.stem,
                "path": path,
                "parameters": parameters,
                "spaces": spaces.get(path.stem, set()),
            }

    def for_space(self, space):
        return {k: v for k, v in self.items.items() if space in v["spaces"]}
