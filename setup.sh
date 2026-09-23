#!/usr/bin/env bash
#
# Set up IL4OP: Isaac Sim, the vendored IsaacLab 2.3.2 and this package.
#
#   ./setup.sh                     # create the "IL4OP" conda environment and install everything
#   ./setup.sh --with-robot-lab    # also install robot_lab (needed by the Go2W tasks)
#   ./setup.sh --use-current-env   # install into the environment that is already active
#   ./setup.sh --dry-run           # only print what would be executed
#
set -euo pipefail

ENV_NAME="IL4OP"
PYTHON_VERSION="3.11"
TORCH_VERSION="2.7.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
ISAACSIM_VERSION="5.1.0"
ISAACSIM_INDEX="https://pypi.nvidia.com"
ROBOT_LAB_TAG="v2.3.2"
ROBOT_LAB_URL="https://github.com/fan-ziqi/robot_lab.git"

# the six extensions of the vendored IsaacLab, installed editable from this repository
ISAACLAB_EXTENSIONS=(isaaclab isaaclab_assets isaaclab_contrib isaaclab_mimic isaaclab_rl isaaclab_tasks)

USE_CURRENT_ENV=0
WITH_ROBOT_LAB=0
DRY_RUN=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    # print the comment block at the top of this file
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
    exit 0
}

log()  { printf '\n\033[1;32m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

run() {
    printf '    $ %s\n' "$*"
    [ "$DRY_RUN" -eq 1 ] || "$@"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --env) ENV_NAME="$2"; shift 2 ;;
        --use-current-env) USE_CURRENT_ENV=1; shift ;;
        --with-robot-lab) WITH_ROBOT_LAB=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        *) die "unknown option: $1 (use --help)" ;;
    esac
done

cd "$REPO_ROOT"
[ -d IsaacLab/source/isaaclab ] || die "run this script from the IL4OP repository (IsaacLab/ not found)"

# ---------------------------------------------------------------- environment
if [ "$USE_CURRENT_ENV" -eq 1 ]; then
    log "Using the active environment: $(python -c 'import sys; print(sys.prefix)' 2>/dev/null || echo unknown)"
else
    command -v conda >/dev/null 2>&1 || die "conda not found; install Miniconda or pass --use-current-env"
    log "Preparing the '$ENV_NAME' conda environment (Python $PYTHON_VERSION)"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "    environment already exists, reusing it"
    else
        run conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
    fi
    # make `conda activate` usable inside a non-interactive shell
    if [ "$DRY_RUN" -eq 0 ]; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
    else
        printf '    $ conda activate %s\n' "$ENV_NAME"
    fi
fi

command -v nvidia-smi >/dev/null 2>&1 || warn "nvidia-smi not found: Isaac Sim needs an NVIDIA GPU with a CUDA 12.8 capable driver"

run python -m pip install --upgrade pip

# ---------------------------------------------------------------- torch first
# installed before Isaac Sim so that pip does not resolve a different CUDA build
log "Installing PyTorch $TORCH_VERSION (CUDA 12.8)"
run python -m pip install "torch==$TORCH_VERSION" torchvision --index-url "$TORCH_INDEX"

# ------------------------------------------------------------------ isaac sim
log "Installing Isaac Sim $ISAACSIM_VERSION"
run python -m pip install "isaacsim[all,extscache]==$ISAACSIM_VERSION" --extra-index-url "$ISAACSIM_INDEX"

# ------------------------------------------------------- vendored IsaacLab
# the repository already contains IsaacLab 2.3.2: install it from source, never from pip
log "Installing the vendored IsaacLab extensions (editable)"
for ext in "${ISAACLAB_EXTENSIONS[@]}"; do
    run python -m pip install -e "IsaacLab/source/$ext"
done

# --------------------------------------------------------------- this project
log "Installing the project dependencies and isaaclab_experiments"
run python -m pip install -r requirements.txt
run python -m pip install -e .

# ------------------------------------------------------- robot_lab (optional)
if [ "$WITH_ROBOT_LAB" -eq 1 ]; then
    log "Installing robot_lab $ROBOT_LAB_TAG (Go2W tasks)"
    if [ -d robot_lab ]; then
        echo "    robot_lab/ already present, reusing it"
    else
        run git clone --branch "$ROBOT_LAB_TAG" --depth 1 "$ROBOT_LAB_URL" robot_lab
    fi
    # editable_mode=compat: the default editable install is shadowed by the robot_lab/
    # directory of this repository, which makes `import robot_lab` resolve to an empty
    # namespace package
    run python -m pip install -e robot_lab/source/robot_lab --no-deps --config-settings editable_mode=compat
fi

# ---------------------------------------------------------------- verification
log "Checking the installation"
run python tools/check_environment.py

if [ "$DRY_RUN" -eq 1 ]; then
    log "Dry run finished, nothing was installed"
else
    log "Done. Next steps"
    cat <<'NEXT'
    conda activate IL4OP                      (if the script created the environment)
    python -m app                             launch a planning experiment from the GUI
    python isaaclab_experiments/planning.py --space discrete --log True

    The first Isaac Sim start downloads shader and asset caches: it can take
    10-20 minutes without printing anything.
NEXT
fi
