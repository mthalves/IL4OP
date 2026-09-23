#!/usr/bin/env bash
#
# Set up IL4OP: Isaac Sim, the vendored IsaacLab 2.3.2 and this package.
#
#   ./setup.sh                     # create the "IL4OP" conda environment and install everything
#   ./setup.sh --with-robot-lab    # also install robot_lab (needed by the Go2W tasks)
#   ./setup.sh --use-current-env   # install into the environment that is already active
#   ./setup.sh --install-conda     # download and install Miniconda if conda is missing
#   ./setup.sh --dry-run           # only print what would be executed
#
# Do not run this script with sudo: conda and the Python environment must belong
# to your own user account.
#
set -euo pipefail

ENV_NAME="IL4OP"
PYTHON_VERSION="3.11"
# conda-forge only: the Anaconda default channels require their Terms of Service to be
# accepted, which would make a fresh, non-interactive installation fail
CONDA_CHANNEL="conda-forge"
TORCH_VERSION="2.7.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
ISAACSIM_VERSION="5.1.0"
ISAACSIM_INDEX="https://pypi.nvidia.com"
ROBOT_LAB_TAG="v2.3.2"
ROBOT_LAB_URL="https://github.com/fan-ziqi/robot_lab.git"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# the six extensions of the vendored IsaacLab, installed editable from this repository
ISAACLAB_EXTENSIONS=(isaaclab isaaclab_assets isaaclab_contrib isaaclab_mimic isaaclab_rl isaaclab_tasks)

USE_CURRENT_ENV=0
WITH_ROBOT_LAB=0
INSTALL_CONDA=0
ALLOW_ROOT=0
DRY_RUN=0
CONDA_BASE=""
CONDA_WAS_INSTALLED=0
CONDA_ON_PATH=0

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

find_conda_base() {
    if command -v conda >/dev/null 2>&1; then
        conda info --base 2>/dev/null && return 0
    fi
    local prefix
    for prefix in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" "/opt/conda"; do
        if [ -x "$prefix/bin/conda" ]; then
            echo "$prefix"
            return 0
        fi
    done
    return 1
}

install_miniconda() {
    local installer="${TMPDIR:-/tmp}/miniconda-installer.sh"
    log "Installing Miniconda into $HOME/miniconda3"
    command -v curl >/dev/null 2>&1 || die "curl is required to download Miniconda"
    run curl -fsSL "$MINICONDA_URL" -o "$installer"
    if [ "$DRY_RUN" -eq 0 ]; then
        # a truncated download produces a binary that cannot unpack itself later
        local size
        size="$(stat -c %s "$installer")"
        [ "$size" -gt 50000000 ] || die "the Miniconda installer is only $size bytes: the download was interrupted, run the script again"
    fi
    run bash "$installer" -b -p "$HOME/miniconda3"
    run rm -f "$installer"
    CONDA_BASE="$HOME/miniconda3"
    CONDA_WAS_INSTALLED=1

    # the batch installer does not touch the shell configuration, so `conda` would not
    # exist in the user's terminal once this script exits
    local shell_name
    shell_name="$(basename "${SHELL:-bash}")"
    case "$shell_name" in
        bash|zsh|fish) ;;
        *) shell_name="bash" ;;
    esac
    run "$CONDA_BASE/bin/conda" init "$shell_name"
    warn "your ~/.${shell_name}rc was updated by 'conda init $shell_name'; open a new terminal to use conda"
}

# locate conda, check that the installation is usable and load its shell hook
ensure_conda() {
    local base
    if command -v conda >/dev/null 2>&1; then
        CONDA_ON_PATH=1
        # a conda left over from a root install keeps answering on PATH but cannot run
        conda info --base >/dev/null 2>&1 ||
            warn "the 'conda' on your PATH does not work ($(command -v conda)); looking for a usable installation"
    fi
    if base="$(find_conda_base)" && [ -n "$base" ]; then
        CONDA_BASE="$base"
    elif [ "$INSTALL_CONDA" -eq 1 ]; then
        install_miniconda
    else
        die "conda was not found.

  Install Miniconda:
      curl -fsSL $MINICONDA_URL -o /tmp/miniconda.sh
      bash /tmp/miniconda.sh -b -p \$HOME/miniconda3
      \$HOME/miniconda3/bin/conda init bash && exec bash

  or re-run this script with --install-conda,
  or with --use-current-env to install into the active Python environment."
    fi

    local hook="$CONDA_BASE/etc/profile.d/conda.sh"
    [ -f "$hook" ] || die "conda was found at $CONDA_BASE but $hook is missing: the installation looks incomplete"

    # shellcheck disable=SC1090
    source "$hook"
    command -v conda >/dev/null 2>&1 || die "could not initialise conda from $hook"
    if ! conda --version >/dev/null 2>&1; then
        die "'conda --version' failed: the installation at $CONDA_BASE is broken.

  This happens when conda was installed with sudo (the files belong to root) or when
  the installer download was truncated. Remove it and install again as your own user:
      rm -rf $CONDA_BASE        # sudo rm -rf, if it belongs to root
      ./setup.sh --install-conda"
    fi
    echo "    $(conda --version) at $CONDA_BASE"
}

# make sure the interpreter that will receive the packages is the expected one
verify_active_env() {
    command -v python >/dev/null 2>&1 || die "no python on PATH after activating the environment"

    local prefix version
    prefix="$(python -c 'import sys; print(sys.prefix)')"
    version="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

    [ "$version" = "$PYTHON_VERSION" ] || die "the active environment runs Python $version, but IL4OP requires $PYTHON_VERSION"
    python -m pip --version >/dev/null 2>&1 || die "pip is not available in $prefix"
    [ "${CONDA_DEFAULT_ENV:-}" = "base" ] && warn "installing into the conda 'base' environment is not recommended"

    echo "    Python $version at $prefix"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --env) ENV_NAME="$2"; shift 2 ;;
        --use-current-env) USE_CURRENT_ENV=1; shift ;;
        --with-robot-lab) WITH_ROBOT_LAB=1; shift ;;
        --install-conda) INSTALL_CONDA=1; shift ;;
        --allow-root) ALLOW_ROOT=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        *) die "unknown option: $1 (use --help)" ;;
    esac
done

cd "$REPO_ROOT"
[ -d IsaacLab/source/isaaclab ] || die "run this script from the IL4OP repository (IsaacLab/ not found)"

# running as root installs conda into /root, where the user account cannot read it
if [ "$(id -u)" -eq 0 ] && [ "$ALLOW_ROOT" -eq 0 ]; then
    die "do not run this script as root${SUDO_USER:+ (you used sudo as '$SUDO_USER')}.

  conda and the Python environment must belong to your own user account: installed as
  root they land in /root/miniconda3 and fail with errors such as
      Could not load PyInstaller's embedded PKG archive ... (/root/miniconda3/_conda)

  Run it without sudo, or pass --allow-root if you really are root (e.g. in a container)."
fi

# ---------------------------------------------------------------- environment
if [ "$USE_CURRENT_ENV" -eq 1 ]; then
    log "Using the environment that is already active"
    verify_active_env
else
    log "Checking the conda installation"
    ensure_conda

    log "Preparing the '$ENV_NAME' conda environment (Python $PYTHON_VERSION)"
    if conda env list | awk '$1 != "#" {print $1}' | grep -qx "$ENV_NAME"; then
        echo "    environment already exists, reusing it"
    elif ! run conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" -c "$CONDA_CHANNEL" --override-channels; then
        die "could not create the '$ENV_NAME' environment.

  If conda reports that the Terms of Service of the Anaconda channels were not
  accepted, either accept them:
      conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
      conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

  or create the environment yourself from conda-forge and re-run with --use-current-env:
      conda create -y -n $ENV_NAME python=$PYTHON_VERSION -c $CONDA_CHANNEL --override-channels"
    fi

    # `conda activate` needs the hook that ensure_conda already sourced
    if [ "$DRY_RUN" -eq 0 ]; then
        conda activate "$ENV_NAME" || die "could not activate '$ENV_NAME'"
        verify_active_env
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
# isaaclab depends on flatdict, which ships no wheel and whose setup.py imports
# pkg_resources, removed in setuptools 82. pip would build it in an isolated environment
# with the newest setuptools and fail, so build it here against an older one.
flatdict_spec="$(grep -oE '"flatdict[^"]*"' IsaacLab/source/isaaclab/setup.py | tr -d '"' | head -1)"
run python -m pip install "setuptools<82" wheel
run python -m pip install "${flatdict_spec:-flatdict}" --no-build-isolation
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
    if [ "$CONDA_WAS_INSTALLED" -eq 1 ]; then
        cat <<NEXT
    exec \$SHELL                               reload the shell so that 'conda' is available
                                              (or: source $CONDA_BASE/etc/profile.d/conda.sh)
NEXT
    elif [ "$USE_CURRENT_ENV" -eq 0 ] && [ "$CONDA_ON_PATH" -eq 0 ]; then
        warn "conda is installed at $CONDA_BASE but is not on your PATH"
        cat <<NEXT
    $CONDA_BASE/bin/conda init $(basename "${SHELL:-bash}") && exec \$SHELL
NEXT
    fi
    cat <<NEXT
    conda activate $ENV_NAME
    python -m app                             launch a planning experiment from the GUI
    python isaaclab_experiments/planning.py --space discrete --log True

    The first Isaac Sim start downloads shader and asset caches: it can take
    10-20 minutes without printing anything.
NEXT
fi
