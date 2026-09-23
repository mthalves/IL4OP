#!/bin/bash
#SBATCH --job-name=il4op-train
#SBATCH --partition=main
#SBATCH --gres=gpu:1

# These are intentionally configurable.
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Train an IL4OP policy on the cluster. Every argument is forwarded to the training
# script, so the job is submitted from the repository root as:
#
#   mkdir -p logs
#   sbatch tools/slurm/train.sbatch --task IL4OP-Velocity-Rough-Unitree-Go1-v0 \
#       --headless --num_envs 4096 --max_iterations 20000
#
# Environment variables:
#   IL4OP_PATH          repository root (default: the directory the job was submitted from)
#   IL4OP_ENV           conda environment to activate (default: IL4OP)
#   IL4OP_SKIP_CONDA=1  skip the conda activation, e.g. when modules already provide Python
#   IL4OP_TRAIN_SCRIPT  training entry point (default: isaaclab_experiments/train_rsl_rl.py)
#
# Resume an interrupted run by appending:
#   --resume --load_run <timestamp-dir> --checkpoint model_<n>.pt

set -euo pipefail

IL4OP_PATH="${IL4OP_PATH:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${IL4OP_ENV:-IL4OP}"
TRAIN_SCRIPT="${IL4OP_TRAIN_SCRIPT:-isaaclab_experiments/train_rsl_rl.py}"

if [ ! -d "$IL4OP_PATH/isaaclab_experiments" ] || [ ! -d "$IL4OP_PATH/IsaacLab" ]; then
    echo "ERROR: '$IL4OP_PATH' is not the IL4OP repository; submit from its root or set IL4OP_PATH." >&2
    exit 1
fi
cd "$IL4OP_PATH"

# some IsaacLab tooling expects this variable
export ISAACLAB_PATH="$IL4OP_PATH/IsaacLab"

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "started   : $(date --iso-8601=seconds)"
echo "directory : $PWD"
echo "cpus      : ${SLURM_CPUS_PER_TASK:-?}   memory: ${SLURM_MEM_PER_NODE:-?} MB"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi unavailable"

# ---------------------------------------------------------------- Isaac Lab setup
# add here whatever the cluster needs, for example:
#   module load cuda/12.8
#
# a batch shell does not run the user's rc files, so conda has to be initialised here
if [ "${IL4OP_SKIP_CONDA:-0}" != "1" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -n "$CONDA_BASE" ] || CONDA_BASE="$HOME/miniconda3"
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
    else
        echo "ERROR: conda not found at '$CONDA_BASE'; set IL4OP_ENV/IL4OP_SKIP_CONDA." >&2
        exit 1
    fi
fi
echo "python    : $(command -v python)"
echo

# stream the output into the log instead of buffering it
export PYTHONUNBUFFERED=1

# the first Isaac Sim start of a node writes its shader and asset caches under $HOME
# and can take 10-20 minutes without printing anything
set +e
python "$TRAIN_SCRIPT" "$@"
status=$?
set -e

echo
echo "finished  : $(date --iso-8601=seconds)  (exit $status)"
exit $status
