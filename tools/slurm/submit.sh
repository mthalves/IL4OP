#!/usr/bin/env bash
#
# Interactive submission of an IL4OP / IsaacLab training job to SLURM.
#
#   ./tools/slurm/submit.sh              ask everything, then submit
#   ./tools/slurm/submit.sh --dry-run    ask everything, print the command, submit nothing
#
# The wizard asks for the user, the task and the conda environment, offers the usual
# training and SLURM options, and finally submits tools/slurm/train.sbatch with a job
# name of the form "<user>-<task>" so that squeue and the logs stay readable. It can also
# send SLURM notifications by e-mail and open TensorBoard on the training directory.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_SCRIPT="tools/slurm/train.sbatch"
DRY_RUN=0

# defaults offered by the advanced options
DEF_NUM_ENVS=4096
DEF_MAX_ITERATIONS=20000
DEF_TIME="1-00:00:00"
DEF_PARTITION="main"
DEF_CPUS=16
DEF_MEM="96G"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
title() { printf '\n\033[1;32m== %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()   { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
[ "${1:-}" = "--help" ] && { sed -n '3,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

cd "$REPO_ROOT"
[ -d isaaclab_experiments ] && [ -d IsaacLab ] || die "run this from the IL4OP repository"
command -v sbatch >/dev/null 2>&1 || die "sbatch not found: this machine is not a SLURM submit host"

# --------------------------------------------------------------------- helpers
# ask_text <prompt> <default> -> answer on stdout
ask_text() {
    local prompt="$1" default="${2:-}" answer=""
    if [ -n "$default" ]; then
        read -r -p "$prompt [$default]: " answer || true
    else
        read -r -p "$prompt: " answer || true
    fi
    printf '%s' "${answer:-$default}"
}

ask_yes_no() {
    local answer=""
    read -r -p "$1 [y/N]: " answer || true
    [[ "${answer,,}" == y* ]]
}

# ask_menu <prompt> <option>... -> index (1-based) of the chosen option in MENU_CHOICE
ask_menu() {
    local prompt="$1"; shift
    local options=("$@") i answer
    for i in "${!options[@]}"; do
        printf '  %2d) %s\n' "$((i + 1))" "${options[$i]}"
    done
    while true; do
        # an aborted or piped-out input must not spin forever
        read -r -p "$prompt [1-${#options[@]}]: " answer || { echo; die "input ended before a choice was made"; }
        if [[ "$answer" =~ ^[0-9]+$ ]] && [ "$answer" -ge 1 ] && [ "$answer" -le "${#options[@]}" ]; then
            MENU_CHOICE="$answer"
            return 0
        fi
        echo "    please answer with a number between 1 and ${#options[@]}"
    done
}

# the task identifiers are read from the sources, so no simulator has to be started
list_il4op_tasks() {
    grep -rhoE '^[A-Z_]+ENV_ID[[:space:]]*=[[:space:]]*"[^"]+"' isaaclab_experiments/*/__init__.py 2>/dev/null |
        sed -E 's/.*"([^"]+)".*/\1/' | grep -v -- '-Play-v0' | sort -u
}

list_isaaclab_tasks() {
    grep -rhoE 'id="[A-Za-z0-9_.-]+-v[0-9]+"' IsaacLab/source/isaaclab_tasks --include=__init__.py 2>/dev/null |
        sed -E 's/id="([^"]+)"/\1/' | grep -v -- '-Play-v0' | sort -u
}

# tasks of the same environment differ only by their terrain, which is offered separately
VARIANTS=("Flat-Z" "Flat" "Rough")

variant_of() {
    local id="$1" v
    for v in "${VARIANTS[@]}"; do
        [[ "$id" == *"-$v-"* ]] && { printf '%s' "$v"; return; }
    done
    printf '%s' "-"
}

family_of() {
    local id="$1" v
    v="$(variant_of "$id")"
    [ "$v" = "-" ] && { printf '%s' "$id"; return; }
    printf '%s' "${id/-$v-/-}"
}

# ----------------------------------------------------------------- 1. the user
title "IL4OP training submission"
USER_NAME="$(ask_text "1) Your user name" "${USER:-}")"
[ -n "$USER_NAME" ] || warn "no user name given: the job name will only carry the task"

while true; do
    MAIL_USER="$(ask_text "   e-mail for SLURM notifications (empty = none)" "")"
    [ -z "$MAIL_USER" ] && break
    [[ "$MAIL_USER" == *@*.* ]] && break
    warn "'$MAIL_USER' does not look like an e-mail address"
done

# ------------------------------------------------------------ 2. task source
title "2) Where does the task come from?"
ask_menu "   source" "isaaclab_experiments (IL4OP tasks)" "IsaacLab (tasks shipped with the simulator)"
if [ "$MENU_CHOICE" = "1" ]; then
    mapfile -t ALL_TASKS < <(list_il4op_tasks)
else
    mapfile -t ALL_TASKS < <(list_isaaclab_tasks)
fi
[ "${#ALL_TASKS[@]}" -gt 0 ] || die "no task found in the sources"

# ------------------------------------------------------- 3. environment
title "3) Which environment? (${#ALL_TASKS[@]} available)"
FILTER="$(ask_text "   filter, e.g. Go1 or Anymal (empty = list all)" "")"
FAMILIES=()
for task in "${ALL_TASKS[@]}"; do
    [ -n "$FILTER" ] && [[ "${task,,}" != *"${FILTER,,}"* ]] && continue
    family="$(family_of "$task")"
    [[ " ${FAMILIES[*]-} " == *" $family "* ]] || FAMILIES+=("$family")
done
[ "${#FAMILIES[@]}" -gt 0 ] || die "no task matches '$FILTER'"
ask_menu "   environment" "${FAMILIES[@]}"
FAMILY="${FAMILIES[$((MENU_CHOICE - 1))]}"

# ------------------------------------------------------------ 4. problem type
MATCHING=()
for task in "${ALL_TASKS[@]}"; do
    [ "$(family_of "$task")" = "$FAMILY" ] && MATCHING+=("$task")
done
if [ "${#MATCHING[@]}" -eq 1 ]; then
    TASK="${MATCHING[0]}"
    echo "   only one variant of this environment: $TASK"
else
    title "4) Which type of problem?"
    LABELS=()
    for task in "${MATCHING[@]}"; do
        LABELS+=("$(variant_of "$task")   ($task)")
    done
    ask_menu "   problem" "${LABELS[@]}"
    TASK="${MATCHING[$((MENU_CHOICE - 1))]}"
fi

# --------------------------------------------------------- 5. conda environment
title "5) Which conda environment?"
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
[ -n "$CONDA_BASE" ] || for prefix in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
    [ -x "$prefix/bin/conda" ] && { CONDA_BASE="$prefix"; break; }
done
[ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || die "conda was not found.

  IL4OP needs its conda environment. Install everything with:
      ./setup.sh --install-conda
  and submit again."
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# name and prefix of every environment ("*" marks the active one, so the path is the last field)
CONDA_ENVS=(); CONDA_PREFIXES=()
while read -r name prefix; do
    CONDA_ENVS+=("$name"); CONDA_PREFIXES+=("$prefix")
done < <(conda env list | awk '$1 !~ /^#/ && NF > 1 {print $1, $NF}')
[ "${#CONDA_ENVS[@]}" -gt 0 ] || die "no conda environment found; create one with ./setup.sh"
ask_menu "   environment" "${CONDA_ENVS[@]}"
CONDA_ENV="${CONDA_ENVS[$((MENU_CHOICE - 1))]}"
CONDA_ENV_PREFIX="${CONDA_PREFIXES[$((MENU_CHOICE - 1))]}"

# asking the environment's own interpreter is instant, unlike "conda run"
echo "   checking that '$CONDA_ENV' has IL4OP installed..."
ENV_PYTHON="$CONDA_ENV_PREFIX/bin/python"
[ -x "$ENV_PYTHON" ] || die "'$CONDA_ENV' has no interpreter at $ENV_PYTHON; recreate it with ./setup.sh --env $CONDA_ENV"
if ! "$ENV_PYTHON" -c \
    "import importlib.metadata as m; m.version('isaaclab'); m.version('isaaclab-experiments')" >/dev/null 2>&1; then
    die "'$CONDA_ENV' does not have IsaacLab and isaaclab_experiments installed.

  Install them into it with:
      ./setup.sh --env $CONDA_ENV
  or activate that environment and run ./setup.sh --use-current-env"
fi
echo "   ok"

# ------------------------------------------------------------ 6. advanced options
NUM_ENVS="$DEF_NUM_ENVS"; MAX_ITERATIONS="$DEF_MAX_ITERATIONS"
TIME_LIMIT="$DEF_TIME"; PARTITION="$DEF_PARTITION"; CPUS="$DEF_CPUS"; MEM="$DEF_MEM"
SEED=""; VIDEO=0; EXTRA=""

title "6) Advanced options"
if ask_yes_no "   change the training and SLURM defaults?"; then
    echo "   -- training"
    NUM_ENVS="$(ask_text "   environments to simulate" "$DEF_NUM_ENVS")"
    MAX_ITERATIONS="$(ask_text "   training iterations" "$DEF_MAX_ITERATIONS")"
    SEED="$(ask_text "   seed (empty = default)" "")"
    ask_yes_no "   record a video during training?" && VIDEO=1
    EXTRA="$(ask_text "   extra arguments for the training script (empty = none)" "")"
    echo "   -- slurm"
    TIME_LIMIT="$(ask_text "   time limit (D-HH:MM:SS)" "$DEF_TIME")"
    PARTITION="$(ask_text "   partition" "$DEF_PARTITION")"
    CPUS="$(ask_text "   cpus per task" "$DEF_CPUS")"
    MEM="$(ask_text "   memory" "$DEF_MEM")"
else
    echo "   using the defaults: $NUM_ENVS envs, $MAX_ITERATIONS iterations, $TIME_LIMIT on '$PARTITION'"
fi

# ------------------------------------------------------------------- submission
SHORT_TASK="$(printf '%s' "$TASK" | sed -E 's/^(Isaac|IL4OP)-//; s/-v[0-9]+$//' | tr '[:upper:]' '[:lower:]')"
JOB_NAME="$(printf '%s' "${USER_NAME:+$USER_NAME-}$SHORT_TASK" | cut -c1-60)"

TRAIN_ARGS=(--task "$TASK" --headless --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS")
[ -n "$SEED" ] && TRAIN_ARGS+=(--seed "$SEED")
[ "$VIDEO" = "1" ] && TRAIN_ARGS+=(--video)
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && TRAIN_ARGS+=($EXTRA)

SBATCH_ARGS=(
    --job-name="$JOB_NAME"
    --partition="$PARTITION"
    --time="$TIME_LIMIT"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --export="ALL,IL4OP_ENV=$CONDA_ENV,IL4OP_PATH=$REPO_ROOT"
)

# the job name already starts with the user name, so %u would repeat it; it is only added
# when the name given here differs from the account the job is submitted with
if [ "$USER_NAME" != "${USER:-}" ]; then
    LOG_STEM="logs/%u-%x-%j"
    LOG_SHOWN="logs/${USER:-user}-$JOB_NAME"
else
    LOG_STEM="logs/%x-%j"
    LOG_SHOWN="logs/$JOB_NAME"
fi
SBATCH_ARGS+=(--output="$LOG_STEM.out" --error="$LOG_STEM.err")

if [ -n "$MAIL_USER" ]; then
    SBATCH_ARGS+=(--mail-user="$MAIL_USER" --mail-type=END,FAIL,TIME_LIMIT)
fi

# a port per user keeps two students on the same node out of each other's way
TB_PORT=$((6006 + $(id -u) % 100))
while ss -ltn 2>/dev/null | grep -q ":$TB_PORT "; do
    TB_PORT=$((TB_PORT + 1))
done
TB_BIN="$CONDA_ENV_PREFIX/bin/tensorboard"

print_monitor() {
    local job_id="$1"
    local log_file="$LOG_SHOWN-$job_id.out"
    title "Follow the run"
    printf '  output      : tail -f %s\n' "$log_file"
    printf '  queue       : squeue -u %s\n' "${USER:-$USER_NAME}"
    printf '  cancel      : scancel %s\n' "$job_id"
    echo
    bold "  TensorBoard"
    if [ -x "$TB_BIN" ]; then
        printf '    on %s : %s --logdir logs/rsl_rl --port %s\n' "$(hostname -s)" "$TB_BIN" "$TB_PORT"
    else
        printf '    tensorboard was not found in the %s environment\n' "$CONDA_ENV"
        printf '    install it with: %s -m pip install tensorboard\n' "$CONDA_ENV_PREFIX/bin/python"
    fi
    printf '    from your computer: ssh -L %s:localhost:%s %s@%s\n' \
        "$TB_PORT" "$TB_PORT" "${USER:-user}" "$(hostname -f 2>/dev/null || hostname)"
    printf '    then open         : http://localhost:%s\n' "$TB_PORT"
}

title "Summary"
printf '  user        : %s\n' "${USER_NAME:-(none)}"
printf '  task        : %s\n' "$TASK"
printf '  conda env   : %s\n' "$CONDA_ENV"
printf '  job name    : %s\n' "$JOB_NAME"
printf '  logs        : %s-<jobid>.out / .err\n' "$LOG_SHOWN"
printf '  notify      : %s\n' "${MAIL_USER:-(none)}"
printf '  slurm       : %s, %s cpus, %s, %s\n' "$PARTITION" "$CPUS" "$MEM" "$TIME_LIMIT"
printf '  training    : %s\n' "${TRAIN_ARGS[*]}"
echo
echo "  sbatch ${SBATCH_ARGS[*]} $SBATCH_SCRIPT ${TRAIN_ARGS[*]}"
echo

mkdir -p logs
if [ "$DRY_RUN" = "1" ]; then
    print_monitor "<jobid>"
    echo
    bold "dry run: nothing was submitted"
    exit 0
fi
ask_yes_no "Submit this job?" || { echo "cancelled"; exit 0; }

JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" "${TRAIN_ARGS[@]}")"
title "Submitted job $JOB_ID"
[ -n "$MAIL_USER" ] && echo "  a notification will be sent to $MAIL_USER when it ends, fails or runs out of time"
print_monitor "$JOB_ID"

echo
if [ -x "$TB_BIN" ] && ask_yes_no "Start TensorBoard now in the background?"; then
    nohup "$TB_BIN" --logdir logs/rsl_rl --port "$TB_PORT" > "logs/tensorboard-$TB_PORT.log" 2>&1 &
    echo "$!" > "logs/tensorboard-$TB_PORT.pid"
    title "TensorBoard running on port $TB_PORT (pid $!)"
    printf '  tunnel it   : ssh -L %s:localhost:%s %s@%s\n' \
        "$TB_PORT" "$TB_PORT" "${USER:-user}" "$(hostname -f 2>/dev/null || hostname)"
    printf '  stop it     : kill $(cat logs/tensorboard-%s.pid)\n' "$TB_PORT"
fi

echo
echo "The first Isaac Sim start on a node builds its shader cache and can take"
echo "10-20 minutes without printing anything."
