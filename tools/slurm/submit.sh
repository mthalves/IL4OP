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

# ask_valid <prompt> <default> <regex> <what is expected> -> the first answer that matches
# (an empty answer is the default). The question is asked again until the answer is valid.
ask_valid() {
    local prompt="$1" default="$2" pattern="$3" expected="$4" answer tries=0
    while true; do
        answer="$(ask_text "$prompt" "$default")"
        [[ "$answer" =~ $pattern ]] && { printf '%s' "$answer"; return 0; }
        printf '\033[1;33m[warn]\033[0m %s\n' "'$answer' is not valid: $expected" >&2
        # an input that has ended answers with the default every time: do not ask for ever
        tries=$((tries + 1))
        [ "$tries" -lt 20 ] || die "no valid answer to '$prompt' after $tries attempts"
    done
}

# ask_confirm <prompt> <default y|n>: only y/yes and n/no are answers; anything else asks again
ask_confirm() {
    local prompt="$1" default="$2" answer hint="[y/N]"
    [ "$default" = "y" ] && hint="[Y/n]"
    while true; do
        read -r -p "$prompt $hint: " answer || { echo; answer=""; }
        case "${answer,,}" in
            "") [ "$default" = "y" ]; return ;;
            y|yes) return 0 ;;
            n|no) return 1 ;;
        esac
        echo "    please answer y or n"
    done
}

ask_yes_no() { ask_confirm "$1" n; }

# the same question for an option that is on unless it is turned off
ask_yes_default() { ask_confirm "$1" y; }

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
USER_NAME="$(ask_valid "1) Your user name" "${USER:-}" '^[A-Za-z0-9_.-]*$' "letters, digits, '_', '.' and '-' only (it goes into the job name)")"
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
while true; do
    FILTER="$(ask_text "   filter, e.g. Go1 or Anymal (empty = list all)" "")"
    FAMILIES=()
    for task in "${ALL_TASKS[@]}"; do
        [ -n "$FILTER" ] && [[ "${task,,}" != *"${FILTER,,}"* ]] && continue
        family="$(family_of "$task")"
        [[ " ${FAMILIES[*]-} " == *" $family "* ]] || FAMILIES+=("$family")
    done
    [ "${#FAMILIES[@]}" -gt 0 ] && break
    warn "no task matches '$FILTER'"
done
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

# ----------------------------------------------------- 6. from scratch, or resume
SEED=""; VIDEO=0; EXTRA=""
OVERRIDES=(); RUN_PARTS=()
RESUME=0; RESUME_PATH=""; RESUME_ITER=""; RESUME_EXPERIMENT=""

title "6) Start from scratch or resume a run?"
ask_menu "   training" "from scratch" "resume a run from one of its checkpoints"
if [ "$MENU_CHOICE" = "2" ]; then
    # a Go1 run only fits the variant and the terrain it was made for: the network has to match
    WANT_CEP="-"; WANT_CCR="-"; WANT_TERRAIN="-"
    if [[ "$TASK" == *Unitree-Go1* ]]; then
        WANT_CEP=false; WANT_CCR=false; WANT_TERRAIN=flat
        [[ "$TASK" == *-CEP-* || "$TASK" == *-ContactFramework-* ]] && WANT_CEP=true
        [[ "$TASK" == *-CCR-* || "$TASK" == *-ContactFramework-* ]] && WANT_CCR=true
        [[ "$TASK" == *-Rough-* ]] && WANT_TERRAIN=rough
    fi
    R_PATH=(); R_LAST=(); R_AIR=(); R_CCR=(); R_SEED=(); R_NAME=(); R_LABEL=()
    while IFS=$'\t' read -r path experiment last air ccr_weight seed cep ccr terrain run_name; do
        if [ "$WANT_CEP" != "-" ]; then
            [ "$cep" = "$WANT_CEP" ] && [ "$ccr" = "$WANT_CCR" ] && [ "$terrain" = "$WANT_TERRAIN" ] || continue
        fi
        R_PATH+=("$path"); R_LAST+=("$last"); R_AIR+=("$air"); R_CCR+=("$ccr_weight"); R_SEED+=("$seed"); R_NAME+=("$run_name")
        R_LABEL+=("${path#logs/rsl_rl/}   (iteration $last, seed $seed)")
    done < <("$ENV_PYTHON" tools/slurm/run_info.py logs/rsl_rl)
    [ "${#R_PATH[@]}" -gt 0 ] || die "no finished run of this task under logs/rsl_rl to resume.

  Runs are looked for in logs/rsl_rl/<experiment>/<run>/model_<n>.pt on this machine; copy the
  run here, or train from scratch."
    echo "   runs that fit $TASK:"
    ask_menu "   run" "${R_LABEL[@]}"
    PICK=$((MENU_CHOICE - 1))
    RESUME=1; RESUME_PATH="${R_PATH[$PICK]}"; RESUME_EXPERIMENT="$(basename "$(dirname "$RESUME_PATH")")"
    while true; do
        RESUME_ITER="$(ask_text "   checkpoint iteration (empty = the last, ${R_LAST[$PICK]})" "")"
        [ -n "$RESUME_ITER" ] || RESUME_ITER="${R_LAST[$PICK]}"
        [[ "$RESUME_ITER" =~ ^[0-9]+$ ]] && [ -f "$RESUME_PATH/model_$RESUME_ITER.pt" ] && break
        warn "there is no model_$RESUME_ITER.pt in $RESUME_PATH (checkpoints are saved every 50 iterations)"
    done
    DEF_MAX_ITERATIONS="$(ask_valid "   MORE iterations to train after it" "10000" '^[1-9][0-9]*$' "a whole number above zero")"
    # the run continues with the settings it was trained with, or the reward would change midway
    [ "${R_SEED[$PICK]}" != "-" ] && SEED="${R_SEED[$PICK]}"
    if [ "$WANT_CEP" != "-" ]; then
        [ "${R_AIR[$PICK]}" != "-" ] && OVERRIDES+=("env.rewards.feet_air_time.weight=${R_AIR[$PICK]}")
        [ "${R_CCR[$PICK]}" != "-" ] && OVERRIDES+=("env.rewards.contact_consistency.weight=${R_CCR[$PICK]}")
        echo "   keeping the run's rewards: feet air time ${R_AIR[$PICK]}, CCR ${R_CCR[$PICK]}, seed ${R_SEED[$PICK]}"
    fi
    [ "${R_NAME[$PICK]}" != "-" ] && RUN_PARTS+=("${R_NAME[$PICK]}")
    RUN_PARTS+=("resume$RESUME_ITER")
    warn "settings given to the first run as extra arguments (agent.algorithm..., env...) are not recorded
         here: repeat them in the extra arguments below"
fi

# ------------------------------------------------------------ 7. advanced options
NUM_ENVS="$DEF_NUM_ENVS"; MAX_ITERATIONS="$DEF_MAX_ITERATIONS"
TIME_LIMIT="$DEF_TIME"; PARTITION="$DEF_PARTITION"; CPUS="$DEF_CPUS"; MEM="$DEF_MEM"

title "7) Advanced options"
if ask_yes_no "   change the training and SLURM defaults?"; then
    echo "   -- training"
    NUM_ENVS="$(ask_valid "   environments to simulate" "$DEF_NUM_ENVS" '^[1-9][0-9]*$' "a whole number above zero")"
    MAX_ITERATIONS="$(ask_valid "   training iterations$([ "$RESUME" = 1 ] && echo " (more, on top of the checkpoint)")" "$DEF_MAX_ITERATIONS" '^[1-9][0-9]*$' "a whole number above zero")"
    SEED="$(ask_valid "   seed (empty = default)" "$SEED" '^[0-9]*$' "a whole number, or empty")"
    if [[ "$TASK" == *Unitree-Go1* ]] && [ "$RESUME" = "0" ]; then
        echo "   -- Go1 task options"
        # the weight the task is written with: "on" must mean a weight above zero, whatever it is
        AIR_WEIGHT="$(sed -nE 's/^FEET_AIR_TIME_WEIGHT[[:space:]]*=[[:space:]]*([0-9.]+).*/\1/p' \
            isaaclab_experiments/go1_locomotion/rough_env_cfg.py | head -1)"
        if ! ask_yes_default "   train with the feet air time (swing time) reward? (weight in the task: ${AIR_WEIGHT:-unknown})"; then
            OVERRIDES+=("env.rewards.feet_air_time.weight=0.0"); RUN_PARTS+=("noairtime")
        elif [ -z "$AIR_WEIGHT" ] || [ "$AIR_WEIGHT" = "0" ] || [ "$AIR_WEIGHT" = "0.0" ]; then
            warn "FEET_AIR_TIME_WEIGHT is ${AIR_WEIGHT:-unreadable} in rough_env_cfg.py, so 'on' would train without it; using 0.25"
            OVERRIDES+=("env.rewards.feet_air_time.weight=0.25")
        fi
        if [[ "$TASK" == *-CCR-* || "$TASK" == *-ContactFramework-* ]]; then
            while true; do
                CCR_WEIGHT="$(ask_text "   weight of the contact consistency reward (CCR)" "0.05")"
                [[ "$CCR_WEIGHT" =~ ^[0-9]*\.?[0-9]+$ ]] && break
                warn "'$CCR_WEIGHT' is not a positive number"
            done
            if [ "$CCR_WEIGHT" != "0.05" ]; then
                OVERRIDES+=("env.rewards.contact_consistency.weight=$CCR_WEIGHT"); RUN_PARTS+=("ccr$CCR_WEIGHT")
            fi
        fi
        echo "   -- general"
    fi
    ask_yes_no "   record a video during training?" && VIDEO=1
    EXTRA="$(ask_text "   extra arguments for the training script (empty = none)" "")"
    echo "   -- slurm"
    TIME_LIMIT="$(ask_valid "   time limit (D-HH:MM:SS)" "$DEF_TIME" '^([0-9]+-)?[0-9]+(:[0-5][0-9]){0,2}$' "minutes, HH:MM:SS, or D-HH:MM:SS")"
    PARTITION="$(ask_valid "   partition" "$DEF_PARTITION" '^[A-Za-z0-9_.,-]+$' "a partition name")"
    CPUS="$(ask_valid "   cpus per task" "$DEF_CPUS" '^[1-9][0-9]*$' "a whole number above zero")"
    MEM="$(ask_valid "   memory" "$DEF_MEM" '^[1-9][0-9]*[KMGT]?$' "a number with an optional unit, e.g. 96G")"
else
    echo "   using the defaults: $NUM_ENVS envs, $MAX_ITERATIONS iterations, $TIME_LIMIT on '$PARTITION'"
fi

# ------------------------------------------------------------------- submission
SHORT_TASK="$(printf '%s' "$TASK" | sed -E 's/^(Isaac|IL4OP)-//; s/-v[0-9]+$//' | tr '[:upper:]' '[:lower:]')"
JOB_NAME="$(printf '%s' "${USER_NAME:+$USER_NAME-}$SHORT_TASK" | cut -c1-60)"

TRAIN_ARGS=(--task "$TASK" --headless --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS")
if [ -n "$SEED" ]; then
    TRAIN_ARGS+=(--seed "$SEED")
    [ "$RESUME" = "1" ] || RUN_PARTS+=("seed$SEED")      # a resumed run carries the name it had
fi
if [ "$RESUME" = "1" ]; then
    # the checkpoint is matched as a regular expression: the whole file name, not a prefix of it
    TRAIN_ARGS+=(--resume --experiment_name "$RESUME_EXPERIMENT" --load_run "$(basename "$RESUME_PATH")"
                 --checkpoint "model_${RESUME_ITER}\\.pt\$")
fi
[ "$VIDEO" = "1" ] && TRAIN_ARGS+=(--video)
# the run folder is "<timestamp>_<run name>": what the options changed is in its name, so runs
# of one task made with different settings or seeds are told apart without opening a config
if [ "${#RUN_PARTS[@]}" -gt 0 ] && [[ "$EXTRA" != *--run_name* ]]; then
    RUN_NAME="$(IFS=_; printf '%s' "${RUN_PARTS[*]}")"
    TRAIN_ARGS+=(--run_name "$RUN_NAME")
fi
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && TRAIN_ARGS+=($EXTRA)
# Hydra overrides of the configuration come last
[ "${#OVERRIDES[@]}" -gt 0 ] && TRAIN_ARGS+=("${OVERRIDES[@]}")

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
[ "$RESUME" = "1" ] && printf '  resumes     : %s at iteration %s, for %s more\n' "$RESUME_PATH" "$RESUME_ITER" "$MAX_ITERATIONS"
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
