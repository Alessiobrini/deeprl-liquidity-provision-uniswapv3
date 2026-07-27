#!/bin/bash
#SBATCH --job-name=uniswap-rolling
#SBATCH --requeue
#SBATCH --array=0-17
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/roll_%A_%a.out
#SBATCH --error=slurm_logs/roll_%A_%a.err
#
# Walk-forward rolling windows, as a SLURM array.
#
# This is a THIN wrapper. It maps $SLURM_ARRAY_TASK_ID onto the same work-unit index
# a laptop passes with `--shard i --of n`, and runs the identical entry point. There
# is no cluster-only code path, so a laptop shard and a cluster shard are the same
# computation and their outputs merge without reconciliation.
#
#   --array=0-17  with  --of 18   =>  18 shards, 8 work units each on the core panel
#
# CPU, not GPU: measured at ~6,100 steps/s on CPU, and the nets are far too small for
# a GPU to beat kernel-launch overhead. The env also emits float64, which MPS cannot
# take at all. Do not move this to gpu-common; there is nothing to gain and it burns
# the allocation.
#
# Usage, from the project root on DCC:
#   mkdir -p slurm_logs
#   sbatch scripts/slurm/rolling_array.sh outputs/algos_v1 ppo event_driven none default
#   sbatch scripts/slurm/rolling_array.sh outputs/action_geometry_v1 ppo event_driven none default spacing "45 50 55" "480 540 600"
#   # then, once the array drains:
#   python -m src.deeprl_liquidity_provision_uniswapv3.experiments.rolling \
#          --out outputs/rolling_v1 --aggregate
#
# Resumable: a unit whose JSON already exists is skipped, so a requeued or preempted
# task picks up where it stopped rather than redoing the shard.

# NOTE: the #SBATCH lines above are COMMENTS. sbatch parses them itself and the shell
# never expands them, so a ${VAR} there is a literal and the job dies on an unwritable
# log path. They must stay relative (resolved against the submit directory) or be
# hardcoded. Only the lines below this point are shell.
set -euo pipefail
export PYTHONNOUSERSITE=1

# Set these for your own account, or export them before sbatch:
#   PROJECT_ROOT  the project checkout on the cluster
#   CONDA_PREFIX  the conda env (conda activate sets it for you)
PROJECT=${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}
PYTHON=${CONDA_PREFIX:?activate the deeprl-uniswap env before sbatch}/bin/python3
OUT=${1:-outputs/rolling_v1}
ALGO=${2:-ppo}
SCHEDULE=${3:-event_driven}
SHAPING=${4:-none}
EXTRACTOR=${5:-default}
WIDTH_UNITS=${6:-spacing}
WIDTHS_ARG_SET=${7+x}
WIDTHS_TEXT=${7:-"45 50 55"}
EXECUTION_WIDTHS_TEXT=${8:-""}
N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-18}
read -r -a WIDTHS <<< "$WIDTHS_TEXT"
read -r -a EXECUTION_WIDTHS <<< "$EXECUTION_WIDTHS_TEXT"

cd "$PROJECT"

# Absolute paths: a compute node's working directory is not the project root, and the
# panel is resolved relative to the package, so this must be right before anything runs.
if [ ! -x "$PYTHON" ]; then
  echo "no interpreter at $PYTHON; build the env from environment.yml first" >&2
  exit 1
fi
if [ ! -d "$PROJECT/data/processed" ]; then
  echo "no panel at $PROJECT/data/processed; run scripts/sync_to_cluster.sh first" >&2
  exit 1
fi

EXTRA=()
if [ -n "$WIDTHS_ARG_SET" ]; then
  # As in run_algos.sh, preserve the integer-valued argparse default (and therefore
  # the completed collection's resume hash) unless a treatment explicitly supplies
  # widths.  An explicit list is parsed as floats and deliberately gets its own hash.
  EXTRA+=(--widths "${WIDTHS[@]}")
fi
if [ "$EXTRACTOR" = "paper" ]; then
  EXTRA+=(--paper-extractor)
elif [ "$EXTRACTOR" != "default" ]; then
  echo "extractor must be 'default' or 'paper', got '$EXTRACTOR'" >&2
  exit 2
fi
if [ -n "$EXECUTION_WIDTHS_TEXT" ]; then
  EXTRA+=(--execution-widths "${EXECUTION_WIDTHS[@]}")
fi

echo "host $(hostname)  algo ${ALGO}  schedule ${SCHEDULE}  shaping ${SHAPING}  extractor ${EXTRACTOR}  width_units ${WIDTH_UNITS}  widths ${WIDTHS_TEXT}  execution_widths ${EXECUTION_WIDTHS_TEXT:-default}  task ${SLURM_ARRAY_TASK_ID}/${N_SHARDS}  out ${OUT}"

srun "$PYTHON" -m src.deeprl_liquidity_provision_uniswapv3.experiments.rolling \
  --out "$OUT" \
  --algo "$ALGO" \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --of "${N_SHARDS}" \
  --steps 20000 \
  --seeds 42 123 \
  --schedule "$SCHEDULE" \
  --shaping "$SHAPING" \
  --width-units "$WIDTH_UNITS" \
  "${EXTRA[@]}"
