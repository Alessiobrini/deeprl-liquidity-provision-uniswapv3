#!/bin/bash
#SBATCH --job-name=uniswap-ppo-budget
#SBATCH --requeue
#SBATCH --array=0-143%48
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/ppo_budget_%A_%a.out
#SBATCH --error=slurm_logs/ppo_budget_%A_%a.err
#
# PPO budget ablation: 3 budgets x 2 pools x 24 rolling windows = 144 units.
# Each array task owns one deterministic unit. The throttle caps concurrent tasks at
# 48; change only the %48 submission throttle if the cluster allocation requires it.
# The small PPO networks and environment are CPU-bound, so this requests no GPU.

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT=${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}
PYTHON=${CONDA_PREFIX:?activate the project environment before sbatch}/bin/python3
OUT=${1:-outputs/ppo_budget_ablation_v1}
N_UNITS=144

cd "$PROJECT"
if [ ! -x "$PYTHON" ]; then
  echo "no interpreter at $PYTHON; activate the project environment" >&2
  exit 1
fi
if [ ! -d "$PROJECT/data/processed" ]; then
  echo "no panel at $PROJECT/data/processed; sync the data bundle first" >&2
  exit 1
fi

echo "host $(hostname) task ${SLURM_ARRAY_TASK_ID}/${N_UNITS} out $OUT"
srun "$PYTHON" \
  -m src.deeprl_liquidity_provision_uniswapv3.experiments.ppo_budget_ablation \
  --out "$OUT" \
  --shard "$SLURM_ARRAY_TASK_ID" \
  --of "$N_UNITS"
